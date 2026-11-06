"""Greek sentences for TTS, taken from the Greek side of OPUS corpora. Separate from the NMT pipeline.

    python constructors/opus.py --config config/my_config_tts.yaml --corpora TED2020 SETIMES QED wikimedia WikiMatrix
    python preprocessing/tts.py --config config/my_config_tts.yaml --steps all
    python preprocessing/report.py inspect --config config/my_config_tts.yaml --stage tts_clean --corpus QED --reason stress

Steps: `clean` (normalize, strip text nobody would read aloud, filter), `dedup` (one copy per sentence, ignoring
case and punctuation), `export` (tts/{corpus}/clean.csv and needs_tn.csv), `report`.

Sentences with digits, symbols, abbreviations or acronyms need text normalization before they can be spoken, and
Greek makes that ambiguous ("3" is "τρεις" or "τρία" depending on the noun), so they go to their own bucket. Their
`needs_tn` column says why.
"""
import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

import argparse
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xxhash

from config.parser import Parser
from preprocessing import paths, report
from preprocessing.decontaminate import match_key
from preprocessing.dedup import first_occurrences
from preprocessing.normalize import LATIN_LETTER, Normalizer
from preprocessing.shards import MaskFn, read_shard, read_stage, run_stage, write_json

CAPITALS = 'Α-ΩΆΈΉΊΌΎΏΪΫ'
LOWER = 'α-ωάέήίόύώϊϋΐΰ'
LETTERS = CAPITALS + LOWER
ACCENTED = set('ΆΈΉΊΌΎΏάέήίόύώΐΰ')
WORD = re.compile(f'[{LETTERS}]+')
TOKEN = re.compile(f'[{LETTERS}0-9]+')
VOWEL_GROUP = re.compile('[αεηιουωΑΕΗΙΟΥΩϊϋΪΫάέήίόύώΐΰΆΈΉΊΌΎΏ]+')

## Canonical punctuation
APOSTROPHE = re.compile('[΄´’ʼ′‘]')
QUOTE = re.compile('[“”„‟]')
DASH = re.compile('[‐‑‒–—―−]')
REPEATED_DASH = re.compile('-{2,}')
SPACE_BEFORE_PUNCT = re.compile(r'\s+([,.;!:·])')

## Text nobody would read aloud
BRACKETS = re.compile(r'\[[^\[\]]*\]')  # references [1], photo credits [AFP]
STAGE_DIRECTION = re.compile(r'\(\s*(?:γέλι[αο]|χειροκρ[οό]τ\w+|μουσική|βίντεο|αριστερά|δεξιά|κέντρο|(?:στα )?αγγλικά)'
                             r'(?:\s+και\s+\w+)?\s*\)', re.IGNORECASE)
LEADING_MARK = re.compile(r'^(?:-+|>+)\s*')
SPEAKER = re.compile(rf'^(?:[{CAPITALS}]{{1,4}}|Άντρας|Άνδρας|Γυναίκα|Κοινό|Παιδί):\s+')

## Characters
POLYTONIC = re.compile('[ἀ-῿]')
PARENTHESES = re.compile(r'[()]')
TN_SYMBOL = re.compile(r'[%€$£°&+/]')
ALLOWED = re.compile(rf'''^[{LETTERS}0-9\s.,;!·:'"«»\-%€$£°&+/]*$''')

## Wikipedia maintenance notices and hatnotes (often glued to the next sentence), SETIMES bylines
BOILERPLATE = re.compile(r'καλοδεχούμενη|^(?:Κύριο (?:λήμμα|άρθρο)|Δείτε επίσης):|^Αυτό το λήμμα αφορά|συνέβαλ\w+ (?:και \w+ )?στο άρθρο'
                         r'|^Στο άρθρο αυτό συνέβαλ')

## Sentence shape
STARTS_WELL = re.compile(rf'^["«]?[{CAPITALS}0-9]')
ENDS_WELL = re.compile(r'''(?<!\.)[.;!]["»']?$''')  # a trailing ellipsis usually means a cut-off subtitle
MISUSED_SEMICOLON = re.compile(rf';\s+[{LOWER}]')  # in Greek ';' is a question mark; this is a copied English ';'
REPEATED_WORD = re.compile(rf'(?<![{LETTERS}])([{LETTERS}]+)(?: \1){{2,}}(?![{LETTERS}])', re.IGNORECASE)

## Text normalization needed
DIGIT = re.compile(r'\d')
DOTTED = re.compile(rf'(?<![{LETTERS}.])(?:[{LETTERS}]{{1,2}}\.){{2,}}')  # π.χ. Η.Π.Α.
INITIAL = re.compile(rf'(?<![{LETTERS}.])[{CAPITALS}]\.(?!$)')  # Κ. Καραμανλής
ORDINAL = re.compile(rf"(?<![{LETTERS}])[ΑΒΓΔΕΖΗΙΛΞΟΠΡΥΦΧΨΩ]'")  # Γ' Παγκόσμιος, but not the elided Σ' or Τ'
BEFORE_LOWERCASE = re.compile(rf'(?<![{LETTERS}.])[{LETTERS}]{{1,5}}\.(?=\s*[{LOWER},0-9])')  # χλμ. από, 2.250 μ., που
SHORT_FORMS = 'κ|κα|δρ|βλ|σελ|αρ|αριθ|τηλ|χλμ|εκατ|δισ|χιλ|δηλ|γενν|γεν|απεβ|περ|πχ|επιμ|αι|εκ'
LEXICON = re.compile(rf'(?<![{LETTERS}.])(?:{SHORT_FORMS})\.', re.IGNORECASE)
UNDOTTED_FORMS = {'εκατ', 'δισ', 'χιλ', 'απεβ', 'γενν', 'αριθ', 'επιμ'}
ENCLITICS = {'μου', 'σου', 'του', 'της', 'με', 'σε', 'τον', 'την', 'τη', 'το', 'μας', 'σας', 'τους', 'τα', 'τις', 'των'}


def strip_non_speech(text: str) -> str:
    text = APOSTROPHE.sub("'", text)
    text = QUOTE.sub('"', text)
    text = DASH.sub('-', text).replace('…', '...').replace('?', ';')
    text = REPEATED_DASH.sub('-', text)
    text = BRACKETS.sub(' ', text)
    text = STAGE_DIRECTION.sub(' ', text).strip()
    text = LEADING_MARK.sub('', text)
    text = SPEAKER.sub('', text)
    return SPACE_BEFORE_PUNCT.sub(r'\1', ' '.join(text.split()))


def is_acronym(word: str) -> bool:
    # "ΟΗΕ" and "πΓΔΜ", but not the name "ΜακΚλάναχαν"
    n_upper = sum(c.isupper() for c in word)
    return n_upper >= 2 and n_upper > len(word) / 2


def has_no_vowel(word: str) -> bool:
    return len(word) >= 2 and not VOWEL_GROUP.search(word)


def stress_error(text: str) -> bool:
    """A word of two or more syllables needs exactly one accent; a second one is only right before an enclitic.

    Vowel runs count as one syllable, so "για" and "ποιος" need no accent, at the price of missing some real errors.
    """
    matches = list(WORD.finditer(text))
    for i, match in enumerate(matches):
        word = match.group(0)
        if is_acronym(word) or word.lower() in UNDOTTED_FORMS:
            continue
        before = text[match.start() - 1:match.start()]
        after = text[match.end():match.end() + 1]
        # prefixes of hyphenated compounds, "17χρονος" and elided forms ("'λεγε", "ειν'") can correctly lack an accent
        if before in ('-', "'") or before.isdigit() or after in ('-', "'") or after.isdigit():
            continue
        accents = sum(c in ACCENTED for c in word)
        if accents == 0 and len(VOWEL_GROUP.findall(word)) >= 2:
            return True
        if accents > 2:
            return True
        if accents == 2:
            following = matches[i + 1].group(0).lower() if i + 1 < len(matches) else ''
            if following not in ENCLITICS:
                return True
    return False


def needs_tn(text: str) -> str:
    """Comma-separated reasons why this sentence needs text normalization; '' when it can be read as written."""
    words = WORD.findall(text)
    flags = []
    if DIGIT.search(text):
        flags.append('digits')
    if TN_SYMBOL.search(text):
        flags.append('symbols')
    if any(is_acronym(w) for w in words):
        flags.append('acronym')
    if (DOTTED.search(text) or INITIAL.search(text) or ORDINAL.search(text) or BEFORE_LOWERCASE.search(text)
            or LEXICON.search(text) or any(has_no_vowel(w) or w.lower() in UNDOTTED_FORMS for w in words)):
        flags.append('abbreviation')
    return ','.join(flags)


def check(text: str, min_words: int, max_words: int) -> Tuple[str, str]:
    """Returns (removal reason or '', text-normalization flags). A sentence gets the first rule it fails."""
    if not text:
        return 'empty', ''
    if LATIN_LETTER.search(text):
        return 'latin', ''
    if POLYTONIC.search(text):
        return 'polytonic', ''
    if PARENTHESES.search(text):
        return 'parentheses', ''
    if not ALLOWED.match(text):
        return 'other_chars', ''
    n_words = len(TOKEN.findall(text))
    if n_words < min_words or n_words > max_words:
        return 'length', ''
    if not STARTS_WELL.match(text) or not ENDS_WELL.search(text):
        return 'fragment', ''
    if text.count('"') % 2 or text.count('«') != text.count('»'):
        return 'unbalanced_quotes', ''
    words = WORD.findall(text)
    if sum(is_acronym(w) for w in words) > len(words) / 2:
        return 'all_caps', ''
    if BOILERPLATE.search(text):
        return 'boilerplate', ''
    if MISUSED_SEMICOLON.search(text):
        return 'semicolon', ''
    if REPEATED_WORD.search(text):
        return 'repeated_word', ''
    if stress_error(text):
        return 'stress', ''
    return '', needs_tn(text)


class GreekCleaner:
    def __init__(self, normalize_args, min_words: int, max_words: int):
        self.normalizer = Normalizer(normalize_args)
        self.min_words = min_words
        self.max_words = max_words

    def text(self, text: str) -> str:
        return strip_non_speech(self.normalizer.text(text, 'el'))

    def __call__(self, df: pd.DataFrame):
        el = [self.text(t) for t in df['el']]
        results = [check(t, self.min_words, self.max_words) for t in el]
        out = pd.DataFrame({'el': el,
                            'needs_tn': [r[1] for r in results],
                            'corpus': df['corpus'].to_numpy(),
                            'doc_id': df['doc_id'].to_numpy(),
                            'line_id': df['line_id'].to_numpy()})
        return out, np.array([r[0] for r in results], dtype=object)


def clean(conf, corpora: List[str], overwrite: bool = False):
    step = GreekCleaner(conf.filter_args.normalize, conf.tts_args.min_words, conf.tts_args.max_words)
    return run_stage(conf, 'tts_clean', 'raw', corpora, lambda corpus, i, src: step, overwrite=overwrite)


def hash_shard(path: str) -> np.ndarray:
    texts = read_shard(path, columns=['el'])['el']
    return np.fromiter((xxhash.xxh64_intdigest(match_key(t)) for t in texts), np.uint64, len(texts))


def dedup(conf, corpora: List[str], overwrite: bool = False):
    # Corpora come in priority order, so the first copy of a sentence decides which corpus it is credited to.
    files = [f for c in corpora for f in paths.shard_files(paths.stage_dir(conf, 'tts_clean', c))]
    with ProcessPoolExecutor(max_workers=conf.num_workers) as executor:
        hashed = list(executor.map(hash_shard, files, chunksize=1))

    reasons = np.full(sum(len(h) for h in hashed), 'duplicate', dtype=object)
    reasons[first_occurrences(np.concatenate(hashed))] = ''
    ends = np.cumsum([len(h) for h in hashed])
    shard_reasons = dict(zip(files, np.split(reasons, ends[:-1])))

    # Global step: which copy survives depends on every corpus, so it is always recomputed.
    return run_stage(conf, 'tts_dedup', 'tts_clean', corpora,
                     lambda corpus, i, src: MaskFn(shard_reasons[src]), overwrite=True)


def export(conf, corpora: List[str], overwrite: bool = False):
    composition, reasons = {'clean': {}, 'needs_tn': {}}, Counter()
    for corpus in corpora:
        df = read_stage(conf, 'tts_dedup', corpus)
        is_clean = df['needs_tn'] == ''
        folder = paths.tts_dir(conf, corpus)
        os.makedirs(folder, exist_ok=True)
        df[is_clean][['line_id', 'el']].to_csv(os.path.join(folder, 'clean.csv'), index=False)
        df[~is_clean][['line_id', 'el', 'needs_tn']].to_csv(os.path.join(folder, 'needs_tn.csv'), index=False)

        composition['clean'][corpus] = int(is_clean.sum())
        composition['needs_tn'][corpus] = int((~is_clean).sum())
        reasons.update(df[~is_clean]['needs_tn'].str.split(',').explode().tolist())
        print(f'[export] {corpus}: clean {is_clean.sum():,}, needs_tn {(~is_clean).sum():,}')

    composition['needs_tn_reasons'] = dict(reasons.most_common())
    write_json(composition, os.path.join(paths.tts_dir(conf), '_composition.json'))


STEPS = {
    'clean': clean,
    'dedup': dedup,
    'export': export,
    'report': lambda conf, corpora, overwrite: report.table(conf, corpora, stages=['raw'] + paths.TTS_STAGES,
                                                            out=paths.tts_dir(conf, 'reports')),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--steps', nargs='+', default=['all'], choices=['all'] + list(STEPS))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    conf = Parser().get_args(args.exp, args.config)
    corpora = conf.get_corpora(conf.tts_args.corpora)
    for step in list(STEPS) if 'all' in args.steps else [s for s in STEPS if s in args.steps]:
        print(f'\n===== {step} =====')
        STEPS[step](conf, corpora, overwrite=args.overwrite)
