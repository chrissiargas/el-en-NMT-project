"""Step 2: normalization. Keeps casing and punctuation; never lowercases or strips accents.

NFC already maps the Greek question mark (U+037E) to ';', oxia to tonos, and letter + combining acute to
the precomposed letter. The Greek-specific fixes below cover what NFC leaves alone.
"""
import html
import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

GREEK_LETTERS = 'ΆΈ-ώἀ-ῼ'
GREEK_LETTER = re.compile(f'[{GREEK_LETTERS}]')
LATIN_LETTER = re.compile(r'[A-Za-zÀ-ɏ]')

HTML_TAG = re.compile(r'</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>')
CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
# soft hyphen, zero-width space/joiners, bidi marks and embeddings, word joiner, BOM
INVISIBLE = re.compile('[­​-‏‪-‮⁠-⁤﻿]')
WHITESPACE = re.compile(r'\s+')

# "΄Αρθρο" -> "Άρθρο": a spacing tonos or acute typed before a word-initial capital vowel.
SPACING_TONOS = re.compile(r'(?<!\w)[΄´]([ΑΕΗΙΟΥΩ])')
TONOS_CAPITALS = dict(zip('ΑΕΗΙΟΥΩ', 'ΆΈΉΊΌΎΏ'))

WORD = re.compile(r'\w+')
LATIN_TO_GREEK = dict(zip('ABEZHIKMNOPTXYovuµ', 'ΑΒΕΖΗΙΚΜΝΟΡΤΧΥονυμ'))
GREEK_TO_LATIN = {g: l for l, g in LATIN_TO_GREEK.items() if l != 'µ'}

GREEK_WORD = re.compile(f'[{GREEK_LETTERS}]+')
APOSTROPHES = "'’ʼ"


def fix_homoglyphs(text: str, lang: Optional[str] = None) -> str:
    """Within a mixed-script word, swap the minority script's look-alike letters into the majority script.

    On a tie ("Tο") the side's own script wins: Greek for lang='el', Latin for lang='en'.
    """

    def fix(match):
        word = match.group(0)
        n_greek = len(GREEK_LETTER.findall(word))
        n_latin = len(LATIN_LETTER.findall(word))
        to_greek = n_greek > n_latin or (n_greek == n_latin and lang == 'el')
        to_latin = n_latin > n_greek or (n_greek == n_latin and lang == 'en')
        if n_greek and n_latin and to_greek:
            latin = [c for c in word if LATIN_LETTER.match(c)]
            if all(c in LATIN_TO_GREEK for c in latin):
                return ''.join(LATIN_TO_GREEK.get(c, c) for c in word)
        elif n_greek and n_latin and to_latin:
            greek = [c for c in word if GREEK_LETTER.match(c)]
            if all(c in GREEK_TO_LATIN for c in greek):
                return ''.join(GREEK_TO_LATIN.get(c, c) for c in word)
        elif n_greek and 'µ' in word:
            return word.replace('µ', 'μ')
        return word

    return WORD.sub(fix, text)


def fix_final_sigma(text: str) -> str:
    """σ at the end of a Greek word becomes ς, and ς inside a word becomes σ.

    Left alone: single letters, elisions such as σ' and abbreviations of up to 3 letters before a period.
    """

    def fix(match):
        word = match.group(0)
        if len(word) < 2:
            return word
        after = text[match.end():match.end() + 1]
        body = word[:-1].replace('ς', 'σ')
        last = word[-1]
        if last == 'σ' and not after.isalnum() and after not in APOSTROPHES and not (after == '.' and len(word) <= 3):
            last = 'ς'
        return body + last

    return GREEK_WORD.sub(fix, text)


def normalize(text: str,
              lang: Optional[str] = None,
              spacing_tonos: bool = True,
              homoglyphs: bool = True,
              final_sigma: bool = True) -> str:
    if not text:
        return ''

    text = HTML_TAG.sub(' ', html.unescape(text))
    text = unicodedata.normalize('NFC', text)
    text = CONTROL.sub(' ', text)
    text = INVISIBLE.sub('', text)

    if spacing_tonos:
        text = SPACING_TONOS.sub(lambda m: TONOS_CAPITALS[m.group(1)], text)
    if homoglyphs:
        text = fix_homoglyphs(text, lang)
    if final_sigma:
        text = fix_final_sigma(text)

    return WHITESPACE.sub(' ', text).strip()


class Normalizer:
    """Pipeline step: normalize both sides and drop pairs left with an empty side."""

    def __init__(self, args):
        self.kwargs = dict(spacing_tonos=args.fix_spacing_tonos,
                           homoglyphs=args.fix_homoglyphs,
                           final_sigma=args.fix_final_sigma)

    def text(self, text: str, lang: str) -> str:
        return normalize(text, lang, **self.kwargs)

    def __call__(self, df: pd.DataFrame):
        df = df.copy()
        df['el'] = [self.text(t, 'el') for t in df['el']]
        df['en'] = [self.text(t, 'en') for t in df['en']]
        reasons = np.where((df['el'] == '') | (df['en'] == ''), 'empty', '').astype(object)
        return df, reasons


def run(conf, corpora, overwrite: bool = False):
    from preprocessing.shards import run_stage

    step = Normalizer(conf.filter_args.normalize)
    return run_stage(conf, 'normalize', 'raw', corpora, lambda corpus, i, src: step, overwrite=overwrite)


if __name__ == "__main__":
    for sentence in ['΄Αρθρο 5: Η εφαρµογή του κανονισμου', 'Tο κείμενo είναι σωστό;',
                     'Ο αριθμοσ των ατομων σ\' αυτή την πολη', 'βλ. σσ. 12-14 &amp; <b>Πίνακας</b>',
                     'The Οlympic Games​ took place in Athens;']:
        print(repr(sentence), '->', repr(normalize(sentence, 'en' if sentence.startswith('The') else 'el')))
