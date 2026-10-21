"""Step 4: rule-based filters. Each removed pair gets the first rule it fails, in the order below.

Word counts are stored as `el_words` / `en_words`; the set builders use them for length windows.
"""
import re
from typing import Set

import numpy as np
import pandas as pd

from preprocessing.normalize import GREEK_LETTERS

LETTER = re.compile(r'[^\W\d_]')
GREEK = re.compile(f'[{GREEK_LETTERS}]')
LATIN = re.compile(r'[A-Za-zÀ-ɏ]')
NON_ALPHA = re.compile(r'[^\w\s]|[\d_]')
# Greek writes 1.234,5 where English writes 1,234.5, so separators are dropped before comparing.
NUMBER = re.compile(r'\d+(?:[.,]\d+)*')
URL = re.compile(r'(?:https?://|www\.)[^\s<>"]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+', re.IGNORECASE)


def numbers(text: str) -> Set[str]:
    return {re.sub(r'[.,]', '', n).lstrip('0') or '0' for n in NUMBER.findall(text)}


def urls(text: str) -> Set[str]:
    return {u.rstrip('.,;:!?)\'"').lower() for u in URL.findall(text)}


class RuleFilter:
    def __init__(self, args):
        self.a = args
        self.repeat = re.compile(r'\b(\w+)\b(?:\s+\1\b){%d,}' % (args.max_repeat - 1), re.IGNORECASE)

    def script_share(self, text: str, script: re.Pattern) -> float:
        letters = len(LETTER.findall(text))
        return len(script.findall(text)) / letters if letters else 0.0

    def non_alpha_share(self, text: str) -> float:
        chars = len(text) - text.count(' ')
        return len(NON_ALPHA.findall(text)) / chars if chars else 1.0

    def reason(self, el: str, en: str, n_el: int, n_en: int) -> str:
        a = self.a
        if '�' in el or '�' in en:
            return 'replacement_char'
        if not (a.min_words <= n_el <= a.max_words and a.min_words <= n_en <= a.max_words):
            return 'length'
        if el.casefold() == en.casefold():
            return 'identical'
        if not a.min_char_ratio <= len(el) / len(en) <= a.max_char_ratio:
            return 'char_ratio'
        if not LETTER.search(el) or not LETTER.search(en):
            return 'no_letters'
        if max(self.non_alpha_share(el), self.non_alpha_share(en)) > a.max_non_alpha:
            return 'non_alpha'
        if self.script_share(el, GREEK) < a.min_greek:
            return 'script_el'
        if self.script_share(en, LATIN) < a.min_latin:
            return 'script_en'
        if self.repeat.search(el) or self.repeat.search(en):
            return 'repeated_token'
        if a.match_numbers and numbers(el) != numbers(en):
            return 'numbers'
        if a.match_urls and urls(el) != urls(en):
            return 'urls'
        return ''

    def __call__(self, df: pd.DataFrame):
        df = df.copy()
        df['el_words'] = np.array([len(t.split()) for t in df['el']], dtype=np.int32)
        df['en_words'] = np.array([len(t.split()) for t in df['en']], dtype=np.int32)
        reasons = [self.reason(el, en, n_el, n_en)
                   for el, en, n_el, n_en in zip(df['el'], df['en'], df['el_words'], df['en_words'])]
        return df, reasons


def run(conf, corpora, overwrite: bool = False):
    from preprocessing.shards import run_stage

    step = RuleFilter(conf.filter_args.rules)
    return run_stage(conf, 'rules', 'dedup', corpora, lambda corpus, i, src: step, overwrite=overwrite)
