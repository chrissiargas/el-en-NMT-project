"""Step 1b (plan 3.1 and 4.3): remove every training pair that overlaps a dev/test sentence.

A pair is removed when either side equals an eval sentence after normalization, lowercasing and dropping
punctuation, or when it shares a word n-gram (13 by default) with any eval sentence. It runs after
normalization so that formatting differences do not hide matches.
"""
import os
import pickle
import re
from typing import Dict, List

import pandas as pd
import xxhash

from preprocessing import paths
from preprocessing.normalize import Normalizer
from preprocessing.shards import read_jsonl

PUNCT = re.compile(r'[^\w\s]|_')
WHITESPACE = re.compile(r'\s+')

_INDEX_CACHE: Dict[str, Dict] = {}


def match_key(text: str) -> str:
    return WHITESPACE.sub(' ', PUNCT.sub(' ', text.lower())).strip()


def ngram_hashes(tokens: List[str], n: int):
    return {xxhash.xxh64_intdigest(' '.join(tokens[i:i + n])) for i in range(len(tokens) - n + 1)}


def eval_names(conf) -> List[str]:
    return list(conf.eval_sets) + paths.heldout_names(conf)


def build_index(conf) -> str:
    d = conf.filter_args.decontaminate
    index = {'el': set(), 'en': set(), 'ngrams': set(), 'n': d.ngram}
    normalizer = Normalizer(conf.filter_args.normalize)

    for name in eval_names(conf):
        path = paths.eval_file(conf, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} is missing. Build it first (constructors/ or the holdout step); '
                                    f'decontamination must see every eval set.')
        for row in read_jsonl(path):
            for lang in ('el', 'en'):
                key = match_key(normalizer.text(row[lang], lang))
                tokens = key.split()
                if len(tokens) >= d.min_words:
                    index[lang].add(key)
                index['ngrams'] |= ngram_hashes(tokens, d.ngram)

    out = os.path.join(paths.root(conf), 'eval', '_decontamination_index.pkl')
    with open(out, 'wb') as f:
        pickle.dump(index, f)
    print(f'[decontaminate] index: {len(index["el"]):,} el / {len(index["en"]):,} en sentences, '
          f'{len(index["ngrams"]):,} {d.ngram}-grams from {eval_names(conf)}')
    return out


def load_index(path: str) -> Dict:
    if path not in _INDEX_CACHE:
        with open(path, 'rb') as f:
            _INDEX_CACHE[path] = pickle.load(f)
    return _INDEX_CACHE[path]


class Decontaminator:
    def __init__(self, index_path: str):
        self.index_path = index_path

    def reason(self, el: str, en: str, index: Dict) -> str:
        key_el, key_en = match_key(el), match_key(en)
        if key_el in index['el'] or key_en in index['en']:
            return 'exact_match'
        for key in (key_el, key_en):
            if not ngram_hashes(key.split(), index['n']).isdisjoint(index['ngrams']):
                return 'ngram_overlap'
        return ''

    def __call__(self, df: pd.DataFrame):
        index = load_index(self.index_path)
        return df, [self.reason(el, en, index) for el, en in zip(df['el'], df['en'])]


def run(conf, corpora, overwrite: bool = False):
    from preprocessing.shards import run_stage

    step = Decontaminator(build_index(conf))
    return run_stage(conf, 'decontaminate', 'normalize', corpora, lambda corpus, i, src: step, overwrite=overwrite)
