"""Step 1: carve in-domain held-out test sets out of the raw corpora before any processing.

Units are blocks of `block_size` consecutive lines that never cross an OPUS document (from the `.ids`
file). When a corpus ships as a single document (SETIMES, TED2020) the blocks are simply consecutive
lines. Whole units are drawn at random until `size` usable pairs are collected. The held-out sentences
leave the training data in the decontamination step, which matches on normalized text.
"""
import os

import numpy as np
import pandas as pd

from preprocessing import paths
from preprocessing.normalize import Normalizer
from preprocessing.shards import read_stage, write_jsonl


def units(df: pd.DataFrame, block_size: int) -> pd.Series:
    if df['doc_id'].notna().any():
        doc = df['doc_id'].fillna('')
        position = df.groupby(doc, sort=False).cumcount()
        return doc + '#' + (position // block_size).astype(str)
    return (df['line_id'] // block_size).astype(str)


def carve(conf, corpus: str) -> pd.DataFrame:
    h, rules = conf.filter_args.holdout, conf.filter_args.rules
    df = read_stage(conf, 'raw', corpus, columns=['el', 'en', 'doc_id', 'line_id'])

    normalizer = Normalizer(conf.filter_args.normalize)
    el = [normalizer.text(t, 'el') for t in df['el']]
    en = [normalizer.text(t, 'en') for t in df['en']]
    n_words = lambda texts: np.array([len(t.split()) for t in texts])
    usable = (n_words(el) >= 1) & (n_words(en) >= 1) & (n_words(el) <= rules.max_words) & \
             (n_words(en) <= rules.max_words) & (np.array(el) != np.array(en))

    df['unit'] = units(df, h.block_size)
    usable_per_unit = pd.Series(usable).groupby(df['unit'].values, sort=False).sum()

    rng = np.random.default_rng(conf.seed)
    order = usable_per_unit.index[rng.permutation(len(usable_per_unit))]
    cumulative = usable_per_unit[order].cumsum().values
    n_units = int(np.searchsorted(cumulative, h.size) + 1)
    chosen = set(order[:n_units])

    picked = df['unit'].isin(chosen).values & usable
    test = df[picked].sort_values('line_id')
    print(f'[holdout] {corpus}: {len(test):,} pairs from {n_units} units '
          f'({df.loc[picked, "doc_id"].nunique()} documents)')
    return test


def run(conf, corpora=None, overwrite: bool = False):
    for corpus in conf.filter_args.holdout.corpora:
        out = paths.eval_file(conf, f'heldout_{corpus}')
        if os.path.exists(out) and not overwrite:
            print(f'[holdout] {out} exists, skipping')
            continue

        test = carve(conf, corpus)
        write_jsonl([{'id': f'{corpus}-{int(r.line_id)}', 'el': r.el, 'en': r.en, 'origin': None,
                      'corpus': corpus, 'doc_id': r.doc_id}
                     for r in test.itertuples()], out)
