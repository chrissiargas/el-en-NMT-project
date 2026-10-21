"""Step 3: exact dedup across all corpora, then cap repeated sources.

Corpora are processed in priority order (tier, then config/corpora.yaml order), so the first copy of a
duplicated pair is the curated one. Each pair and each side is reduced to a 64-bit hash, so the global
step works on numpy arrays: about 24 bytes per row, i.e. ~3.6 GB for 150M rows.

The cap applies to both sides, because each set serves both directions: at most `max_translations`
distinct translations per identical Greek sentence and per identical English sentence.
"""
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
import xxhash

from preprocessing import paths
from preprocessing.shards import MaskFn, read_shard, run_stage


def hash_shard(path: str):
    df = read_shard(path, columns=['el', 'en'])
    n = len(df)
    pair = np.fromiter((xxhash.xxh64_intdigest(a + '\t' + b) for a, b in zip(df['el'], df['en'])), np.uint64, n)
    el = np.fromiter((xxhash.xxh64_intdigest(a) for a in df['el']), np.uint64, n)
    en = np.fromiter((xxhash.xxh64_intdigest(b) for b in df['en']), np.uint64, n)
    return pair, el, en


def first_occurrences(hashes: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(hashes), dtype=bool)
    keep[np.unique(hashes, return_index=True)[1]] = True
    return keep


def over_cap(hashes: np.ndarray, alive: np.ndarray, cap: int) -> np.ndarray:
    """Rows of `alive` whose hash has already been seen `cap` times earlier in the array."""
    idx = np.flatnonzero(alive)
    over = np.zeros(len(hashes), dtype=bool)
    if not len(idx):
        return over
    order = np.argsort(hashes[idx], kind='stable')
    sorted_hashes = hashes[idx][order]
    starts = np.r_[True, sorted_hashes[1:] != sorted_hashes[:-1]]
    group_start = np.maximum.accumulate(np.where(starts, np.arange(len(idx)), 0))
    rank = np.arange(len(idx)) - group_start
    over[idx[order[rank >= cap]]] = True
    return over


def run(conf, corpora: List[str], overwrite: bool = False):
    shard_lists = {c: paths.shard_files(paths.stage_dir(conf, 'decontaminate', c)) for c in corpora}
    files = [f for c in corpora for f in shard_lists[c]]

    with ProcessPoolExecutor(max_workers=conf.num_workers) as executor:
        hashed = list(executor.map(hash_shard, files, chunksize=1))

    pair = np.concatenate([h[0] for h in hashed]) if hashed else np.zeros(0, np.uint64)
    el = np.concatenate([h[1] for h in hashed]) if hashed else np.zeros(0, np.uint64)
    en = np.concatenate([h[2] for h in hashed]) if hashed else np.zeros(0, np.uint64)

    reasons = np.full(len(pair), '', dtype=object)
    alive = first_occurrences(pair)
    reasons[~alive] = 'duplicate_pair'

    cap = conf.filter_args.dedup.max_translations
    for side, hashes in (('el', el), ('en', en)):
        over = over_cap(hashes, alive, cap)
        reasons[over] = f'{side}_over_cap'
        alive &= ~over

    print(f'[dedup] {len(pair):,} pairs -> {int(alive.sum()):,} kept')

    shard_reasons, start = {}, 0
    for f, h in zip(files, hashed):
        shard_reasons[f] = reasons[start:start + len(h[0])]
        start += len(h[0])

    # Global step: which copy survives depends on every corpus at once, so it is always recomputed.
    return run_stage(conf, 'dedup', 'decontaminate', corpora,
                     lambda corpus, i, src: MaskFn(shard_reasons[src]), overwrite=True)
