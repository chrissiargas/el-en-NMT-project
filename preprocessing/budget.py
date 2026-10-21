"""Step 7: budget cut by Bicleaner score.

Tier A keeps everything, except corpora that set `min_bicleaner` (EUbookshop: 0.5). The other tiers keep
pairs at or above their threshold (`min_bicleaner` of the corpus, else `budget.min_bicleaner`), then the
best-scoring pairs up to a temperature-balanced share of `budget.total`, so no single crawl dominates.
Corpora with a `cap` (OpenSubtitles) never keep more than that.
"""
import math
from typing import List

import numpy as np

from preprocessing import paths
from preprocessing.sampling import allocate
from preprocessing.shards import MaskFn, read_shard, run_stage


def run(conf, corpora: List[str], overwrite: bool = False):
    b = conf.filter_args.budget
    files, scores, reasons, available = {}, {}, {}, {}
    shard_scores = {}

    for corpus in corpora:
        c = conf.registry[corpus]
        files[corpus] = paths.shard_files(paths.stage_dir(conf, 'score', corpus))
        for f in files[corpus]:
            shard_scores[f] = read_shard(f, ['bicleaner'])['bicleaner'].to_numpy()
        scores[corpus] = np.concatenate([shard_scores[f] for f in files[corpus]])

        threshold = c.min_bicleaner if c.min_bicleaner is not None else (None if c.tier == 'A' else b.min_bicleaner)
        if threshold is None:
            eligible = np.ones(len(scores[corpus]), dtype=bool)
        else:
            assert not np.isnan(scores[corpus]).all(), f'{corpus} has no Bicleaner scores; run the score step'
            eligible = scores[corpus] >= threshold

        reasons[corpus] = np.where(eligible, '', 'below_bicleaner').astype(object)
        if c.tier != 'A':
            available[corpus] = int(min(eligible.sum(), c.cap or math.inf))

    quotas = allocate(available, b.total, b.temperature)
    for corpus, quota in quotas.items():
        eligible = np.flatnonzero(reasons[corpus] == '')
        ranked = eligible[np.argsort(-scores[corpus][eligible], kind='stable')]
        reasons[corpus][ranked[quota:]] = 'over_budget'
        print(f'[budget] {corpus}: {available[corpus]:,} above threshold, keeping {quota:,}')

    shard_reasons = {}
    for corpus in corpora:
        start = 0
        for f in files[corpus]:
            shard_reasons[f] = reasons[corpus][start:start + len(shard_scores[f])]
            start += len(shard_scores[f])

    # Global step: the cut depends on every corpus at once, so it is always recomputed.
    return run_stage(conf, 'budget', 'score', corpora,
                     lambda corpus, i, src: MaskFn(shard_reasons[src]), overwrite=True)
