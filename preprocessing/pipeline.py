"""Run the filtering pipeline (plan section 3) and build the final sets (section 4).

    python constructors/opus.py && python constructors/flores.py && python constructors/ntrex.py
    python preprocessing/pipeline.py --steps all
    python preprocessing/pipeline.py --steps rules langid --corpora Europarl SETIMES
    python preprocessing/pipeline.py --steps normalize --overwrite

Per-shard steps resume: finished shards are skipped unless --overwrite is given or their input changed.
dedup, budget and sets are global and always recomputed over the selected corpora, so run them on the
full corpus selection.
"""
import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

import argparse
import time

from config.parser import Parser
from nmt_utils.environment import set_environment
from preprocessing import budget, build_sets, decontaminate, dedup, holdout, lid, normalize, report, rules, scoring

STEPS = {
    'holdout': holdout.run,              # 1   in-domain held-out test sets
    'normalize': normalize.run,          # 2
    'decontaminate': decontaminate.run,  # 1b  remove FLORES / NTREX / held-out overlap
    'dedup': dedup.run,                  # 3   global
    'rules': rules.run,                  # 4
    'langid': lid.run,                   # 5
    'score': scoring.run,                # 6   GPUs
    'budget': budget.run,                # 7   global
    'sets': build_sets.run,              # section 4, global
    'report': lambda conf, corpora, overwrite: report.table(conf, corpora),
}
GLOBAL = {'dedup', 'budget', 'sets'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--steps', nargs='+', default=['all'], choices=['all'] + list(STEPS))
    parser.add_argument('--corpora', nargs='+', default=None, help='default: corpora of main_args.tiers')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    conf = set_environment(Parser().get_args(args.exp, args.config))
    corpora = conf.get_corpora(args.corpora)
    steps = list(STEPS) if 'all' in args.steps else [s for s in STEPS if s in args.steps]

    if args.corpora and GLOBAL & set(steps):
        print(f'WARNING: {sorted(GLOBAL & set(steps))} only see the selected corpora: {corpora}')

    print(f'Steps: {steps}\nCorpora ({len(corpora)}): {corpora}')
    for step in steps:
        start = time.time()
        print(f'\n===== {step} =====')
        STEPS[step](conf, corpora, overwrite=args.overwrite)
        print(f'===== {step} done in {(time.time() - start) / 60:.1f} min =====')
