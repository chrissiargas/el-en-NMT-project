"""Removal-rate tables (plan section 3, "Check every step by eye") and a sample viewer.

    python preprocessing/report.py table
    python preprocessing/report.py inspect --stage rules --corpus HPLT --kind removed --reason numbers -n 20
"""
import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

import argparse
from typing import List, Optional

import pandas as pd

from config.parser import Parser, TIERS
from preprocessing import paths
from preprocessing.shards import KEPT_SAMPLES, REMOVED_SAMPLES, read_shard, read_stats


def table(conf, corpora: List[str], stages: Optional[List[str]] = None, out: Optional[str] = None):
    all_stages = stages or paths.STAGES
    counts, reasons = [], []
    for corpus in corpora:
        row = {'corpus': corpus, 'tier': conf.registry[corpus].tier}
        for stage in all_stages:
            stats = read_stats(paths.stage_dir(conf, stage, corpus))
            row[stage] = stats['out'] if stats else None
            if stats and stage == 'raw':
                row['raw_in'] = stats['in']
            for reason, n in (stats['removed'].items() if stats else []):
                reasons.append({'stage': stage, 'corpus': corpus, 'reason': reason, 'removed': n,
                                'pct_of_step_input': round(100 * n / max(stats['in'], 1), 2)})
        counts.append(row)

    counts = pd.DataFrame(counts).drop(columns=['raw_in'], errors='ignore')
    stages = [s for s in all_stages if counts[s].notna().any()]
    totals = counts.groupby('tier', sort=False)[stages].sum(min_count=1).reset_index()
    totals['corpus'] = 'TOTAL ' + totals['tier']
    totals = totals.sort_values('tier', key=lambda t: t.map(TIERS.index))
    counts = pd.concat([counts, totals], ignore_index=True)[['corpus', 'tier'] + stages]

    removal = counts[['corpus', 'tier']].copy()
    for prev, stage in zip(stages, stages[1:]):
        removal[f'{stage} %'] = (100 * (1 - counts[stage] / counts[prev])).round(1)
    removal['kept %'] = (100 * counts[stages[-1]] / counts[stages[0]]).round(1)

    out = out or paths.report_dir(conf)
    os.makedirs(out, exist_ok=True)
    counts.to_csv(os.path.join(out, 'counts.csv'), index=False)
    removal.to_csv(os.path.join(out, 'removal_rates.csv'), index=False)
    pd.DataFrame(reasons).to_csv(os.path.join(out, 'removal_reasons.csv'), index=False)
    with open(os.path.join(out, 'removal_rates.md'), 'w', encoding='utf-8') as f:
        f.write('## Pairs after each step\n\n' + counts.to_markdown(index=False, floatfmt=',.0f') + '\n\n')
        f.write('## Removed at each step (% of that step\'s input)\n\n' + removal.to_markdown(index=False) + '\n')

    print(removal.to_string(index=False))
    print(f'\nWritten to {out}')
    return counts, removal


def inspect(conf, stage: str, corpus: str, kind: str = 'removed', reason: str = None, n: int = 20):
    folder = paths.stage_dir(conf, stage, corpus)
    df = read_shard(os.path.join(folder, REMOVED_SAMPLES if kind == 'removed' else KEPT_SAMPLES))
    if reason:
        df = df[df['reason'] == reason]
    for row in df.head(n).itertuples():
        tag = f' [{row.reason}]' if kind == 'removed' else ''
        print(f'--- {corpus} line {row.line_id}{tag}\nEL: {row.el}')
        if hasattr(row, 'en'):  # the TTS stages keep Greek only
            print(f'EN: {row.en}')
    if kind == 'removed' and 'reason' in df:
        print(f'\nReasons in sample: {df["reason"].value_counts().to_dict()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['table', 'inspect'])
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--corpora', nargs='+', default=None)
    parser.add_argument('--stage', choices=paths.STAGES + paths.TTS_STAGES, default='rules')
    parser.add_argument('--corpus', type=str, default=None)
    parser.add_argument('--kind', choices=['kept', 'removed'], default='removed')
    parser.add_argument('--reason', type=str, default=None)
    parser.add_argument('-n', type=int, default=20)
    args = parser.parse_args()

    conf = Parser().get_args(args.exp, args.config)
    if args.command == 'table':
        table(conf, conf.get_corpora(args.corpora))
    else:
        inspect(conf, args.stage, args.corpus, args.kind, args.reason, args.n)
