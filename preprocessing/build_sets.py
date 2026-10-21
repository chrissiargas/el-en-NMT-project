"""Plan section 4: build the Bridge set, the SFT set and the CPO source pool from the budget stage.

Every set stores both languages, so one build serves el->en and en->el; preprocessing/prepare.py picks
the direction. Selection runs on the small numeric columns first and reads the text of chosen rows only.

- bridge: `tier_a_share` random Tier A pairs plus the best-scoring A-/B pairs, both spread over corpora
  by temperature. `is_clean` marks Tier A and pairs with Bicleaner >= `clean_bicleaner`, for the
  noisy-to-clean ending of stage 1.
- sft: the human-translated `sft.corpora` plus the top `web_top_fraction` of each A-/B corpus, within the
  length window, where Bicleaner and LaBSE both pass; best Bicleaner first.
- preference_pool: the next-best SFT candidates, disjoint from sft, as CPO sources.
"""
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from preprocessing import paths
from preprocessing.sampling import allocate
from preprocessing.shards import read_shard, write_json, write_shards

META = ['corpus', 'tier', 'bicleaner', 'labse', 'el_words', 'en_words']


def load_meta(conf, corpus: str) -> pd.DataFrame:
    frames = []
    for f in paths.shard_files(paths.stage_dir(conf, 'budget', corpus)):
        m = read_shard(f, META)
        frames.append(m.assign(shard=f, row=np.arange(len(m))))
    assert frames, f'No budget-stage shards for {corpus}'
    return pd.concat(frames, ignore_index=True)


def gather(selected: pd.DataFrame) -> pd.DataFrame:
    """Attach el/en text to selected metadata rows."""
    if not len(selected):
        return pd.DataFrame(columns=['el', 'en'] + [c for c in selected.columns if c not in ('shard', 'row')])
    parts = []
    for shard, rows in selected.groupby('shard', sort=False):
        text = read_shard(shard, ['el', 'en']).iloc[rows['row'].values].reset_index(drop=True)
        parts.append(pd.concat([text, rows.drop(columns=['shard', 'row']).reset_index(drop=True)], axis=1))
    return pd.concat(parts, ignore_index=True)


def take(pools: Dict[str, pd.DataFrame], total: int, temperature: float, by_score: bool, seed: int) -> pd.DataFrame:
    quotas = allocate({c: len(p) for c, p in pools.items()}, total, temperature)
    picked = []
    for corpus, quota in quotas.items():
        pool = pools[corpus]
        if by_score:
            picked.append(pool.sort_values('bicleaner', ascending=False, kind='stable').head(quota))
        else:
            picked.append(pool.sample(n=quota, random_state=seed))
    return pd.concat(picked) if picked else pd.DataFrame(columns=META + ['shard', 'row'])


def save(conf, name: str, selected: pd.DataFrame):
    data = gather(selected).sample(frac=1.0, random_state=conf.seed).reset_index(drop=True)
    folder = paths.set_dir(conf, name)
    write_shards(data, folder, conf.shard_size)

    composition = data.groupby(['tier', 'corpus']).agg(pairs=('el', 'size'), bicleaner=('bicleaner', 'mean'),
                                                       labse=('labse', 'mean'), el_words=('el_words', 'mean'))
    write_json({'pairs': len(data), 'corpora': composition.reset_index().round(3).to_dict(orient='records')},
               os.path.join(folder, '_composition.json'))
    print(f'[sets] {name}: {len(data):,} pairs\n{composition.round(3).to_string()}')


def build_bridge(conf, meta: Dict[str, pd.DataFrame]):
    b, T = conf.set_args.bridge, conf.set_args.temperature
    target_a = int(b.size * b.tier_a_share)

    tier_a = take({c: m for c, m in meta.items() if conf.registry[c].tier == 'A'}, target_a, T, False, conf.seed)
    web = take({c: m for c, m in meta.items() if conf.registry[c].tier != 'A'}, b.size - target_a, T, True, conf.seed)

    selected = pd.concat([tier_a, web])
    selected['is_clean'] = (selected['tier'] == 'A') | (selected['bicleaner'] >= b.clean_bicleaner)
    save(conf, 'bridge', selected)


def build_sft(conf, meta: Dict[str, pd.DataFrame]):
    s, T = conf.set_args.sft, conf.set_args.temperature

    pools = {}
    for corpus, m in meta.items():
        tier = conf.registry[corpus].tier
        if corpus not in s.corpora and tier not in ('A-', 'B'):
            continue
        if m['bicleaner'].isna().all():
            print(f'[sets] WARNING: {corpus} has no scores and cannot enter the SFT set; check scoring')
            continue

        ok = m['el_words'].between(s.min_words, s.max_words) & m['en_words'].between(s.min_words, s.max_words) & \
             (m['bicleaner'] >= s.min_bicleaner) & (m['labse'] >= s.min_labse)
        if corpus not in s.corpora:
            ok &= m['bicleaner'] >= m['bicleaner'].quantile(1 - s.web_top_fraction)
        pools[corpus] = m[ok]

    sft = take(pools, s.size, T, True, conf.seed)
    save(conf, 'sft', sft)

    rest = {c: p.drop(index=sft.index[sft['corpus'] == c]) for c, p in pools.items()}
    save(conf, 'preference_pool', take(rest, conf.set_args.preference_size, T, True, conf.seed))


def run(conf, corpora: List[str], overwrite: bool = False):
    meta = {corpus: load_meta(conf, corpus) for corpus in corpora}

    build_bridge(conf, meta)
    build_sft(conf, meta)
