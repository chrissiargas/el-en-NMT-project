"""Parquet shard I/O and the per-shard stage runner shared by every pipeline step.

A step is a picklable callable `fn(df) -> (df, reasons)`, where `reasons` holds '' for each kept row and a
short label for each removed one. `run_stage` applies it to every shard of every corpus in parallel, writes
the kept rows to the next stage, and records counts per reason plus samples of kept and removed pairs.
"""
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from preprocessing import paths

SIDECAR = '_shards'
STATS = '_stats.json'
KEPT_SAMPLES = '_samples_kept.parquet'
REMOVED_SAMPLES = '_samples_removed.parquet'


def read_shard(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def read_stage(conf, stage: str, corpus: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    files = paths.shard_files(paths.stage_dir(conf, stage, corpus))
    assert files, f'No shards for {corpus} in stage {stage}'
    return pd.concat([read_shard(f, columns) for f in files], ignore_index=True)


def write_shard(df: pd.DataFrame, path: str):
    table = pa.Table.from_pandas(df, preserve_index=False)
    # All-None columns come out as `null`; make them strings so every shard of a stage shares one schema.
    table = table.cast(pa.schema([
        pa.field(f.name, pa.string() if pa.types.is_null(f.type) else f.type) for f in table.schema
    ]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    pq.write_table(table, tmp, compression='zstd')
    os.replace(tmp, path)


def write_shards(df: pd.DataFrame, folder: str, shard_size: int) -> List[str]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    files = []
    for i, start in enumerate(range(0, max(len(df), 1), shard_size)):
        files.append(os.path.join(folder, paths.shard_name(i)))
        write_shard(df.iloc[start:start + shard_size], files[-1])
    return files


def write_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_stats(folder: str) -> Optional[Dict]:
    path = os.path.join(folder, STATS)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_jsonl(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def read_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def keep_all(n: int) -> np.ndarray:
    return np.full(n, '', dtype=object)


class PassThrough:
    def __call__(self, df: pd.DataFrame):
        return df, keep_all(len(df))


class MaskFn:
    """Applies reasons computed beforehand for a whole corpus (global steps such as dedup and budget)."""

    def __init__(self, reasons: np.ndarray):
        self.reasons = reasons

    def __call__(self, df: pd.DataFrame):
        assert len(df) == len(self.reasons), f'{len(df)} rows but {len(self.reasons)} reasons'
        return df, self.reasons


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=seed) if len(df) else df


def sample_by_reason(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if not len(df):
        return df
    return pd.concat([sample_rows(g, n, seed) for _, g in df.groupby('reason', sort=True)], ignore_index=True)


def _sidecar(dst: str, suffix: str) -> str:
    folder, name = os.path.split(dst)
    return os.path.join(folder, SIDECAR, name.replace('.parquet', suffix))


def _process(task):
    fn, src, dst, n_samples, seed = task
    df = read_shard(src)
    n_in = len(df)
    df, reasons = fn(df)
    reasons = np.asarray(reasons, dtype=object)
    assert len(reasons) == len(df) == n_in, 'a step must return one reason per input row'
    keep = reasons == ''

    write_shard(df[keep], dst)
    write_shard(sample_rows(df[keep], n_samples, seed), _sidecar(dst, '.kept.parquet'))
    write_shard(sample_by_reason(df[~keep].assign(reason=reasons[~keep]), n_samples, seed),
                _sidecar(dst, '.removed.parquet'))
    write_json({'in': n_in, 'out': int(keep.sum()), 'removed': dict(Counter(reasons[~keep].tolist()))},
               _sidecar(dst, '.json'))
    return dst


def _is_done(src: str, dst: str) -> bool:
    stats = _sidecar(dst, '.json')
    return os.path.exists(dst) and os.path.exists(stats) and os.path.getmtime(dst) >= os.path.getmtime(src)


def finalize(conf, stage: str, corpus: str):
    """Merge the per-shard stats and samples of one corpus."""
    folder = paths.stage_dir(conf, stage, corpus)
    n_samples, seed = conf.filter_args.samples, conf.seed

    stats = {'in': 0, 'out': 0, 'removed': Counter()}
    kept, removed = [], []
    for dst in paths.shard_files(folder):
        with open(_sidecar(dst, '.json'), 'r', encoding='utf-8') as f:
            s = json.load(f)
        stats['in'] += s['in']
        stats['out'] += s['out']
        stats['removed'].update(s['removed'])
        kept.append(read_shard(_sidecar(dst, '.kept.parquet')))
        removed.append(read_shard(_sidecar(dst, '.removed.parquet')))

    stats['removed'] = dict(stats['removed'].most_common())
    write_json(stats, os.path.join(folder, STATS))
    write_shard(sample_rows(pd.concat(kept, ignore_index=True), n_samples, seed), os.path.join(folder, KEPT_SAMPLES))
    write_shard(sample_by_reason(pd.concat(removed, ignore_index=True), n_samples, seed),
                os.path.join(folder, REMOVED_SAMPLES))

    removed_pct = 100 * (1 - stats['out'] / max(stats['in'], 1))
    print(f'[{stage}] {corpus}: {stats["in"]:,} -> {stats["out"]:,} ({removed_pct:.1f}% removed)')
    return stats


def run_stage(conf,
              stage: str,
              src_stage: str,
              corpora: List[str],
              make_fn: Callable[[str, int, str], Callable],
              overwrite: bool = False,
              num_workers: Optional[int] = None,
              mp_context=None,
              initializer=None,
              initargs=()):
    """Run `make_fn(corpus, shard_index, src_path)` on every shard of `src_stage` and write `stage`."""
    tasks = []
    for corpus in corpora:
        src_dir = paths.stage_dir(conf, src_stage, corpus)
        dst_dir = paths.stage_dir(conf, stage, corpus)
        src_files = paths.shard_files(src_dir)
        if not src_files:
            print(f'[{stage}] {corpus}: no shards in {src_dir}, skipping')
            continue

        if overwrite and os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)

        # Drop outputs whose input shard no longer exists (the upstream step was re-sharded).
        src_names = {os.path.basename(f) for f in src_files}
        for dst in paths.shard_files(dst_dir):
            if os.path.basename(dst) not in src_names:
                os.remove(dst)

        for i, src in enumerate(src_files):
            dst = os.path.join(dst_dir, os.path.basename(src))
            if _is_done(src, dst):
                continue
            tasks.append((make_fn(corpus, i, src), src, dst, conf.filter_args.samples, conf.seed + i))

    if tasks:
        workers = min(num_workers or conf.num_workers, len(tasks))
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context,
                                 initializer=initializer, initargs=initargs) as executor:
            futures = [executor.submit(_process, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=stage, ncols=100, colour='green'):
                future.result()

    return {corpus: finalize(conf, stage, corpus) for corpus in corpora
            if paths.shard_files(paths.stage_dir(conf, stage, corpus))}
