"""On-disk layout. Everything lives under {dataset_path}/{pair}:

    downloads/{corpus}/{version}/{pair}.txt.zip    OPUS Moses archives
    stages/{stage}/{corpus}/part-NNNNN.parquet     one folder per pipeline step
    stages/{stage}/{corpus}/_stats.json            rows in/out and removal reasons
    stages/{stage}/{corpus}/_samples_*.parquet     kept/removed pairs to read by eye
    eval/{name}.jsonl                              FLORES, NTREX, held-out sets
    sets/{name}/part-NNNNN.parquet                 bridge, sft, preference_pool
    reports/                                       removal-rate tables
    tts/{corpus}/{clean,needs_tn}.csv              Greek sentences for TTS (preprocessing/tts.py)
"""
import glob
import os
from typing import List

STAGES = ['raw', 'normalize', 'decontaminate', 'dedup', 'rules', 'langid', 'score', 'budget']
TTS_STAGES = ['tts_clean', 'tts_dedup']


def root(conf) -> str:
    return os.path.join(conf.dataset_path, conf.pair)


def download_dir(conf, corpus: str) -> str:
    return os.path.join(root(conf), 'downloads', corpus, conf.registry[corpus].version)


def stage_dir(conf, stage: str, corpus: str) -> str:
    assert stage in STAGES + TTS_STAGES, f'Unknown stage {stage}'
    return os.path.join(root(conf), 'stages', stage, corpus)


def shard_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, 'part-*.parquet')))


def shard_name(index: int) -> str:
    return f'part-{index:05d}.parquet'


def eval_file(conf, name: str) -> str:
    return os.path.join(root(conf), 'eval', f'{name}.jsonl')


def heldout_names(conf) -> List[str]:
    return [f'heldout_{corpus}' for corpus in conf.filter_args.holdout.corpora]


def set_dir(conf, name: str) -> str:
    return os.path.join(root(conf), 'sets', name)


def report_dir(conf) -> str:
    return os.path.join(root(conf), 'reports')


def tts_dir(conf, name: str = '') -> str:
    return os.path.join(root(conf), 'tts', name)
