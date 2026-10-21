"""Load a built set (bridge / sft / preference_pool) and an eval set as a direction-specific DatasetDict.

Sets store both languages; `direction` ('el-en' or 'en-el') maps them to `source` / `target`. Eval sources
go through the training normalization (the model never saw un-normalized input); eval references stay
untouched. Call nmt_utils.environment.set_environment() first so the Hugging Face caches go to the config paths.
"""
import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

from typing import Optional

from config.parser import Parser
from preprocessing import paths
from preprocessing.normalize import Normalizer
from preprocessing.shards import read_jsonl


def directional(dataset, direction: str):
    src, tgt = direction.split('-')
    return dataset.rename_columns({src: 'source', tgt: 'target'})


def load_set(conf: Parser, name: str, direction: str, clean_only: bool = False):
    from datasets import load_dataset

    files = paths.shard_files(paths.set_dir(conf, name))
    assert files, f'No shards for set {name}; run `python preprocessing/pipeline.py --steps sets`'
    dataset = load_dataset('parquet', data_files=files, split='train')
    if clean_only:
        dataset = dataset.filter(lambda batch: batch['is_clean'], batched=True)
    return directional(dataset, direction)


def load_eval(conf: Parser, name: str, direction: str, normalize_source: bool = True):
    from datasets import Dataset

    rows = read_jsonl(paths.eval_file(conf, name))
    if normalize_source:
        src = direction.split('-')[0]
        normalizer = Normalizer(conf.filter_args.normalize)
        rows = [{**row, src: normalizer.text(row[src], src)} for row in rows]
    columns = ['id', 'el', 'en', 'origin', 'corpus']
    return directional(Dataset.from_list([{k: row.get(k) for k in columns} for row in rows]), direction)


def make_data_module(conf: Optional[Parser] = None,
                     exp: int = 0,
                     set_name: Optional[str] = None,
                     direction: Optional[str] = None,
                     eval_set: Optional[str] = None,
                     clean_only: bool = False):
    """train: the built set (all of it, or only `is_clean` rows for the end of stage 1); eval: FLORES dev."""
    from datasets import DatasetDict

    conf = conf if conf is not None else Parser().get_args(exp)
    t = conf.train_args
    direction = direction or t.direction

    train = load_set(conf, set_name or t.train_set, direction, clean_only)
    if t.randomize:
        train = train.shuffle(seed=conf.seed)
    if t.micro_data:
        train = train.select(range(min(t.micro_size, len(train))))

    return DatasetDict({
        'train': train,
        'eval': load_eval(conf, eval_set or t.eval_set, direction),
    })


if __name__ == '__main__':
    from nmt_utils.environment import set_environment

    data = make_data_module(set_environment())
    print(data)
    print(data['train'][0])
    print(data['eval'][0])
