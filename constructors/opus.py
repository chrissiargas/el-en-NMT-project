"""Download OPUS Moses archives and convert them to the `raw` stage (parquet shards).

    python constructors/opus.py                       # every corpus selected by main_args.tiers
    python constructors/opus.py --corpora Europarl SETIMES
    python constructors/opus.py --download_only

Each row keeps the untouched text plus `doc_id` (the source document from the archive's `.ids` file, when
there is one) and `line_id` (position in the archive), so every later pair traces back to OPUS.
"""
import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

import argparse
import io
import json
import zipfile
from itertools import zip_longest
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import pandas as pd
import requests
from tqdm import tqdm

from config.parser import Parser
from preprocessing import paths
from preprocessing.shards import write_json, write_shard, STATS

API = 'https://opus.nlpl.eu/opusapi/'


class opus:
    def __init__(self, conf: Parser):
        self.conf = conf
        self.source, self.target = conf.pair.split('-')

    def resolve(self, corpus: str) -> Dict:
        version = self.conf.registry[corpus].version
        response = requests.get(API, params={'corpus': corpus, 'source': self.source, 'target': self.target,
                                             'preprocessing': 'moses', 'version': version}, timeout=60)
        response.raise_for_status()
        entries = response.json()['corpora']
        assert entries, f'OPUS has no Moses release of {corpus} {version} for {self.conf.pair}'
        return entries[0]

    def download(self, corpus: str) -> str:
        folder = paths.download_dir(self.conf, corpus)
        target = os.path.join(folder, f'{self.conf.pair}.txt.zip')
        if os.path.exists(target):
            return target

        entry = self.resolve(corpus)
        os.makedirs(folder, exist_ok=True)
        write_json(entry, os.path.join(folder, 'opus.json'))
        part = target + '.part'
        done = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {'Range': f'bytes={done}-'} if done else {}

        with requests.get(entry['url'], headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status()
            if done and response.status_code != 206:  # server ignored the range: start over
                done = 0
            total = done + int(response.headers.get('Content-Length', 0))
            with open(part, 'ab' if done else 'wb') as f, \
                    tqdm(total=total, initial=done, unit='B', unit_scale=True, desc=corpus, ncols=100) as bar:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))

        os.replace(part, target)
        return target

    def members(self, archive: zipfile.ZipFile) -> Dict[str, str]:
        found = {}
        for suffix in (self.source, self.target, 'ids'):
            names = [n for n in archive.namelist() if n.endswith(f'.{self.conf.pair}.{suffix}')]
            assert len(names) <= 1, f'Ambiguous {suffix} files: {names}'
            if names:
                found[suffix] = names[0]
        assert self.source in found and self.target in found, f'No Moses text files in {archive.namelist()}'
        return found

    def convert(self, corpus: str) -> int:
        archive_path = self.download(corpus)
        folder = paths.stage_dir(self.conf, 'raw', corpus)
        tier = self.conf.registry[corpus].tier

        def lines(archive, name):
            # newline='\n': a stray '\r' inside a sentence must not split it and shift the alignment
            with io.TextIOWrapper(archive.open(name), encoding='utf-8', errors='replace', newline='\n') as f:
                for line in f:
                    yield line.rstrip('\n')

        def flush(buffer, index):
            write_shard(pd.DataFrame(buffer).assign(corpus=corpus, tier=tier),
                        os.path.join(folder, paths.shard_name(index)))

        if os.path.exists(folder):
            for f in paths.shard_files(folder):
                os.remove(f)

        with zipfile.ZipFile(archive_path) as archive:
            m = self.members(archive)
            ids = lines(archive, m['ids']) if 'ids' in m else None
            buffer = {'el': [], 'en': [], 'doc_id': [], 'line_id': []}
            n, shard = 0, 0
            for el, en in zip_longest(lines(archive, m['el']), lines(archive, m['en'])):
                assert el is not None and en is not None, f'{corpus}: el and en files differ in length'
                doc = next(ids, None) if ids else None
                buffer['el'].append(el)
                buffer['en'].append(en)
                buffer['doc_id'].append(doc.split('\t', 1)[0] if doc else None)
                buffer['line_id'].append(n)
                n += 1
                if len(buffer['el']) == self.conf.shard_size:
                    flush(buffer, shard)
                    buffer = {k: [] for k in buffer}
                    shard += 1
            if buffer['el'] or shard == 0:
                flush(buffer, shard)
            else:
                shard -= 1

        with open(os.path.join(paths.download_dir(self.conf, corpus), 'opus.json'), 'r') as f:
            expected = json.load(f)['alignment_pairs']
        if expected != n:
            print(f'[raw] WARNING {corpus}: {n:,} lines, OPUS lists {expected:,} pairs')
        write_json({'in': n, 'out': n, 'removed': {}}, os.path.join(folder, STATS))
        print(f'[raw] {corpus}: {n:,} pairs in {shard + 1} shards')
        return n


def _convert(args):
    conf, corpus, download_only = args
    builder = opus(conf)
    return builder.download(corpus) if download_only else builder.convert(corpus)


def run(conf, corpora: List[str], download_only: bool = False, overwrite: bool = False, workers: int = 4):
    todo = [c for c in corpora
            if overwrite or download_only or not os.path.exists(os.path.join(paths.stage_dir(conf, 'raw', c), STATS))]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_convert, [(conf, c, download_only) for c in todo]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--corpora', nargs='+', default=None)
    parser.add_argument('--download_only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--workers', type=int, default=4, help='corpora downloaded/converted in parallel')
    args = parser.parse_args()

    conf = Parser().get_args(args.exp, args.config)
    run(conf, conf.get_corpora(args.corpora), args.download_only, args.overwrite, args.workers)
