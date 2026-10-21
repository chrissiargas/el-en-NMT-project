"""FLORES+ dev (997) and devtest (1,012) for el/en -> eval/flores_{dev,devtest}.jsonl

FLORES+ is the maintained continuation of FLORES-200 (openlanguagedata/flores_plus). It is gated: accept
the terms on the Hub and log in (`huggingface-cli login`) first. Both sides are translations of English
Wikimedia text, so for el->en the Greek source is translated text.
"""
import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

import argparse
import json

from config.parser import Parser
from nmt_utils.environment import set_environment
from preprocessing import paths
from preprocessing.shards import write_jsonl

REPO = 'openlanguagedata/flores_plus'
CODES = {'el': 'ell_Grek', 'en': 'eng_Latn'}


class flores:
    def __init__(self, conf: Parser):
        self.conf = conf

    def load_side(self, split: str, lang: str):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=REPO, repo_type='dataset', filename=f'{split}/{CODES[lang]}.jsonl')
        with open(path, 'r', encoding='utf-8') as f:
            return {row['id']: row for row in map(json.loads, f)}

    def build(self, split: str):
        el, en = self.load_side(split, 'el'), self.load_side(split, 'en')
        assert el.keys() == en.keys(), f'FLORES+ {split}: el and en ids differ'

        rows = [{'id': f'flores-{split}-{i}', 'el': el[i]['text'], 'en': en[i]['text'], 'origin': 'en',
                 'corpus': 'flores', 'domain': en[i].get('domain'), 'topic': en[i].get('topic')}
                for i in sorted(el)]
        out = paths.eval_file(self.conf, f'flores_{split}')
        write_jsonl(rows, out)
        print(f'[flores] {split}: {len(rows)} pairs -> {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    conf = set_environment(Parser().get_args(args.exp, args.config))
    builder = flores(conf)
    builder.build('dev')
    builder.build('devtest')
