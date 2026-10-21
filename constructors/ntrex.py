"""NTREX-128 (1,997 news sentences) -> eval/ntrex.jsonl

The English side is the WMT19 news test source; the Greek side is a human translation of it, so for
el->en the Greek source is translated text.
"""
import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

import argparse

import requests

from config.parser import Parser
from preprocessing import paths
from preprocessing.shards import write_jsonl

BASE = 'https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128'
FILES = {'en': 'newstest2019-src.eng.txt', 'el': 'newstest2019-ref.ell.txt'}


class ntrex:
    def __init__(self, conf: Parser):
        self.conf = conf

    def load_side(self, lang: str):
        response = requests.get(f'{BASE}/{FILES[lang]}', timeout=60)
        response.raise_for_status()
        response.encoding = 'utf-8-sig'
        return response.text.rstrip('\n').split('\n')

    def build(self):
        el, en = self.load_side('el'), self.load_side('en')
        assert len(el) == len(en) == 1997, f'NTREX-128: {len(el)} el / {len(en)} en lines, expected 1997'

        rows = [{'id': f'ntrex-{i}', 'el': a.strip(), 'en': b.strip(), 'origin': 'en', 'corpus': 'ntrex'}
                for i, (a, b) in enumerate(zip(el, en))]
        out = paths.eval_file(self.conf, 'ntrex')
        write_jsonl(rows, out)
        print(f'[ntrex] {len(rows)} pairs -> {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    ntrex(Parser().get_args(args.exp, args.config)).build()
