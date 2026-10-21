"""Step 5: language ID on both sides with a fastText model (GlotLID by default).

Sides shorter than `min_words` are not checked: LID is unreliable on a few words and the script-ratio rule
already covers them. The probability of the expected language is kept as `lid_el` / `lid_en` (NaN when
the side was not checked, 0 when another language won).
"""
import os
import urllib.request
from typing import Dict

import numpy as np
import pandas as pd

# name -> (HF repo or URL, file name, Greek label, English label)
MODELS = {
    'glotlid': ('cis-lmu/glotlid', 'model.bin', '__label__ell_Grek', '__label__eng_Latn'),
    'openlid': ('laurievb/OpenLID', 'model.bin', '__label__ell_Grek', '__label__eng_Latn'),
    'lid176': ('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', 'lid.176.bin',
               '__label__el', '__label__en'),
}

_MODEL_CACHE: Dict[str, object] = {}


def model_path(name: str) -> str:
    source, filename, _, _ = MODELS[name]
    if source.startswith('http'):
        path = os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~')), 'fasttext', filename)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(source, path + '.tmp')
            os.replace(path + '.tmp', path)
        return path

    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=source, filename=filename)


def load_model(path: str):
    if path not in _MODEL_CACHE:
        import fasttext
        fasttext.FastText.eprint = lambda *args, **kwargs: None
        _MODEL_CACHE[path] = fasttext.load_model(path)
    return _MODEL_CACHE[path]


class LanguageID:
    def __init__(self, args, path: str):
        self.a = args
        self.path = path
        _, _, self.el_label, self.en_label = MODELS[args.model]

    def scores(self, texts: pd.Series, label: str) -> np.ndarray:
        out = np.full(len(texts), np.nan, dtype=np.float32)
        checked = np.flatnonzero([len(t.split()) >= self.a.min_words for t in texts])
        if len(checked):
            labels, probs = load_model(self.path).predict([texts.iloc[i] for i in checked], k=1)
            out[checked] = [p[0] if l[0] == label else 0.0 for l, p in zip(labels, probs)]
        return out

    def __call__(self, df: pd.DataFrame):
        df = df.copy()
        df['lid_el'] = self.scores(df['el'], self.el_label)
        df['lid_en'] = self.scores(df['en'], self.en_label)
        low_el = (df['lid_el'] < self.a.min_confidence).values  # NaN compares False: unchecked sides pass
        low_en = (df['lid_en'] < self.a.min_confidence).values
        reasons = np.where(low_el, 'lid_el', np.where(low_en, 'lid_en', '')).astype(object)
        return df, reasons


def run(conf, corpora, overwrite: bool = False):
    from preprocessing.shards import run_stage

    args = conf.filter_args.langid
    step = LanguageID(args, model_path(args.model))
    return run_stage(conf, 'langid', 'rules', corpora, lambda corpus, i, src: step,
                     overwrite=overwrite, num_workers=args.num_workers)
