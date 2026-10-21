"""Step 6: adequacy scores as columns, `bicleaner` and `labse`; no rows are removed here.

Scored corpora: tiers in `scoring.tiers`, any corpus with `min_bicleaner` (EUbookshop), and the SFT corpora,
whose selection ranks by these scores. Other corpora pass through with NaN scores. One worker per GPU.

Bicleaner-AI needs TensorFlow, so it runs as a subprocess and may live in another env
(`bicleaner_cmd: 'conda run -n bicleaner bicleaner-ai-classify'`). The en-el model takes English in the
first column and Greek in the second.
"""
import multiprocessing as mp
import os
import shlex
import subprocess
import tempfile
from typing import Dict, List

import numpy as np
import pandas as pd

from preprocessing.shards import keep_all

_MODEL_CACHE: Dict[str, object] = {}


def init_gpu(queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(queue.get())


def scored_corpora(conf, corpora: List[str]) -> List[str]:
    sft = set(conf.set_args.sft.corpora)
    return [c for c in corpora if conf.registry[c].tier in conf.filter_args.scoring.tiers
            or conf.registry[c].min_bicleaner is not None or c in sft]


def labse(en: List[str], el: List[str], model_name: str, batch_size: int) -> np.ndarray:
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device='cuda')
    model = _MODEL_CACHE[model_name]
    kwargs = dict(batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return (model.encode(en, **kwargs) * model.encode(el, **kwargs)).sum(axis=1).astype(np.float32)


def bicleaner(en: List[str], el: List[str], args) -> np.ndarray:
    assert not any('\t' in t or '\n' in t for t in en + el), 'normalized text must not contain tabs or newlines'
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        src, out = os.path.join(tmp, 'input.tsv'), os.path.join(tmp, 'output.tsv')
        with open(src, 'w', encoding='utf-8') as f:
            f.writelines(f'{a}\t{b}\n' for a, b in zip(en, el))

        env = {**os.environ, 'TF_FORCE_GPU_ALLOW_GROWTH': 'true'}
        cmd = shlex.split(args.bicleaner_cmd) + list(args.bicleaner_args) + [src, out, args.bicleaner_model]
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)

        with open(out, 'r', encoding='utf-8') as f:
            scores = np.array([float(line.rstrip('\n').rsplit('\t', 1)[-1]) for line in f], dtype=np.float32)

    assert len(scores) == len(en), f'Bicleaner returned {len(scores)} scores for {len(en)} pairs'
    return scores


class Scorer:
    def __init__(self, args, active: bool):
        self.a = args
        self.active = active

    def __call__(self, df: pd.DataFrame):
        df = df.copy()
        if not self.active or not len(df):
            df['bicleaner'] = np.full(len(df), np.nan, dtype=np.float32)
            df['labse'] = np.full(len(df), np.nan, dtype=np.float32)
            return df, keep_all(len(df))

        en, el = df['en'].tolist(), df['el'].tolist()
        df['labse'] = labse(en, el, self.a.labse_model, self.a.labse_batch_size)

        import torch
        torch.cuda.empty_cache()  # leave the GPU memory to TensorFlow
        df['bicleaner'] = bicleaner(en, el, self.a)
        return df, keep_all(len(df))


def run(conf, corpora, overwrite: bool = False):
    from preprocessing.shards import run_stage

    args = conf.filter_args.scoring
    active = scored_corpora(conf, corpora)
    passive = [c for c in corpora if c not in active]
    print(f'[score] scoring {active}; passing through {passive}')

    stats = run_stage(conf, 'score', 'langid', passive, lambda corpus, i, src: Scorer(args, False),
                      overwrite=overwrite)

    context = mp.get_context('spawn')  # CUDA cannot be re-initialised in forked workers
    queue = context.Queue()
    for gpu in conf.gpus:
        queue.put(gpu)

    stats.update(run_stage(conf, 'score', 'langid', active, lambda corpus, i, src: Scorer(args, True),
                           overwrite=overwrite, num_workers=len(conf.gpus),
                           mp_context=context, initializer=init_gpu, initargs=(queue,)))
    return stats
