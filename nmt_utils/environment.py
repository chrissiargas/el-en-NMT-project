import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

from typing import Optional
from config.parser import Parser


def set_environment(conf: Optional[Parser] = None):
    """Point the Hugging Face caches and TMPDIR at the config paths. Call before any HF import."""
    if conf is None:
        conf = Parser().get_args()

    os.environ['HF_HOME'] = conf.hf_cache
    os.environ['HF_DATASETS_CACHE'] = os.path.join(os.environ['HF_HOME'], 'datasets')
    os.environ['TMPDIR'] = conf.tmpdir
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.makedirs(conf.tmpdir, exist_ok=True)

    # W&B reads WANDB_API_KEY from the shell or from `wandb login`; never hard-code it here.

    return conf


if __name__ == '__main__':
    set_environment()

    from datasets import config

    print(f'Cache Dir: {config.HF_DATASETS_CACHE}')
    from huggingface_hub import constants

    print(f'Hub Dir: {constants.HF_HUB_CACHE}')
