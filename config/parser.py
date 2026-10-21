import argparse
import os
import yaml
from os.path import dirname, abspath
from typing import Dict, List, Optional

TIERS = ['A', 'A-', 'B', "B'"]


class Args:
    def __init__(self):
        pass

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Key {key} not found in Arguments")

    def __repr__(self):
        return f'Args({vars(self)})'


def expand_path(path: str) -> str:
    # Relative paths live under ~, absolute paths are kept as they are.
    return os.path.join(os.path.expanduser('~'), path)


def section(values: Dict, keys: List[str], name: str) -> Args:
    missing = [k for k in keys if k not in values]
    unknown = [k for k in values if k not in keys]
    assert not missing and not unknown, f'{name}: missing keys {missing}, unknown keys {unknown}'

    args = Args()
    for key in keys:
        setattr(args, key, values[key])
    return args


class Parser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="data processing and training parameters"
        )

    def __call__(self, exp: int = 0, config_path: Optional[str] = None, *args, **kwargs):
        project_root = dirname(abspath(__file__))

        if config_path is None:
            if exp == 0:
                config_path = os.path.join(project_root, 'my_config.yaml')
            else:
                config_path = os.path.join(project_root, f'my_config_exp{exp}.yaml')

        self.parser.add_argument(
            '--config',
            default=config_path,
            help='config file location'
        )

        self.parser.add_argument(
            '--registry',
            default=os.path.join(project_root, 'corpora.yaml'),
            help='OPUS corpus registry'
        )

        self.parser.add_argument(
            '--data_args',
            default=dict(),
            type=dict,
            help='paths and machine arguments'
        )

        self.parser.add_argument(
            '--main_args',
            default=dict(),
            type=dict,
            help='corpora and evaluation sets'
        )

        self.parser.add_argument(
            '--filter_args',
            default=dict(),
            type=dict,
            help='filtering pipeline arguments (plan section 3)'
        )

        self.parser.add_argument(
            '--set_args',
            default=dict(),
            type=dict,
            help='final data set arguments (plan section 4)'
        )

        self.parser.add_argument(
            '--train_args',
            default=dict(),
            type=dict,
            help='training data arguments'
        )

    def __getstate__(self):
        # argparse parsers do not pickle; worker processes only need the loaded values.
        state = self.__dict__.copy()
        state.pop('parser', None)
        return state

    def get_args(self, exp: int = 0, config_path: Optional[str] = None):
        self.__call__(exp=exp, config_path=config_path)
        args = self.parser.parse_args(args=[])
        configFile = args.config

        assert configFile is not None

        with open(configFile, 'r') as cf:
            defaultArgs = yaml.load(cf, Loader=yaml.FullLoader)

        keys = vars(args).keys()

        for defaultKey in defaultArgs.keys():
            if defaultKey not in keys:
                print('WRONG ARG: {}'.format(defaultKey))
                assert (defaultKey in keys)

        self.parser.set_defaults(**defaultArgs)
        args = self.parser.parse_args(args=[])
        self.config_path = configFile

        ## Paths & Machine
        self.tmpdir = expand_path(args.data_args['tmpdir'])
        self.hf_cache = expand_path(args.data_args['hf_cache'])
        self.dataset_path = expand_path(args.data_args['dataset_path'])
        self.pair = args.data_args['pair']
        self.num_workers = args.data_args['num_workers']
        self.gpus = args.data_args['gpus']
        self.shard_size = args.data_args['shard_size']
        self.seed = args.data_args['seed']

        ## Corpora & Evaluation Sets
        self.tiers = args.main_args['tiers']
        self.eval_sets = args.main_args['eval_sets']
        self.results_path = expand_path(args.main_args['results_path'])
        self.checkpoint_path = expand_path(args.main_args['checkpoint_path'])

        ## Filtering Pipeline
        f = args.filter_args
        self.filter_args = Args()
        self.filter_args.samples = f['samples']
        self.filter_args.holdout = section(f['holdout'], ['corpora', 'size', 'block_size'], 'filter_args.holdout')
        self.filter_args.normalize = section(f['normalize'], ['fix_spacing_tonos', 'fix_homoglyphs', 'fix_final_sigma'], 'filter_args.normalize')
        self.filter_args.decontaminate = section(f['decontaminate'], ['ngram', 'min_words'], 'filter_args.decontaminate')
        self.filter_args.dedup = section(f['dedup'], ['max_translations'], 'filter_args.dedup')
        self.filter_args.rules = section(f['rules'], [
            'min_words', 'max_words', 'min_char_ratio', 'max_char_ratio', 'min_greek', 'min_latin',
            'max_non_alpha', 'max_repeat', 'match_numbers', 'match_urls'
        ], 'filter_args.rules')
        self.filter_args.langid = section(f['langid'], ['model', 'min_confidence', 'min_words', 'num_workers'], 'filter_args.langid')
        self.filter_args.scoring = section(f['scoring'], [
            'tiers', 'bicleaner_model', 'bicleaner_cmd', 'bicleaner_args', 'labse_model', 'labse_batch_size'
        ], 'filter_args.scoring')
        self.filter_args.budget = section(f['budget'], ['min_bicleaner', 'total', 'temperature'], 'filter_args.budget')

        ## Final Data Sets
        s = args.set_args
        self.set_args = Args()
        self.set_args.temperature = s['temperature']
        self.set_args.bridge = section(s['bridge'], ['size', 'tier_a_share', 'clean_bicleaner'], 'set_args.bridge')
        self.set_args.sft = section(s['sft'], [
            'size', 'corpora', 'web_top_fraction', 'min_bicleaner', 'min_labse', 'min_words', 'max_words'
        ], 'set_args.sft')
        self.set_args.preference_size = s['preference_size']

        ## Training Data
        self.train_args = section(args.train_args, [
            'direction', 'train_set', 'eval_set', 'micro_data', 'micro_size', 'randomize'
        ], 'train_args')
        assert self.train_args.direction in ('el-en', 'en-el'), self.train_args.direction

        ## Corpus Registry
        with open(args.registry, 'r') as rf:
            registry = yaml.load(rf, Loader=yaml.FullLoader)['corpora']

        self.registry = {}
        for name, values in registry.items():
            values = {'enabled': True, 'min_bicleaner': None, 'cap': None, **values}
            corpus = section(values, ['version', 'tier', 'domain', 'enabled', 'min_bicleaner', 'cap'], f'corpora.{name}')
            assert corpus.tier in TIERS, f'corpora.{name}: unknown tier {corpus.tier}'
            self.registry[name] = corpus

        return self

    def get_corpora(self, names: Optional[List[str]] = None, tiers: Optional[List[str]] = None) -> List[str]:
        """Corpus names in dedup priority order: by tier, then by order in corpora.yaml."""
        if names:
            unknown = [n for n in names if n not in self.registry]
            assert not unknown, f'Not in config/corpora.yaml: {unknown}'
        else:
            tiers = tiers if tiers else self.tiers
            names = [n for n, c in self.registry.items() if c.enabled and c.tier in tiers]

        order = list(self.registry.keys())
        return sorted(names, key=lambda n: (TIERS.index(self.registry[n].tier), order.index(n)))


if __name__ == "__main__":
    parser = Parser()
    parser.get_args()
    print(parser.get_corpora())
