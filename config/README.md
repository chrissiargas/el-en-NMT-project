# config

Every setting of the project lives here, as in hung-asr-aws.

| File | Content |
| --- | --- |
| `my_config.yaml` | Paths, filtering thresholds, set sizes and training-data options |
| `my_config_exp{N}.yaml` | Optional experiment variants, selected with `--exp N` |
| `corpora.yaml` | Registry of the OPUS corpora: version, tier and per-corpus overrides |
| `parser.py` | Loads both files into one object and checks every key |

## Loading

```python
from config.parser import Parser

conf = Parser().get_args()                      # my_config.yaml
conf = Parser().get_args(exp=2)                 # my_config_exp2.yaml
conf = Parser().get_args(config_path='x.yaml')  # any file

conf.dataset_path                    # top-level keys become attributes
conf.filter_args.rules.max_words     # nested sections become Args objects
conf.registry['Europarl'].tier       # corpus registry
conf.get_corpora()                   # selected corpora in dedup priority order
```

Every script takes `--exp N` or `--config path` and passes it on. Call `nmt_utils.environment.set_environment(conf)` before importing any Hugging Face library; it sets `HF_HOME`, `HF_DATASETS_CACHE` and `TMPDIR` from the config.

## Strict keys

The parser accepts exactly the keys it knows:

- An unknown top-level section fails with `WRONG ARG`.
- A nested section fails on a missing **or** an unknown key, for example `filter_args.rules: missing keys [...], unknown keys [...]`. A typo stops the run instead of being silently ignored.
- Every key in `train_args` is checked the same way, and `direction` must be `el-en` or `en-el`.

To add an option, add it to `parser.py` (the key list of its section) **and** to every `my_config*.yaml`.

## Paths

Relative paths are joined under your home directory, and absolute paths are kept as they are. This applies to `tmpdir`, `hf_cache`, `dataset_path`, `results_path` and `checkpoint_path`. The defaults point to the remote GPU box (`/home/jovyan/asr-shared/csiargka/cache/...`).

All data lives under `{dataset_path}/{pair}/`; `preprocessing/paths.py` describes the layout.

## `my_config.yaml` sections

### `data_args`: paths and machine

| Key | Meaning |
| --- | --- |
| `tmpdir`, `hf_cache`, `dataset_path` | Temporary files, Hugging Face cache, data root |
| `pair` | OPUS pair code (`el-en`); also the name of the data folder |
| `num_workers` | CPU processes for the per-shard pipeline steps |
| `gpus` | Devices for the scoring step, one worker each |
| `shard_size` | Rows per parquet shard |
| `seed` | Seed for every random choice: held-out units, samples, set selection, shuffling |

### `main_args`: what to process

| Key | Meaning |
| --- | --- |
| `tiers` | Tiers of `corpora.yaml` to process. `B'` (OpenSubtitles) is left out by default. |
| `eval_sets` | Eval files (`eval/{name}.jsonl`) removed from training by `decontaminate`. The held-out sets are added automatically. |
| `results_path`, `checkpoint_path` | Not used yet; reserved for training and evaluation |

### `filter_args`: the filtering pipeline (plan section 3)

`samples` is the number of kept and removed pairs saved per step, corpus and reason, for reading by eye. The other keys are grouped by step:

| Subsection | Step | Keys |
| --- | --- | --- |
| `holdout` | 1. In-domain test sets | `corpora` to hold out from, `size` pairs each, `block_size` consecutive lines per held-out unit (a unit never crosses an OPUS document) |
| `normalize` | 2. Normalization | Greek-specific fixes that can be switched off: `fix_spacing_tonos`, `fix_homoglyphs`, `fix_final_sigma`. NFC, HTML, control-character and whitespace cleanup always run. |
| `decontaminate` | 1b. Remove eval overlap | `ngram` size for overlap matching; `min_words` below which eval sentences are not exact-matched |
| `dedup` | 3. Deduplication | `max_translations` per identical Greek sentence and per identical English sentence |
| `rules` | 4. Rule filters | Word-count range, character-length ratio (el/en), minimum Greek/Latin letter share, `max_non_alpha` share, `max_repeat` for repeated tokens, and whether numbers and URLs must match |
| `langid` | 5. Language ID | `model` (`glotlid`, `openlid` or `lid176`), `min_confidence`, `min_words` below which a side is not checked, `num_workers` (each holds a copy of the model) |
| `scoring` | 6. Adequacy scores | `tiers` to score; Bicleaner model, command and arguments; LaBSE model and batch size |
| `budget` | 7. Budget cut | `min_bicleaner` for non-A tiers, `total` pairs kept across them, `temperature` for spreading that total over corpora |

The thresholds are the plan's starting values; tune them by reading the samples (`python preprocessing/report.py inspect`).

Besides the tiers in `scoring.tiers`, the scoring step also scores any corpus with `min_bicleaner` and every corpus in `set_args.sft.corpora`, because the SFT selection ranks by those scores.

`bicleaner_cmd` can point to another environment, for example `conda run -n bicleaner bicleaner-ai-classify`, since Bicleaner-AI needs TensorFlow.

### `set_args`: final data sets (plan section 4)

| Key | Meaning |
| --- | --- |
| `temperature` | How sets spread over corpora: 1 is proportional to corpus size; higher values move towards equal shares |
| `bridge.size`, `bridge.tier_a_share` | Bridge set size and the share drawn at random from Tier A; the rest is the best-scoring A-/B pairs |
| `bridge.clean_bicleaner` | Rows at or above this score (and all Tier A rows) are marked `is_clean`, for the clean-only end of stage 1 |
| `sft.size`, `sft.corpora` | SFT set size and the human-translated corpora it draws from |
| `sft.web_top_fraction` | Share of each A-/B corpus, by Bicleaner score, that may also enter the SFT set |
| `sft.min_bicleaner`, `sft.min_labse` | Both scores must pass |
| `sft.min_words`, `sft.max_words` | Length window, on both sides |
| `preference_size` | Next-best SFT candidates kept apart as CPO sources, never overlapping the SFT set |

### `train_args`: training data

| Key | Meaning |
| --- | --- |
| `direction` | `el-en` (Greek to English) or `en-el` |
| `train_set` | Set to train on: `bridge` or `sft` |
| `eval_set` | Eval file used for validation (`flores_dev`) |
| `micro_data`, `micro_size` | Train on a small subset, for debugging |
| `randomize` | Shuffle the training set |

## `corpora.yaml`: corpus registry

One entry per OPUS corpus, keyed by its exact OPUS name:

```yaml
EUbookshop: {version: v2, tier: A, domain: eu_publications, min_bicleaner: 0.5}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `version` | yes | OPUS release to download |
| `tier` | yes | `A` (curated or human-translated), `A-` (curated crawl, mined Wikipedia), `B` (web-mined), `B'` (subtitles) |
| `domain` | yes | Label for reports and set composition |
| `enabled` | no, default `True` | `False` keeps the entry but skips the corpus |
| `min_bicleaner` | no | Keep only pairs scoring at least this; also applies to Tier A |
| `cap` | no | Maximum pairs kept after the budget cut |

**Order matters.** When the same pair appears in several corpora, dedup keeps the copy from the corpus listed first: by tier (A, A-, B, B'), then by position in the file. Human-translated corpora are therefore listed first.

The comment block at the top of the file lists every OPUS el-en corpus left out on purpose (mirrors, broken metadata, non-sentence data, UI strings), with the reason. The pair count after each entry comes from the OPUS API on 2026-10-08.

Four small Tier A corpora are not in the plan (EUconst, SPC, TildeMODEL, ECDC; about 26K pairs). `Wikipedia v1.0` is listed but disabled.
