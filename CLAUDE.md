# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Greek↔English MT research code. The plan (`claude_notes/Greek–English NMT Project Guide.md`) composes two frozen LLMs (Llama-Krikri-8B and an English-strong LLM) through a learned cross-attention bridge, following the Dual-Fusion ASR project (`~/PycharmProjects/hung-asr-aws`), whose layout and conventions this repo copies. So far only the **data side** exists: config, OPUS/eval-set constructors, the filtering pipeline (plan section 3) and the set builders (section 4). `nmt_utils/` (model, training), `baselines/` and `evaluations/` are still placeholders.

Code is developed locally in PyCharm and run on the remote GPU box (`/home/jovyan`, 4 GPUs), where data and caches live. There is no test suite or linter.

## Running things

Run everything from the repo root. Each entrypoint does `sys.path.insert` on the repo root and imports packages as `config.*`, `preprocessing.*`, `nmt_utils.*`. All take `--exp N` (`config/my_config_exp{N}.yaml`) or `--config path`.

```bash
python constructors/opus.py [--corpora Europarl SETIMES] [--download_only]   # OPUS -> stages/raw
python constructors/flores.py      # FLORES+ dev/devtest (gated: accept terms, huggingface-cli login)
python constructors/ntrex.py       # NTREX-128

python preprocessing/pipeline.py --steps all
python preprocessing/pipeline.py --steps rules langid --corpora Europarl --overwrite
python preprocessing/report.py table
python preprocessing/report.py inspect --stage rules --corpus HPLT --reason numbers -n 20
```

Steps, in order: `holdout` (1), `normalize` (2), `decontaminate` (1b), `dedup` (3), `rules` (4), `langid` (5), `score` (6, GPUs), `budget` (7), `sets` (section 4), `report`.

## Architecture

### Config
- `config/parser.py::Parser.get_args(exp)` loads `my_config.yaml` (sections `data_args`, `main_args`, `filter_args`, `set_args`, `train_args`) and `config/corpora.yaml` (the OPUS corpus registry). Nested sections go through `section()`, which fails on **missing and unknown** keys. To add an option, add it to `parser.py` and to every `my_config*.yaml`.
- `get_corpora()` returns corpus names in dedup priority order (tier A, A-, B, B', then file order). Order in `corpora.yaml` therefore decides which duplicate survives.
- Relative paths are joined under `~`; absolute paths are kept as they are.
- `nmt_utils/environment.py::set_environment(conf)` must run before any Hugging Face import (sets `HF_HOME`, `HF_DATASETS_CACHE`, `TMPDIR`). Preprocessing modules import HF libraries lazily inside functions for this reason.

### Data layout (`preprocessing/paths.py`)
Everything is under `{dataset_path}/{pair}/`: `downloads/`, `stages/{stage}/{corpus}/part-NNNNN.parquet`, `eval/{name}.jsonl`, `sets/{name}/`, `reports/`. Each stage folder also holds `_stats.json` (rows in/out, removals per reason) and `_samples_{kept,removed}.parquet` (for the plan's "read 100 kept and 100 removed").

### Pipeline mechanics (`preprocessing/shards.py`)
- A step is a picklable callable `fn(df) -> (df, reasons)`; `''` keeps a row, any other string removes it and becomes the reason label. `run_stage` maps it over shards with a process pool and resumes: a shard is skipped when its output is newer than its input.
- Global steps (`dedup`, `budget`) compute one reason array per shard in the main process, then apply it through `MaskFn`; they always recompute. `sets` reads the `budget` stage directly.
- Held-out pairs are not cut out of the corpora by `holdout`; `decontaminate` removes them (and any copies elsewhere) by matching normalized text against every eval set. It refuses to run if an eval file is missing.
- `score` runs one spawned worker per GPU. Bicleaner-AI is a subprocess (`bicleaner_cmd`), so it can live in a TensorFlow env of its own. The en-el model reads English in column 1 and Greek in column 2.
- Sets keep both languages. `preprocessing/prepare.py::make_data_module` maps them to `source`/`target` for `train_args.direction` and normalizes eval sources (never references).

### Greek text for TTS (`preprocessing/tts.py`)
A separate CPU pipeline run with `--config config/my_config_tts.yaml`. It reads `stages/raw` and writes `stages/tts_*` and `tts/{clean,needs_tn}`. Don't use `pipeline.py --corpora` for this: `dedup` would overwrite the NMT stage with a subset of the corpora.

## Gotchas
- Don't name a module after a stdlib one (`io.py`, `parser`-like names in new packages). Scripts run as `python preprocessing/x.py` put `preprocessing/` first on `sys.path`.
- `Parser` drops its argparse object when pickled (`__getstate__`), so `conf` can be sent to worker processes.
- fastText needs numpy < 2.
