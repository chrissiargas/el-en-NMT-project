# preprocessing

Turns the raw OPUS corpora (from `constructors/`) into clean, deduplicated, scored pairs (plan section 3), builds the training sets (section 4), and loads them for training.

```bash
python preprocessing/pipeline.py --steps all
python preprocessing/pipeline.py --steps rules langid --corpora Europarl SETIMES
python preprocessing/pipeline.py --steps normalize --overwrite
python preprocessing/report.py table
python preprocessing/report.py inspect --stage rules --corpus HPLT --reason numbers -n 20
```

Run from the repo root after the three constructors. Every script takes `--exp N` or `--config path`; all thresholds live in `config/my_config.yaml` (see `config/README.md`).

## Steps

`pipeline.py` runs the steps in this order. Each step reads the previous stage and writes its own folder under `stages/`.

| Step | Module | Plan | What it does | Removal reasons |
| --- | --- | --- | --- | --- |
| `holdout` | `holdout.py` | 3.1, 4.4 | Draws the in-domain test sets (`eval/heldout_{corpus}.jsonl`) from the raw corpora, in whole blocks of consecutive lines that never cross an OPUS document. Writes no stage. | none |
| `normalize` | `normalize.py` | 3.2 | Removes HTML, applies Unicode NFC, drops control and zero-width characters, collapses whitespace. Greek fixes: spacing tonos before capitals, Latin/Greek look-alike letters inside a word, final sigma. Keeps casing and punctuation. | `empty` |
| `decontaminate` | `decontaminate.py` | 3.1, 4.3 | Removes pairs that match an eval sentence (after normalizing, lowercasing and dropping punctuation) or share a 13-word sequence with one. This also removes the held-out pairs from their own corpus. | `exact_match`, `ngram_overlap` |
| `dedup` | `dedup.py` | 3.3 | Exact dedup across all corpora, keeping the copy from the highest-priority corpus. Then at most `max_translations` per identical Greek sentence and per identical English sentence. | `duplicate_pair`, `el_over_cap`, `en_over_cap` |
| `rules` | `rules.py` | 3.4 | Rule filters, applied in the order of the reasons on the right; a pair gets the first rule it fails. Adds `el_words` / `en_words`. | `replacement_char`, `length`, `identical`, `char_ratio`, `no_letters`, `non_alpha`, `script_el`, `script_en`, `repeated_token`, `numbers`, `urls` |
| `langid` | `lid.py` | 3.5 | Language ID on both sides with fastText (GlotLID by default). Sides shorter than `min_words` are not checked. Adds `lid_el` / `lid_en`. | `lid_el`, `lid_en` |
| `score` | `scoring.py` | 3.6 | Adds `bicleaner` and `labse` scores, one worker per GPU. Unscored corpora pass through with empty scores. | none |
| `budget` | `budget.py` | 3.7 | Tier A keeps everything unless the corpus sets `min_bicleaner`. Other tiers keep pairs above the threshold, then the best-scoring ones up to their share of `budget.total`. | `below_bicleaner`, `over_budget` |
| `sets` | `build_sets.py` | 4 | Builds `sets/bridge`, `sets/sft` and `sets/preference_pool`. | none |
| `report` | `report.py` | 3 | Writes the removal-rate tables to `reports/`. | none |

Notes on the order:

- **Decontamination runs after normalization.** The plan puts it first, but matching only works on normalized text.
- **Held-out pairs stay in the corpora until `decontaminate`.** `holdout` only writes the test files; `decontaminate` then removes those sentences, and any copies in other corpora, from training. It refuses to run if any eval file is missing.
- **Number matching ignores separators.** Greek writes 1.234,5 where English writes 1,234.5.
- **The dedup cap applies to both sides**, because each set serves both directions.

## Final sets

All sets store both languages, so one build serves el→en and en→el.

| Set | Selection | Extra column |
| --- | --- | --- |
| `bridge` | `tier_a_share` random Tier A pairs plus the best-scoring A-/B pairs. Both parts are spread over corpora by temperature. | `is_clean`: Tier A, or Bicleaner ≥ `clean_bicleaner`. Used for the clean-only end of stage 1. |
| `sft` | Pairs from the human-translated `sft.corpora` plus the top `web_top_fraction` of each A-/B corpus. Both Bicleaner and LaBSE must pass, and both sides must be within the length window. Best Bicleaner first. | none |
| `preference_pool` | The next-best SFT candidates, never overlapping `sft`. CPO sources; the candidate translations come later. | none |

Each set folder has `_composition.json`, with the pairs and mean scores per corpus.

## Loading for training

```python
from nmt_utils.environment import set_environment
from preprocessing.prepare import make_data_module

conf = set_environment()
data = make_data_module(conf)                                   # train_args from the config
data = make_data_module(conf, set_name='sft', direction='en-el', eval_set='flores_devtest')
data = make_data_module(conf, clean_only=True)                  # is_clean rows of the bridge set
```

Returns a `DatasetDict` with `train` and `eval`, where the `el` / `en` columns are renamed to `source` / `target` for the direction. Eval sources get the same normalization as training data; references are never changed.

## Output layout

Under `{dataset_path}/{pair}/` (defined in `paths.py`):

```
stages/{stage}/{corpus}/part-NNNNN.parquet    kept pairs
stages/{stage}/{corpus}/_stats.json           rows in and out, removals per reason
stages/{stage}/{corpus}/_samples_kept.parquet     up to `samples` kept pairs
stages/{stage}/{corpus}/_samples_removed.parquet  up to `samples` removed pairs per reason
stages/{stage}/{corpus}/_shards/              per-shard stats and samples (used to resume)
eval/{name}.jsonl                             FLORES, NTREX, held-out sets
eval/_decontamination_index.pkl
sets/{bridge,sft,preference_pool}/
reports/counts.csv, removal_rates.{csv,md}, removal_reasons.csv
```

Every row keeps `corpus`, `tier` and `line_id`, so any pair can be traced back to its OPUS archive.

## Checking the data

The plan asks you to read samples of kept and removed pairs at every step:

```bash
python preprocessing/report.py table                     # pairs and % removed per step and corpus
python preprocessing/report.py inspect --stage dedup --corpus HPLT --kind removed -n 20
python preprocessing/report.py inspect --stage langid --corpus Europarl --kind kept
```

`reports/removal_rates.md` is the removal-rate table for the report.

## Resuming and re-running

- **Per-shard steps** (`normalize`, `decontaminate`, `rules`, `langid`, `score`) skip a shard when its output is newer than its input. Use `--overwrite` after changing a threshold; an upstream re-run makes downstream shards out of date automatically.
- **Global steps** (`dedup`, `budget`, `sets`) depend on all corpora at once, so they always recompute. Run them on the full corpus selection: with `--corpora`, they only see the corpora you list.
- **Decontamination** does not notice a new or changed eval file. Re-run it with `--overwrite`.

## Writing a new step

A step is a picklable callable `fn(df) -> (df, reasons)`:

- `df` is one shard as a pandas DataFrame. The step may add columns.
- `reasons` has one string per row: `''` keeps the row, anything else removes it and becomes the reason label.

Pass it to `shards.run_stage(conf, stage, src_stage, corpora, make_fn)`, which runs it over every shard in parallel and writes the stats and samples. Add the stage name to `paths.STAGES` and the step to `pipeline.STEPS`.

## Requirements and pitfalls

- **Hugging Face imports.** Call `set_environment` before importing Hugging Face libraries. The modules here import them inside functions for that reason.
- **Language ID.** It needs `fasttext-wheel`, which needs numpy < 2. In a local test, `lid176` at confidence 0.8 rejected about 3% of clean English; GlotLID kept most of those.
- **Bicleaner-AI.** It needs TensorFlow, so install it in a separate environment and set `scoring.bicleaner_cmd`, for example `conda run -n bicleaner bicleaner-ai-classify`. The en-el model reads English in column 1 and Greek in column 2. The real Bicleaner hasn't been tested with this code yet; only a stub with the same command-line shape has.
- **Memory.** `dedup` holds three 64-bit hashes per pair, about 3.6 GB for 150M pairs.
- **Module names.** Don't name a module here after a standard-library module (`io.py`, for example). Running `python preprocessing/x.py` puts this folder first on the import path, so it would shadow the real one.
