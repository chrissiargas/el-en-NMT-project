# constructors

Step 0 of the data pipeline: download each source and write it in the layout `preprocessing/` reads. Nothing here filters or normalizes text. One script per source, as in hung-asr-aws.

Run from the repo root. Every script takes `--exp N` (`config/my_config_exp{N}.yaml`) or `--config path`. Paths below are relative to `{dataset_path}/{pair}/` (see `preprocessing/paths.py`).

```bash
python constructors/opus.py        # training corpora -> stages/raw/
python constructors/flores.py      # eval/flores_dev.jsonl, eval/flores_devtest.jsonl
python constructors/ntrex.py       # eval/ntrex.jsonl
python preprocessing/pipeline.py --steps all
```

Run all three before the pipeline: the `decontaminate` step refuses to run while any eval file is missing.

## `opus.py`: training corpora

For each corpus in `config/corpora.yaml` that is enabled and in `main_args.tiers` (or each one named with `--corpora`):

1. **Resolve** the Moses download URL for that corpus and version through the OPUS API.
2. **Download** the archive to `downloads/{corpus}/{version}/el-en.txt.zip`. Interrupted downloads resume from the `.part` file, and the API entry is saved next to the archive as `opus.json`.
3. **Convert** the aligned `.el` / `.en` files into parquet shards of `data_args.shard_size` rows in `stages/raw/{corpus}/part-NNNNN.parquet`.

| Column | Content |
| --- | --- |
| `el`, `en` | Untouched text |
| `doc_id` | Source document from the archive's `.ids` file, or null. Used by the `holdout` step. Some corpora (SETIMES, TED2020) ship as a single document. |
| `line_id` | Position in the archive, so every pair traces back to OPUS |
| `corpus`, `tier` | From the registry |

Checks:
- Fails if the `.el` and `.en` files have different line counts.
- Warns if the row count differs from the pair count OPUS lists.
- Splits lines on `\n` only, so a stray `\r` inside a sentence cannot shift the alignment.

It writes `_stats.json`, which `preprocessing/report.py` uses for raw counts. Corpora that are already converted are skipped.

| Flag | Effect |
| --- | --- |
| `--corpora A B` | Only these corpora (names as in `corpora.yaml`) |
| `--download_only` | Fetch archives, skip conversion |
| `--overwrite` | Convert again even if `stages/raw/{corpus}` exists |
| `--workers N` | Corpora processed in parallel (default 4) |

## `flores.py`: FLORES+ dev and devtest

Downloads `dev/` and `devtest/` for `ell_Grek` and `eng_Latn` from [`openlanguagedata/flores_plus`](https://huggingface.co/datasets/openlanguagedata/flores_plus), the maintained continuation of FLORES-200, and joins the two languages by sentence ID.

- Output: `eval/flores_dev.jsonl` (997 pairs, checkpoint selection) and `eval/flores_devtest.jsonl` (1,012 pairs, main test set).
- The dataset is gated: accept its terms on the Hub and run `huggingface-cli login` first.
- The Hub cache goes to `data_args.hf_cache` (the script calls `set_environment`).

## `ntrex.py`: NTREX-128

Downloads the English source (`newstest2019-src.eng.txt`) and the Greek reference (`newstest2019-ref.ell.txt`) from [MicrosoftTranslator/NTREX](https://github.com/MicrosoftTranslator/NTREX). It checks that both have 1,997 lines and writes `eval/ntrex.jsonl`.

## Eval file format

One JSON object per line:

```json
{"id": "ntrex-0", "el": "...", "en": "...", "origin": "en", "corpus": "ntrex"}
```

FLORES rows also carry `domain` and `topic`. The text is stored as published. `decontaminate` normalizes it before matching, and `preprocessing/prepare.py` normalizes only the source side when loading it for evaluation.

`origin` is the language the text was first written in. It is `en` for both FLORES and NTREX, so for el→en the Greek source is translated text.

The in-domain held-out test sets (`eval/heldout_{corpus}.jsonl`) have the same format. They are built by the `holdout` step in `preprocessing/`, not here, because they come out of corpora that `opus.py` has already downloaded.
