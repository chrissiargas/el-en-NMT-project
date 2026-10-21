# Greek–English NMT: Data Preprocessing Progress

Progress report for the data phase of the project (plan sections 2–4), October 2026.

## Summary

- **Data chosen.** From the ~361.5M el–en pairs listed in OPUS we keep 54 corpora in three quality tiers: 16.7M curated pairs (Tier A), 4.2M curated-crawl pairs (Tier A−) and 126M web-mined pairs (Tier B). The web-mined pairs are only kept after scoring, and only up to a budget. Mirrors, broken entries and non-sentence data are excluded before any processing.
- **Evaluation data fixed first.** FLORES+ (dev and devtest), NTREX-128, and four in-domain held-out sets of about 2,000 pairs each, taken from Europarl, EMEA, TED2020 and SETIMES before any processing.
- **Pipeline implemented.** All ten steps run end to end on a 5-corpus pilot of 377K pairs: download, hold-out, normalization, decontamination, deduplication, rule filters, language ID, adequacy scoring, budget cut and set building. Every step records how many pairs it removes and why, and keeps samples of kept and removed pairs for reading by eye.
- **Not done yet.** The full-scale run on the GPU server, a test of the real Bicleaner-AI scorer, and tuning the thresholds on the full data.

## 1. Datasets and why we chose them

### 1.1 Source: OPUS

[OPUS](https://opus.nlpl.eu/corpora-search/el&en) collects public parallel corpora in one format, with version numbers and line-level alignments. For el–en it lists 114 corpora. We checked every pair count in the plan against the live OPUS listing, and all of them match.

### 1.2 Quality tiers

The corpora differ greatly in quality, size and domain, so each one goes into a tier, and each tier is filtered differently.

| Tier | What it contains | Corpora | Raw pairs | Treatment |
| --- | --- | --- | --- | --- |
| **A** | Human-translated or curated | 49 | 16.7M | Light filtering; everything that passes is kept |
| **A−** | Curated web crawl, mined Wikipedia | 2 | 4.2M | Neural scoring, then kept by score |
| **B** | Web-mined at large scale | 3 | 126.3M | Neural scoring; only the best-scoring pairs, up to a budget |
| B′ | Film subtitles | 1 | 76.8M | Off by default; only if conversational text is in scope |

**Main corpora:**

| Corpus | Tier | Pairs | Domain | Why it is included |
| --- | --- | --- | --- | --- |
| DGT | A | 5.8M | EU legal | Large, professionally translated |
| EUbookshop | A | 4.0M | EU publications | Large, but noisy from PDF extraction, so it is also scored |
| Europarl | A | 1.3M | Parliament | Human-translated speech; a standard MT corpus |
| EMEA + ELRC-2711-EMEA | A | 1.1M + 0.8M | Medical | Medical domain |
| wikimedia | A | 0.9M | Wikipedia | General domain, human-translated |
| ELRC-5067-SciPar | A | 0.7M | Scientific abstracts | Scientific register |
| QED | A | 0.55M | Lecture subtitles | Spoken, educational |
| TED2020 + NeuLab-TedTalks | A | 0.27M + 0.15M | Talks | Spoken, general topics |
| SETIMES | A | 0.23M | News | News domain |
| GlobalVoices, ECB, JRC-Acquis, Tatoeba, Books, 28 small ELRC sets | A | ~0.4M | Mixed (government, news, everyday) | Small but clean; the ELRC sets add Greek and Cypriot government text |
| MaCoCu | A− | 3.6M | Greek web | Curated crawl of Greek websites |
| WikiMatrix | A− | 0.6M | Wikipedia, mined | General domain |
| HPLT, CCMatrix, ParaCrawl | B | 55.6M, 49.3M, 21.4M | Web | The only source of broad general-domain text at scale |

**Why not just the clean data?** Tier A is mostly EU legal and institutional text: DGT and EUbookshop alone are 59% of it. Our target is general-domain translation, so we also need web data (Tier B), even though it is much noisier. Only a scored, budgeted, domain-balanced slice of it is kept.

### 1.3 Excluded before processing

| Problem | Corpora | Why excluded |
| --- | --- | --- |
| Mirrors (same data under two names) | MultiHPLT (= HPLT), NLLB (≈ CCMatrix), MultiMaCoCu (= MaCoCu), every ELRA-W0xxx (= its ELRC twin), unnumbered ELRC sets, ELRC_2682, ELRC_2922 | Pure duplicates; downloading them only costs time. We found ELRC_2682 (a copy of ELRC-2711-EMEA) while checking OPUS; the plan had missed it. |
| Broken metadata | ELRC-EMEA (13.7M "pairs" but only 14.9M tokens), four similar ELRC sets | Sentence segmentation is broken; clean numbered versions exist |
| Not sentences | XLEnt (named entities), LinguaTools-WikiTitles (titles) | Not translation pairs |
| Software interface strings | KDE4, GNOME, translatewiki, tldr-pages | Menus and commands, not running text |
| Other | bible-uedin (archaic register), CCAligned (known to be very noisy), Ubuntu (empty) | Low value for our target domain |

We added four small clean corpora that the plan did not list (EUconst, SPC, TildeMODEL, ECDC; about 26K pairs together).

### 1.4 Evaluation data

| Set | Size | Use |
| --- | --- | --- |
| FLORES+ dev | 997 | Checkpoint selection, threshold tuning |
| FLORES+ devtest | 1,012 | Main test set (general domain) |
| NTREX-128 | 1,997 | Second general-domain test set (news) |
| Held-out Europarl, EMEA, TED2020, SETIMES | ~2,000 each | Domain robustness (parliament, medical, talks, news) |

FLORES+ is the maintained continuation of FLORES-200, with corrected translations. Our scores will therefore not be exactly comparable to papers that use the original FLORES-200. We can switch back if exact comparability matters.

## 2. Possible issues and what we did

| # | Issue | Where | What we did | Status |
| --- | --- | --- | --- | --- |
| 1 | **Duplicates across corpora.** Web crawls overlap heavily, and corpora are re-released under new names. | All tiers, mostly B | Mirrors are excluded up front. A global exact dedup across all corpora keeps the copy from the most trusted corpus (Tier A first). | Implemented, tested |
| 2 | **Test-set contamination.** FLORES comes from Wikimedia projects, so it may also appear in the wikimedia and WikiMatrix training corpora. | Training data vs. all eval sets | Any training pair is removed if either side matches an eval sentence (after normalization) or shares a 13-word sequence with one. The step refuses to run if an eval file is missing. | Implemented, tested |
| 3 | **Held-out sets must be split by document**, or sentences from the same document leak into training. But OPUS rarely gives usable document boundaries: SETIMES and TED2020 are one "document" each, and a Europarl session is about 1,800 lines. | Held-out sets | Held-out sets are drawn as whole blocks of 100 consecutive lines that never cross a document boundary. The same pairs, and any copies, are then removed from training. | Implemented, tested |
| 4 | **Greek text encoding problems.** Accent marks typed as separate characters before capitals ("΄Αρθρο"), Latin letters that look like Greek ones inside Greek words, final-sigma errors, the micro sign µ used for μ. | Mainly Tier A legal text and web text | A normalization step fixes these. Standard Unicode normalization already handles the Greek question mark and the accent variants the plan mentioned. Casing, accents and punctuation are kept. | Implemented, tested on examples |
| 5 | **Misaligned and non-parallel pairs**: a sentence matched with the wrong translation, or a translation with extra content. | Mostly Tier B, also Tier A | Rule filters: length, length ratio, numbers and URLs must match, script share, repeated words. Number matching ignores separators, because Greek writes 1.234,5 where English writes 1,234.5. | Implemented; pilot samples confirm most removals are real misalignments |
| 6 | **Wrong language** on one side. | Web data | Language ID on both sides (GlotLID), requiring confidence ≥ 0.8. Sides with fewer than 4 words are not checked, because language ID is unreliable on very short text. | Implemented; see the finding in 3.2 |
| 7 | **Fluent but inadequate translations**, which rules cannot catch. | Tiers A−, B and EUbookshop | Two neural adequacy scores, Bicleaner-AI and LaBSE similarity, stored as columns so they can be reused for set selection and ablations. | Implemented; LaBSE tested, Bicleaner not yet (see 4) |
| 8 | **One crawl dominating the data.** HPLT alone is 44% of Tier B. | Tier B | The Tier B budget is split across corpora by temperature sampling (T = 1.2), so smaller corpora get a larger share than their size alone would give. | Implemented, tested |
| 9 | **Many translations of the same sentence.** For example, Tatoeba often has several Greek versions of one English sentence. | Tatoeba, web data | At most 3 translations per identical sentence, applied to both the Greek and the English side, because each set serves both directions. In the pilot this removed 7.7% of Tatoeba. | Implemented, tested |
| 10 | **Domain imbalance**: Tier A is mostly EU legal text. | Tier A | Temperature sampling over corpora when the training sets are built; web data adds general-domain coverage. | Implemented |
| 11 | **Contamination we cannot remove.** Krikri was pretrained on 5.5B tokens of Greek–English parallel data, and its translation instruction data comes mainly from ELRC-SHARE. These probably include Europarl, EMEA, SETIMES and ELRC, so the held-out sets may not be clean for Krikri. | Krikri-based models | Cannot be fixed on our side. We will state it as a caveat and look for a test set published after Krikri's release. | Open |
| 12 | **Translated test source.** FLORES and NTREX were both written in English, so for el→en the Greek source is a translation, which is usually easier to translate. | el→en evaluation | Report it. Our held-out sets do not avoid it either: SETIMES, EMEA and TED were also written in English first. A test set written in Greek would have to be found. | Open |
| 13 | **Scale**: about 150M pairs before filtering. | All | Data is processed in parallel shards. Deduplication works on 64-bit hashes (~3.6 GB of memory for 150M pairs). Neural scoring only runs after the cheap filters have shrunk the data. Every step can resume after an interruption. | Implemented; not yet run at full scale |
| 14 | **Licences.** Several ELRC sets and OpenSubtitles have restrictive terms. | ELRC, OpenSubtitles | OpenSubtitles is off by default. | Needs a decision |

## 3. The pipeline

### 3.1 Steps

| Order | Step | What it does |
| --- | --- | --- |
| 0 | Download | Fetches each OPUS corpus and stores it unchanged, keeping each pair's origin (corpus, document, line) so it can be traced back |
| 1 | Hold-out | Draws the in-domain test sets from the raw corpora |
| 2 | Normalize | Unicode and Greek-specific cleanup (issue 4) |
| 3 | Decontaminate | Removes overlap with every evaluation set (issues 2 and 3) |
| 4 | Deduplicate | Global exact dedup, plus the cap on repeated sentences (issues 1 and 9) |
| 5 | Rule filters | Length, length ratio, script, numbers, URLs, repetition (issue 5) |
| 6 | Language ID | GlotLID on both sides (issue 6) |
| 7 | Adequacy scoring | Bicleaner-AI and LaBSE scores, on GPUs (issue 7) |
| 8 | Budget cut | Tier A kept whole; Tiers A− and B kept by score up to a budget (issue 8) |
| 9 | Build training sets | Bridge set (~4M pairs, half Tier A, half the best web data), SFT set (~150K of the cleanest human-translated pairs), and a separate pool of ~20K sources for preference training |

After every step we record the number of pairs removed per reason and per corpus, and keep 100 kept and 100 removed pairs for reading by eye. The removal-rate table for the report is produced automatically.

### 3.2 Changes from the original plan

- **Decontamination runs after normalization, not first.** Matching against test sentences only works reliably on normalized text.
- **Held-out sets are drawn as blocks within documents**, because OPUS document boundaries are missing or too coarse (issue 3).
- **The cap on repeated sentences applies to both sides**, because each training set is used for both translation directions.
- **FLORES+ replaces FLORES-200** (see 1.4).
- **WikiMatrix is scored with Bicleaner like the rest of Tier A−.** The plan's LASER-margin cut needs margin scores, which we don't think the OPUS release includes (not yet checked).

## 4. Pilot run

The whole pipeline was run on 5 corpora: SETIMES, Tatoeba, Books, a small ELRC set and GlobalVoices, 377K pairs in total. To exercise the web-data path, GlobalVoices was treated as Tier B for this run.

| Corpus | Raw | After normalize | After decontaminate | After dedup | After rules | After language ID | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SETIMES | 227,168 | 227,168 | 226,563 | 224,192 | 212,860 | 206,183 | 90.8% |
| Tatoeba | 27,103 | 27,103 | 27,101 | 24,941 | 24,727 | 24,234 | 89.4% |
| Books | 1,285 | 1,285 | 1,285 | 1,285 | 1,131 | 1,100 | 85.6% |
| ELRC-3199-antibiotic | 990 | 989 | 989 | 985 | 912 | 771 | 77.9% |
| GlobalVoices | 120,421 | 115,444 | 115,444 | 112,692 | 98,686 | 90,408 | 75.1% |

What we learned from reading the samples:

- **The held-out mechanism works as intended.** Decontamination removed exactly the 598 held-out SETIMES pairs, plus 7 pairs that overlapped them or NTREX.
- **The rule filters mostly remove real errors.** The `numbers` and `length ratio` rules catch sentences paired with the wrong translation. The largest rule removal, the Greek-script share, hits SETIMES author lines such as "Του Ζοράν Νικολόφσκι για τους Southeast European Times". These are dropped because Latin-script names make up most of the letters. That is acceptable, but it shows the threshold must be tuned on real samples.
- **The language-ID threshold depends on the model.** With the older fastText model (lid176), a 0.8 threshold rejected about 3% of perfectly good English sentences. On those same rejected sentences, GlotLID, our default, accepted 76%. The thresholds will be tuned again on the full data with GlotLID.
- **Caveats.** The pilot used lid176 instead of GlotLID, and a stand-in for Bicleaner-AI that returns random scores. Its budget and set-building numbers only show that the code runs, not real data quality.

## 5. How to proceed

### 5.1 Next steps

1. **Download everything on the GPU server**: all 54 corpora, plus FLORES+ (needs Hugging Face access approval) and NTREX.
2. **Install Bicleaner-AI in its own environment** and confirm it works with our pipeline on about 1,000 pairs.
3. **Measure throughput**, as the plan asks: score 100K pairs with Bicleaner-AI and LaBSE, then extrapolate to the size of Tiers A− and B after the cheap filters. This sets the size of the Tier B budget.
4. **Run steps 1–6 on all corpora.** Read the kept and removed samples per corpus and step, tune the thresholds, and produce the removal-rate table.
5. **Run scoring, the budget cut and set building.** Check the composition of the Bridge and SFT sets per corpus and domain.
6. **Start the baselines** (plan section 5) as soon as the SFT set exists, with B2 (Krikri + LoRA) first.

According to the plan's timeline, steps 1–5 are weeks 1–3.

### 5.2 Points to discuss

1. **Scope of the target domain.** Should conversational text be in scope? This decides whether OpenSubtitles (76.8M pairs, capped at 2M) is used.
2. **Directions.** Both directions, or el→en first?
3. **Licences.** Do the restrictive terms of some ELRC sets and of OpenSubtitles matter for the intended use?
4. **FLORES+ or FLORES-200?** FLORES+ has corrected translations; FLORES-200 is directly comparable with older papers.
5. **Compute.** How much GPU time is available? Scoring Tier B is the bottleneck of the data phase.
6. **An inconsistency in the training plan** that affects the data budget. The plan has stage 1 run 40–60K steps of 64K target tokens and calls that 1–2 epochs. But a 4M-pair Bridge set is only about 120M target tokens, so 40–60K steps would be 20–30 epochs. Either stage 1 needs far fewer steps (~2–4K), or a smaller batch, or a larger Bridge set. A 1K-step pilot will settle the step time.
