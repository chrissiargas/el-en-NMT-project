# Greek–English NMT: Project Guide

Oct 8, 2026 · @Χρηστος

## 1. Overview

The plan is to build Greek↔English MT by composing two frozen LLMs, Llama-Krikri-8B (Greek-strong) and an English-strong LLM, through learned cross-attention in the style of CALM. Before that comes a data phase that reduces OPUS's \~361.5M listed el–en pairs to a few clean, deduplicated training sets.

- **Data (sections 2–4).** What to keep from OPUS, how to filter it, and which final sets feed which training stage.
- **Model and training (sections 5–7).** The baselines that must be beaten first, the composition architecture, and a staged training recipe that mirrors the Dual-Fusion ASR curriculum.
- **Evaluation and planning (sections 8–9).** Metrics, ablations, a timeline, and open questions.

The two directions are trained and reported separately. The model that reads the source language acts as the *augmenting* encoder, and the model that writes the target language acts as the *anchor* decoder. Krikri therefore swaps roles between el→en and en→el.

**Working assumptions to confirm.** Both directions are in scope; the target is general-domain text; compute is about 4 × 80 GB GPUs, as in the ASR project. Section 9 lists what changes if any of these is wrong.

## 2. Corpus selection

Keep about 21M raw pairs of curated and lightly mined text (Tiers A and A−) plus a scored, budget-capped slice of the web-mined corpora (Tier B). Drop mirrors, broken entries and non-sentence data before running any filter. Pair counts are from the [OPUS el&en listing](https://opus.nlpl.eu/corpora-search/el&en).

### 2.1 Remove before filtering

| Problem | Corpora | Action |
| --- | --- | --- |
| Mirror pairs (identical sizes) | HPLT v3 = MultiHPLT v3 · CCMatrix v1 ≈ NLLB v1 · MaCoCu v2 = MultiMaCoCu v2 · each `ELRC-NNN` ↔ its `ELRA-W0xxx` twin · unnumbered ELRC sets ↔ numbered ones (EUIPO\_2017 ↔ 1175, EUR\_LEX ↔ 3562, EU\_publications ↔ 2872, wikipedia\_health ↔ 3058, EC\_EUROPA ↔ 3461, vaccination ↔ 2735) | Keep one of each; prefer the numbered ELRC version |
| Broken metadata (sentence count far above token count) | ELRC-EMEA v1 (13.7M listed), ELRC-presscorner\_covid, ELRC-Press\_Releases, ELRC-EUROPARL\_covid, ELRC-antibiotic | Drop; use ELRC-2711-EMEA, ELRC-3603, ELRC-3290, ELRC-3199 instead |
| Empty | Ubuntu v14.10 | Drop |
| Not sentence pairs | XLEnt (entities), LinguaTools-WikiTitles (titles) | Drop; optionally keep XLEnt as a named-entity glossary for analysis |
| UI strings and commands | KDE4, GNOME, translatewiki, tldr-pages | Drop |
| Archaic register | bible-uedin | Drop |
| Known very noisy crawl | CCAligned v1 (8.9M) | Drop |

### 2.2 What to keep

| Tier | Corpus | Pairs (raw) | Domain | Role |
| --- | --- | --- | --- | --- |
| A | DGT v2021 | 5.8M | EU legal | Light filter |
| A | EUbookshop v2 | 4.0M | EU publications | Filter harder (PDF extraction noise) |
| A | Europarl v8 | 1.3M | Parliament | Light filter |
| A | EMEA v3 + ELRC-2711-EMEA | 1.1M + 0.8M | Medical | Light filter, dedup against each other |
| A | wikimedia v20260327 | 0.9M | Wikipedia | Light filter, check against FLORES |
| A | ELRC-5067-SciPar | 0.7M | Scientific abstracts | Light filter |
| A | QED v2.0a | 0.55M | Lecture subtitles | Light filter |
| A | ELITR-ECA | 0.38M | EU audit reports | Light filter |
| A | TED2020 + NeuLab-TedTalks | 0.27M + 0.15M | Talks | Dedup against each other |
| A | SETIMES v2 | 0.23M | News | Light filter |
| A | GlobalVoices, ECB, JRC-Acquis, Tatoeba, Books, numbered ELRC Greek/Cypriot gov sets | \~0.5M combined | Mixed | Light filter |
| A− | MaCoCu v2 | 3.6M | Greek web (curated crawl) | Score-filter |
| A− | WikiMatrix v1 | 0.6M | Wikipedia, mined | Keep LASER margin ≥ 1.04 (the authors' suggested cut; tune) |
| B | HPLT v3 | 55.6M | Web | Score, keep top-K |
| B | CCMatrix v1 | 49.3M | Web, mined | Score, keep top-K |
| B | ParaCrawl v9 | 21.4M | Web | Score, keep top-K |
| B′ | OpenSubtitles v2024 | 76.8M | Film dialogue | Only if conversational text is in scope; strict filter, cap at \~2M |

Tier A is dominated by EU legal and institutional text. Tier B supplies general-domain coverage, so a domain-balanced sample of it is needed even though it is noisier. Expect the web-mined corpora to overlap heavily with each other; most of their raw volume disappears at cross-corpus deduplication.

## 3. Filtering pipeline

Run the cheap filters on everything, then the neural scorer only on what survives, and finally cut each corpus to a budget by score. [OpusFilter](https://pypi.org/project/opusfilter/) covers most of steps 2–6 from one YAML config. [OpusCleaner](https://github.com/hplt-project/OpusCleaner) is an interactive alternative for inspecting samples per corpus.

1. **Hold out test data first.** Before any processing, carve out document-level dev/test splits from the in-domain corpora you will report on (section 4.4). Then remove every pair whose source or target matches FLORES-200, NTREX-128 or those held-out sets.
2. **Normalize.**
   - Unicode NFC; strip HTML, control and zero-width characters; collapse whitespace.
   - Greek specifics: unify the Greek question mark (U+037E) with `;`, unify precomposed and combining tonos, and fix final sigma (ς) at word ends.
   - Keep casing and punctuation. Do not lowercase or strip accents.
3. **Deduplicate.**
   - Exact dedup on the normalized (el, en) pair across all corpora, Tier A first so the curated copy is the one kept.
   - Cap repeated sources: at most \~3 distinct translations per identical source sentence.
   - Optional near-duplicate removal with MinHash on Tier B.
4. **Rule-based filters** (starting thresholds; tune on samples):
   - Length 1–150 words per side; drop pairs with source identical to target.
   - Character-length ratio between 0.5 and 2.0.
   - Script ratio: ≥ 70% of letters Greek on the el side and ≥ 90% Latin on the en side.
   - Numbers and URLs must match across sides.
   - Non-alphabetic character share ≤ 30%; no lines that are only punctuation or digits.
   - Repeated-token check (a word repeated 5+ times in a row signals crawl garbage).
5. **Language ID** on both sides with OpenLID or GlotLID (fastText `lid.176` as a fallback). Require the correct label with confidence ≥ 0.8.
6. **Adequacy scoring** (Tier A−, B and B′ only; Tier A is mostly human-translated):
   - Primary: [Bicleaner-AI en–el](https://huggingface.co/bitextor/bicleaner-ai-full-en-el), score in \[0, 1\].
   - Secondary: LaBSE cosine similarity. Pairs where the two scorers disagree strongly are worth inspecting by hand.
   - Keep the scores as columns. Do not throw them away after thresholding; sections 4 and 7 reuse them.
7. **Budget cut.**
   - Tier A: keep everything that passes steps 1–5. Apply Bicleaner ≥ 0.5 to EUbookshop only.
   - Tier A−/B: rank by Bicleaner score and keep the top-K. Start with Bicleaner ≥ 0.7, then lower or raise it until the total fits the budget in section 4.
   - Sample Tier B per source corpus so no single crawl dominates the mix.

**Check every step by eye.** For each corpus, read 100 random pairs that were kept and 100 that were removed at each stage. Record the removal rate per step and per corpus in a table; it becomes a results table for the report.

**Throughput.** Steps 1–5 are CPU-bound and fast. Step 6 is the bottleneck: Bicleaner-AI on \~125M Tier B pairs is GPU work. Score only after steps 1–5 have shrunk Tier B, and run it as array jobs, one shard per GPU.

## 4. Final data sets

The filtered pool feeds four sets, one per training stage plus evaluation. Each set is built once for both directions, by swapping source and target. The sizes are starting points chosen for an 8B-scale LLM, not upper limits.

| Set | Used in | Size | Built from | Selection rule |
| --- | --- | --- | --- | --- |
| Bridge set | Stage 1: train the cross-attention bridge | 3–5M pairs | \~50% filtered Tier A, \~50% top Bicleaner Tier A−/B | Domain-balanced sample; temperature sampling over corpora (T ≈ 1.2, as in the ASR mixture) |
| SFT set | Stage 2: LoRA on the anchor | 50K–200K pairs | Human-translated corpora (TED, Europarl, GlobalVoices, SETIMES, QED, Tatoeba, wikimedia) + the top \~0.5% of Tier B where Bicleaner and LaBSE agree | Highest scores only; length 5–80 words; may overlap the Bridge set |
| Preference set | Stage 3: CPO | \~20K source sentences | A separate slice of SFT-quality sources | Per source: reference + model outputs + one strong external system; chosen/rejected ranked by a reference-free metric |
| Dev/test | Model selection and reporting | see 4.4 | FLORES-200, NTREX-128, held-out in-domain sets | Never filtered against, never trained on |

### 4.1 Why the SFT set is small

For LLM-based translation, a small set of very clean parallel data works better than millions of noisy pairs. ALMA reported that fine-tuning on large noisy parallel data can hurt an LLM's translation quality. The Bridge set is larger because the new cross-attention layers start from scratch and need volume, while the frozen LLMs keep their language knowledge.

### 4.2 Preference data

Generate candidates per source: the reference, 4–8 sampled outputs of your Stage 2 model, and one output from a strong external system (e.g. NLLB-3.3B). Rank them with MetricX-QE or CometKiwi and use best/worst as chosen/rejected. **Do not rank with the same metric you report as primary**, otherwise the gains partly reflect metric overfitting.

### 4.3 Contamination

- Remove exact and near matches (normalized string match, then 13-gram overlap) of every dev/test sentence from all training sets.
- FLORES-200 comes from Wikimedia projects, so check the wikimedia and WikiMatrix corpora especially.
- Krikri's pretraining included 5.5B tokens of parallel data that you cannot inspect. State this as a caveat when comparing against public benchmarks, as you did for Common Voice in the ASR paper.

### 4.4 Dev and test sets

| Set | Sentences | Use |
| --- | --- | --- |
| FLORES-200 dev | 997 | Checkpoint selection, threshold tuning |
| FLORES-200 devtest | 1,012 | Main test set, general domain |
| NTREX-128 | 1,997 | Second general-domain test set (news) |
| Held-out in-domain (Europarl, EMEA, TED, SETIMES) | \~2,000 each | Domain robustness; split by document, before deduplication |

## 5. Baselines to beat

The composed model is only worth reporting if it beats **Krikri + LoRA trained on the same data with the same number of trainable parameters** (B2 below). Build B0–B2 first; they take days, not weeks, and they calibrate every later number.

| ID | System | Training | What it shows |
| --- | --- | --- | --- |
| B0a | OPUS-MT el–en / en–el (Helsinki-NLP) | None | Public dedicated-NMT reference |
| B0b | NLLB-200 distilled-600M and 3.3B | None | Multilingual NMT reference |
| B0c | Krikri-8B-Instruct and the English anchor LLM, prompted (0-shot and 5-shot) | None | What each LLM does on its own |
| B1 | NLLB-200 1.3B, fine-tuned | Full fine-tune on the Bridge set, then the SFT set | Strong conventional NMT with your data |
| B2 | Krikri + LoRA | SFT set, then CPO; LoRA rank chosen so trainable parameters match the composed model's bridge + LoRA | **The control the composition must beat** |
| B3 | English anchor + LoRA | Same as B2 | Whether the English model alone is already competitive (mainly for el→en) |
| B4 | B2 with QE reranking or MBR over 16 samples | Inference only | How much decoding alone adds; apply the same decoding to the composed model |

CALM's paper used the same kind of control: it fine-tuned the anchor with LoRA on the composition data to show the gains did not come from the data alone. Report B2 and the composed model side by side in every table.

## 6. Composition architecture

Both LLMs stay frozen and only a \~150M-parameter bridge is trained. The source-language model reads the sentence once; its states enter the target-language model as prefix tokens and through gated cross-attention at three depths. This is the Dual-Fusion layout with Whisper replaced by an LLM.

&#91;embedded content: composed model, el→en · bridge trained, both LLMs frozen\]

The augmenting model runs once per sentence without gradients, like Whisper in the ASR system. This "encoder mode" is a cheaper variant of CALM, where the augmenting model also processes every generated token.

### 6.1 Roles per direction

| Direction | Augmenting model (reads) | Anchor model (writes) | Where the gain should come from |
| --- | --- | --- | --- |
| el→en | Krikri | English LLM | Krikri's Greek understanding fed to a model that writes better English |
| en→el | English LLM | Krikri | The English model's reading of English fed to the stronger Greek writer |

Krikri is already strong in English. A same-size English model may therefore add little in either direction, which is exactly what the B2 control in section 5 tests.

### 6.2 Choosing the English model

- **Llama-3.1-8B-Instruct.** Cheapest, and its tokenizer is a subset of Krikri's. But Krikri was trained from it, so their representations are similar and may carry little complementary information.
- **A different 7–9B family with strong English.** More complementary representations at the same cost. Check the current open models when you start.
- **A larger English model (≥30B) as the el→en anchor.** Highest ceiling. Fits in memory only when loaded frozen in 4-bit, as in QLoRA.

Pick after a one-week check of zero-shot quality (B0c) on FLORES-200 dev.

### 6.3 What carries over from Dual-Fusion

| Component | In the ASR system | In the MT system |
| --- | --- | --- |
| Prompt-level prefix (MLP adapter) | Whisper last layer, downsampled k=5 | Augmenting model's last layer; no downsampling needed |
| Asymmetric downsampling | Needed for 50 Hz audio frames | Drop: text tokens are already compact |
| Layer-wise fusion | Utterance-level attention over Whisper layers | Same, over the augmenting LLM's layers; also try per-token weights |
| Multi-depth gated cross-attention | \~30/60/90% depth | Same; layers 10/20/30 for a 32-layer anchor |
| Zero-initialised tanh gate | Stable start | Keep unchanged |
| Text perturbation (p = 10%) | Masks target history | Keep as target-side word dropout; re-tune p |
| SpecAugment | Input noise on audio | Optional source-side token dropout |
| Diagonality analysis | Monotonic alignment | Not monotonic in MT; compare cross-attention with word alignments (awesome-align or SimAlign) |

### 6.4 Design choice to ablate: does the anchor also read the source text?

- **(a) CALM-style (default).** The anchor's prompt also contains the source sentence as plain text, plus the bridge.
- **(b) Dual-Fusion-style.** The anchor sees only the prefix tokens and cross-attention, never the source text.

(b) shows whether the bridge alone carries the meaning; (a) is likely stronger for el→en, because the anchor can copy names and numbers directly.

### 6.5 Memory

Two frozen 8B models in bf16 take \~32 GB of weights, plus activations for the anchor's forward and backward passes. Loading both in 4-bit cuts weights to roughly 10 GB, which leaves room for longer batches on 80 GB GPUs.

## 7. Training recipe

Training follows the ASR curriculum: first train the bridge with both LLMs frozen, then freeze the bridge and add LoRA, then add a short preference stage. Each direction is a separate run with its own bridge, because the two models swap roles.

| Stage | Trainable | Data | Starting hyperparameters |
| --- | --- | --- | --- |
| 1. Bridge | Adapter, layer fusion, cross-attention, gates (\~150M) | Bridge set (3–5M pairs) | AdamW, LR 1e-4, cosine, 2K warmup; bf16; \~64K target tokens per step; 40–60K steps (1–2 epochs); target word dropout p = 0.1 |
| 2. LoRA SFT | QLoRA on the anchor (r = 64, α = 128; attention + MLP projections); bridge frozen | SFT set (50K–200K pairs) | LR 1e-4, 1–2 epochs, effective batch 64 sentences; early stop on FLORES dev COMET |
| 3. Preference (CPO) | Same LoRA, continued | Preference set (\~20K sources) | Small LR (\~1e-5), 1 epoch, β ≈ 0.1; check that FLORES dev COMET rises and length stays stable |

### 7.1 Details that differ from ASR training

- **Loss only on target tokens.** Mask the prompt and the source text out of the loss.
- **Fixed prompt template** for training and inference, e.g. `Translate the following text from Greek to English.\nGreek: {src}\nEnglish:`.
- **Noisy-to-clean ordering in stage 1.** Use the full Bridge mix for the first \~80% of steps, then only Tier A and the top Bicleaner slice for the rest.
- **Sentence length.** Pack or bucket by length; MT sentences are much shorter than the ASR sequences, so token-based batching wastes less padding than sentence-based batching.
- **Validation.** Every 2K steps, decode FLORES-200 dev greedily and track chrF++ (cheap). Select final checkpoints on COMET.

### 7.2 Decoding

- Beam search, beam 4–5, length penalty 1.0, maximum length about 2 × source length + 20 tokens.
- **Drop the ASR settings for repetition.** A repetition penalty and n-gram blocking hurt MT, where repeated words and phrases are often correct. Re-enable them only if you see degenerate loops.
- Then test QE reranking or MBR over 16 samples (B4) on top of the best checkpoint.

### 7.3 Order of work

1. Stage 1 for el→en with the default design (a), plus B2 at matched parameters.
2. If the stage-1 model is at least close to B2 on FLORES dev, continue to stages 2–3. If not, inspect the gates: a gate that stays near zero means the anchor ignores the bridge.
3. Repeat for en→el.
4. Run the ablations in section 8 with stage 1 only, at a fixed step budget (as with the 12K-step ASR ablations).

## 8. Evaluation and ablations

Report COMET-22 as the primary metric with chrF++ and BLEU beside it, on FLORES-200 devtest, NTREX-128 and the in-domain held-out sets, for both directions. Test every comparison against B2 for significance.

### 8.1 Metrics

| Metric | Type | Use |
| --- | --- | --- |
| COMET-22 | Neural, reference-based | Primary |
| chrF++ | Character n-gram | Surface metric; suits Greek morphology better than BLEU |
| BLEU (sacreBLEU, report the signature) | Word n-gram | For comparison with older work |
| MetricX-24 or xCOMET | Neural, reference-based | Second neural metric; must not be the metric that ranked the preference data |
| CometKiwi | Reference-free QE | Flags likely hallucinations and omissions |
| MQM-style human check | Human | \~200 sentences per direction, errors tagged by type (accuracy, fluency, terminology) |

Significance: paired bootstrap resampling (sacreBLEU and `comet-compare` both implement it).

### 8.2 Ablations

Run each against the full stage-1 model at a fixed step budget, el→en first.

- [ ] Without prefix tokens (cross-attention only)
- [ ] Without cross-attention (prefix only)
- [ ] One injection depth (L20) instead of three
- [ ] Final augmenting layer only vs static layer weights vs layer-wise attention
- [ ] Without the tanh gate
- [ ] Target word dropout p = 0, 0.1, 0.2
- [ ] Design (a) source text in the anchor prompt vs (b) bridge only
- [ ] English model choice (same family vs different family)
- [ ] Bridge set size 1M / 3M / 5M at a fixed Bicleaner threshold
- [ ] Bicleaner threshold 0.5 / 0.7 / 0.85 at a fixed size

The last two form a small data-quality study, which is useful even if the composition gives only modest gains.

### 8.3 Analysis

- **Gate values** per depth over training: do they open, and which depth matters most?
- **Cross-attention vs word alignments:** compare attention maps with awesome-align output to see whether a depth specialises in alignment, as layer 25 did for Racka.
- **Error types:** hallucination and omission rates (low CometKiwi, abnormal length ratio), accuracy on numbers and named entities, and Greek-specific errors in en→el (agreement, tonos, final sigma).
- **Per-domain results:** legal, medical, talks and news held-out sets.

## 9. Timeline, risks and open questions

A first full pass, through both directions and the ablations, takes about 14 weeks. The data phase is front-loaded so baselines can start in week 3.

| Weeks | Work | Output |
| --- | --- | --- |
| 1–2 | Download Tier A/A−/B; remove mirrors and broken sets; hold out test data; normalization, rules, language ID | Clean pool before scoring; removal-rate table |
| 2–3 | Bicleaner and LaBSE scoring of Tier A−/B; budget cut; build the four sets | Bridge, SFT, preference and dev/test sets |
| 3–4 | Baselines B0–B4; English model choice | Baseline table for both directions |
| 5–7 | Stage 1, el→en, design (a) | First composed model |
| 8–9 | Stages 2–3, el→en; full evaluation | el→en results vs B2 |
| 10–12 | en→el all stages; ablations (stage 1, fixed budget) | Ablation table |
| 13–14 | Analysis, MQM check, write-up | Draft report or paper |

**Compute.** Measure before budgeting: score 100K pairs with Bicleaner-AI to estimate throughput, and run a 1K-step stage-1 pilot to estimate step time. Scale the Tier B budget and the step counts from those two numbers.

### 9.1 Risks

| Risk | Early sign | Response |
| --- | --- | --- |
| Composition adds nothing over B2, because Krikri is already bilingual | Stage-1 dev COMET at or below B2 | Try a larger or different-family English anchor for el→en; the data-quality study still stands as a result |
| The anchor ignores the bridge ("lazy LLM") | Gates stay near zero; design (b) fails to train | Raise target word dropout; add a short design-(b) warm-up so the bridge must carry the meaning |
| Benchmark contamination in Krikri's pretraining | Unusually high FLORES scores for B0c | Lean on held-out in-domain sets and a test set published after Krikri's release |
| Neural scoring of Tier B is too slow | Throughput test in week 2 | Score only after the rule filters; subsample each crawl before scoring |
| Metric overfitting from CPO | COMET rises while MetricX and the human check do not | Keep ranking and reporting metrics separate (section 4.2) |

### 9.2 Open questions for the supervisor

- [ ] How much compute is available (GPU type and count, hours per week)?
- [ ] Both directions, or el→en first?
- [ ] Target domain: general, or a specific one (legal, medical, conversational)?
- [ ] Goal: a paper (which venue and deadline) or a working system?
- [ ] Licence constraints: several ELRC sets and OpenSubtitles have restrictive terms; does that matter for the intended use?

## 10. References

Pages opened while writing this guide:

- [OPUS el&en corpora listing](https://opus.nlpl.eu/corpora-search/el&en)
- [OpusFilter](https://pypi.org/project/opusfilter/) · [OpusCleaner](https://github.com/hplt-project/OpusCleaner)
- [Bicleaner-AI en–el model](https://huggingface.co/bitextor/bicleaner-ai-full-en-el)
- [Krikri: Advancing Open Large Language Models for Greek](https://arxiv.org/pdf/2505.13772) (Findings of EMNLP 2025)
- [Llama-Krikri-8B-Instruct model card](https://huggingface.co/ilsp/Llama-Krikri-8B-instruct)

Methods cited from memory; look up the exact versions before citing them in a paper:

- Bansal et al., *LLM Augmented LLMs: Expanding Capabilities through Composition* (CALM), ICLR 2024
- Xu et al., *A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models* (ALMA), ICLR 2024
- Xu et al., *Contrastive Preference Optimization* (CPO / ALMA-R), ICML 2024
- Alves et al., *Tower: An Open Multilingual Large Language Model for Translation-Related Tasks*, 2024
- Yoon et al., *LangBridge*, ACL 2024 · Huang et al., *MindMerger*, NeurIPS 2024 · *LayAlign*, 2025: encoder-to-LLM bridging
- Alayrac et al., *Flamingo*, NeurIPS 2022 (gated cross-attention)
- Schwenk et al., *WikiMatrix*, EACL 2021
- NLLB Team, *No Language Left Behind* (FLORES-200), 2022 · Federmann et al., *NTREX-128*, 2022
- Rei et al., *COMET-22* and *CometKiwi*, WMT 2022
- Siargkas et al., *Unifying Speech and Language through Continuous Dual-Fusion for Medium-Resource Languages* (the ASR system this design adapts)
