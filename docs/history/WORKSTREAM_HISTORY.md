# Tokenizer Workstream History

## Overview

This document reconstructs the full development of the tokenizer optimization pipeline for the Parameter Golf challenge.

The goal of this workstream was to move from a baseline SentencePiece tokenizer to a **corpus-aware, engineered tokenizer system**, with the intent of reducing validation bits-per-byte (val_bpb).

This was not a single script or idea—it evolved through multiple phases:

1. Baseline establishment
2. Dataset analysis
3. N-gram mining
4. Candidate curation
5. Vocabulary engineering
6. Tokenizer implementation (DP)
7. Dataset export
8. Variant search
9. Training integration

---

# Phase 0 — Baseline SentencePiece Pipeline

## Objective
Establish a reliable baseline tokenizer and evaluation framework.

## Key Scripts
- `01_build_merge_list.py`
- `02_build_boost_corpus.py`
- `03_build_training_corpus.py`
- `04_train_tokenizer.py`
- `05_test_tokenizer.py`
- `06_validate_roundtrip.py`
- `07_compare_token_counts.py`

## Outputs
- `fineweb_1024_bpe.model`
- `fineweb_1024_bpe.vocab`
- `baseline_vocab*.jsonl`

## What Actually Happened

At this stage, the goal was not optimization—it was **control**.

We needed:
- A tokenizer that worked
- A way to validate correctness (roundtrip)
- A way to compare token counts across approaches

This phase defined:
> what “normal” looks like

Everything later is measured against this.

---

# Phase 1 — Dataset & Shard Analysis

## Objective
Understand the structure and variability of the FineWeb dataset.

## Key Scripts
- `analyze_fineweb_shards.py`
- `score_shards.py`

## Outputs
- `shard_report.json`
- `shard_scores.csv`
- `top_first_order.txt`
- `bottom_first_order.txt`

## What Actually Happened

This was the first shift away from blind experimentation.

Instead of assuming:
> all data is equal

We started asking:
- Which shards are dense vs noisy?
- Which ones contain more structure?
- Does ordering matter?

This introduced the idea that:
> data ordering and quality might be a lever, not just tokenizer design

---

# Phase 2 — N-gram Mining & Opportunity Discovery

## Objective
Quantify how much compression opportunity exists in the dataset.

## Key Scripts
- `mine_fineweb_ngrams.py`
- `decode_bigrams.py`
- `decode_trigrams.py`
- `estimate_merge_savings.py`
- `estimate_ngram_merge_savings.py`
- `simulate_merge_tokenization.py`

## Outputs
- `ngram_top.csv`
- `decoded_bigrams.jsonl`
- `decoded_trigrams.jsonl`
- `merge_savings_*.json`
- `merge_savings_report.json`
- `retokenization_simulation_report.json`

## What Actually Happened

This phase answered a critical question:

> Is there actually something to optimize?

Findings:
- Bigrams contain the majority of savings (~6–7%)
- Trigrams add smaller incremental gains
- Naive savings estimates overstate real gains
- Overlap-aware (greedy) savings are more realistic

This phase turned:
> intuition → measurable opportunity

---

# Phase 3 — Candidate Filtering & Curation

## Objective
Convert raw n-grams into usable vocabulary candidates.

## Key Scripts
- `filter_bigram_candidates.py`
- `filter_bigram_candidates_v2.py`
- `filter_merge_candidates.py`
- `filter_merge_candidates_v2.py`
- `split_merge_candidates.py`
- `merge_candidate_files.py`
- `auto_curate_subwords.py`
- `auto_curate_subwords_v2.py`

## Outputs
- `filtered_bigram_candidates_*.jsonl`
- `filtered_trigram_candidates_*.jsonl`
- `merge_candidates_*.jsonl`
- `subword_auto_curation_*.csv`
- curated / rejected candidate sets

## What Actually Happened

Raw frequency is not enough.

This phase removed:
- junk merges
- redundant candidates
- low-value tokens

And introduced:
- curated subwords
- structured candidate classes (phrases vs subwords)

This is where the system moved from:
> “what appears often”  
to  
> “what deserves a vocab slot”

---

# Phase 4 — Vocabulary Engineering

## Objective
Construct custom vocabularies using curated candidates.

## Key Scripts
- `10_build_vocab_draft.py`
- `11_build_full_vocab.py`
- `14_build_hybrid_vocab.py`
- `15_build_hybrid_vocab_v2.py`
- `16_build_hybrid_vocab_v3.py`
- `build_hybrid_grid.py`
- `build_hybrid_local_search.py`

## Outputs
- `custom_vocab_full_*.jsonl`
- `custom_vocab.jsonl`
- `hybrid_vocab_1024.jsonl`
- `vocab_variants/*.jsonl`
- `analysis/grid_vocabs*/`

## What Actually Happened

This is the turning point.

Instead of:
> letting SentencePiece decide the vocab

We started:
> explicitly designing the vocab

This included:
- balancing phrases vs subwords
- controlling fallback tokens
- experimenting with vocab size allocation

This phase created the first **real alternatives** to SP.

---

# Phase 5 — DP Tokenizer & Evaluation

## Objective
Implement and validate a custom tokenizer.

## Key Scripts
- `dp_tokenizer_lib.py`
- `dp_tokenizer_eval.py`
- `check_tokenizer_parity.py`
- `compare_baseline_vs_custom.py`
- `compare_vocab_to_baseline.py`
- `analyze_tokenizer_gaps.py`
- `inspect_variant_docs.py`

## Outputs
- `tokenizer_gap_report.txt`
- token comparison logs
- document-level diagnostics

## What Actually Happened

This phase exposed reality.

Key findings:
- Custom tokenizer can approach baseline (~0.5–0.7% gap)
- Main weakness: fallback tokens
- Some documents strongly favor SP segmentation

This phase answered:
> “Where exactly are we losing?”

---

# Phase 6 — Dataset Export (DP)

## Objective
Make the custom tokenizer usable in training.

## Key Scripts
- `export_custom_dp_dataset_mp.py`
- `export_custom_dp_dataset_v2.py`
- `export_custom_dp_dataset_v3.py`
- `export_custom_dp_dataset_v4.py`

## Outputs
- `fineweb_train_*.bin` (custom)
- `fineweb_val_*.bin` (custom)

## What Actually Happened

This is where the tokenizer became **real**.

Before this:
- everything was analysis

After this:
- the model actually trains on the tokenizer

---

# Phase 7 — Variant Generation & Search

## Objective
Explore the tokenizer design space systematically.

## Key Scripts
- `generate_vocab_variants.py`
- `score_vocab_candidate.py`
- `rank_variant_scores.py`
- `analyze_variant_tokenizers.py`
- `eval_hybrid_grid.py`
- `compare_vocab_runs.py`
- `compare_run_logs.py`

## Outputs
- `vocab_variants/*.jsonl`
- ranked score outputs
- grid evaluation results

## What Actually Happened

This phase replaced:
> intuition-driven iteration

with:
> structured search

Instead of asking:
> “Is this vocab good?”

We asked:
> “Which region of the design space is good?”

---

# Phase 8 — Training Integration

## Objective
Leverage tokenizer improvements in model training.

## Key Scripts
- `train_gpt.py`
- `train_gpt_custom_v1_locked.py`
- `train_gpt_custom_v2_init_locked.py`
- `train_gpt_custom_v3_bigram.py`
- `train_gpt_mlx.py`

## What Actually Happened

Tokenizer work started influencing:
- embedding initialization
- bigram-aware features
- architecture choices

This is where:
> tokenizer → model interaction begins

---

# Final State

Current status:

- Custom tokenizer is within ~0.5–0.7% of SentencePiece
- Majority of compression gains identified and captured
- Remaining gap largely due to fallback inefficiencies
- Full pipeline exists from mining → training