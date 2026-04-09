# SCRIPT INDEX

## Overview

This document provides a complete mapping of all scripts in the tokenizer workstream.

Each script is categorized by:
- Phase (where it fits in the pipeline)
- Purpose (what it actually does)
- Inputs (what it consumes)
- Outputs (what it produces)
- Status (Active / Deprecated / Experimental)

---

# Legend

- **Active** → part of current or recent pipeline
- **Deprecated** → replaced by newer version
- **Experimental** → used for testing or exploration

---

# 00 — Baseline SentencePiece Pipeline

## 01_build_merge_list.py
- **Phase:** Baseline
- **Purpose:** Generate merge candidates or phrase seeds for tokenizer shaping
- **Inputs:** Raw text / corpus slices
- **Outputs:** Merge list (txt/json)
- **Status:** Deprecated

---

## 02_build_boost_corpus.py
- **Phase:** Baseline
- **Purpose:** Amplify selected phrases in training corpus
- **Inputs:** Raw corpus + merge list
- **Outputs:** Boosted corpus `.txt`
- **Status:** Deprecated

---

## 03_build_training_corpus.py / .sh
- **Phase:** Baseline
- **Purpose:** Assemble tokenizer training corpus
- **Inputs:** Dataset shards
- **Outputs:** Training corpus `.txt`
- **Status:** Deprecated

---

## 04_train_tokenizer.py
- **Phase:** Baseline
- **Purpose:** Train SentencePiece tokenizer
- **Inputs:** Training corpus
- **Outputs:**
  - `fineweb_1024_bpe.model`
  - `fineweb_1024_bpe.vocab`
- **Status:** Active (baseline reference)

---

## 05_test_tokenizer.py
- **Phase:** Baseline
- **Purpose:** Verify tokenization behavior
- **Inputs:** Sample text
- **Outputs:** Logs
- **Status:** Deprecated

---

## 06_validate_roundtrip.py
- **Phase:** Baseline
- **Purpose:** Ensure encode/decode correctness
- **Inputs:** Tokenized sequences
- **Outputs:** Validation logs
- **Status:** Active

---

## 07_compare_token_counts.py
- **Phase:** Baseline
- **Purpose:** Compare token counts across tokenizers
- **Inputs:** Tokenized outputs
- **Outputs:** Comparison metrics
- **Status:** Active

---

## build_sp_plus_phrases.py
- **Phase:** Baseline Variant
- **Purpose:** Inject phrase tokens into SP vocab
- **Inputs:** SP vocab + phrase list
- **Outputs:** Modified vocab
- **Status:** Experimental

---

## build_sp_shaped_variants.py
- **Phase:** Baseline Variant
- **Purpose:** Generate SP-like vocab variants
- **Inputs:** Baseline vocab
- **Outputs:** Variant vocabs
- **Status:** Deprecated

---

## build_sp_shaped_variants_safe.py
- **Phase:** Baseline Variant
- **Purpose:** Safer SP-shaped vocab modification
- **Inputs:** Baseline vocab
- **Outputs:** Variant vocabs
- **Status:** Active

---

# 10 — Dataset Analysis

## analyze_fineweb_shards.py
- **Phase:** Dataset Analysis
- **Purpose:** Compute shard-level statistics
- **Inputs:** Dataset shards
- **Outputs:**
  - `shard_report.json`
  - summary metrics
- **Status:** Active

---

## score_shards.py
- **Phase:** Dataset Analysis
- **Purpose:** Rank shards based on quality metrics
- **Inputs:** shard stats
- **Outputs:** `shard_scores.csv`
- **Status:** Active

---

# 20 — N-gram Mining

## mine_fineweb_ngrams.py
- **Phase:** Mining
- **Purpose:** Extract frequent n-grams
- **Inputs:** dataset shards
- **Outputs:**
  - `ngram_top.csv`
  - `ngram_candidates.jsonl`
- **Status:** Active

---

## decode_ngrams.py
- **Phase:** Mining
- **Purpose:** Convert token sequences → readable text
- **Inputs:** encoded n-grams
- **Outputs:** decoded text
- **Status:** Deprecated

---

## decode_bigrams.py
- **Phase:** Mining
- **Purpose:** Decode bigram tokens
- **Inputs:** bigram ids
- **Outputs:** `decoded_bigrams.jsonl`
- **Status:** Active

---

## decode_trigrams.py
- **Phase:** Mining
- **Purpose:** Decode trigram tokens
- **Inputs:** trigram ids
- **Outputs:** `decoded_trigrams.jsonl`
- **Status:** Active

---

## estimate_merge_savings.py
- **Phase:** Mining
- **Purpose:** Estimate token savings (naive + greedy)
- **Inputs:** candidate merges
- **Outputs:** `merge_savings_*.json`
- **Status:** Active

---

## estimate_ngram_merge_savings.py
- **Phase:** Mining
- **Purpose:** Savings specific to n-gram-derived merges
- **Inputs:** n-gram candidates
- **Outputs:** savings reports
- **Status:** Active

---

## simulate_merge_tokenization.py
- **Phase:** Mining
- **Purpose:** Simulate tokenizer behavior with merges
- **Inputs:** candidate merges
- **Outputs:** simulation reports
- **Status:** Active

---

# 30 — Candidate Curation

## filter_bigram_candidates.py
- **Phase:** Curation
- **Purpose:** Filter low-quality bigrams
- **Inputs:** raw bigrams
- **Outputs:** filtered jsonl
- **Status:** Deprecated

---

## filter_bigram_candidates_v2.py
- **Phase:** Curation
- **Purpose:** Improved bigram filtering
- **Inputs:** raw bigrams
- **Outputs:** filtered jsonl
- **Status:** Active

---

## filter_merge_candidates.py
- **Phase:** Curation
- **Purpose:** Generic merge filtering
- **Status:** Deprecated

---

## filter_merge_candidates_v2.py
- **Phase:** Curation
- **Purpose:** Improved merge filtering
- **Status:** Active

---

## split_merge_candidates.py
- **Phase:** Curation
- **Purpose:** Separate candidates into categories
- **Outputs:** phrase/subword splits
- **Status:** Active

---

## merge_candidate_files.py
- **Phase:** Curation
- **Purpose:** Combine candidate sources
- **Status:** Active

---

## auto_curate_subwords.py
- **Phase:** Curation
- **Purpose:** Subword scoring
- **Status:** Deprecated

---

## auto_curate_subwords_v2.py
- **Phase:** Curation
- **Purpose:** Improved subword selection
- **Outputs:** `subword_auto_curation.csv`
- **Status:** Active

---

## phrase_utils.py
- **Phase:** Utility
- **Purpose:** Helper functions
- **Status:** Active

---

# 40 — Vocabulary Construction

## 10_build_vocab_draft.py
- **Phase:** Vocab Build
- **Purpose:** Initial vocab assembly
- **Status:** Deprecated

---

## 11_build_full_vocab.py
- **Phase:** Vocab Build
- **Purpose:** Expand to full vocab
- **Status:** Deprecated

---

## 14_build_hybrid_vocab.py
- **Phase:** Vocab Build
- **Purpose:** Early hybrid vocab
- **Status:** Deprecated

---

## 15_build_hybrid_vocab_v2.py
- **Phase:** Vocab Build
- **Purpose:** Improved hybrid vocab
- **Status:** Deprecated

---

## 16_build_hybrid_vocab_v3.py
- **Phase:** Vocab Build
- **Purpose:** Current hybrid vocab logic
- **Outputs:** `custom_vocab_full_*.jsonl`
- **Status:** Active

---

## build_hybrid_grid.py
- **Phase:** Vocab Search
- **Purpose:** Grid search vocab configs
- **Outputs:** grid results
- **Status:** Active

---

## build_hybrid_local_search.py
- **Phase:** Vocab Search
- **Purpose:** Local optimization
- **Status:** Active

---

## generate_vocab_variants.py
- **Phase:** Vocab Search
- **Purpose:** Create multiple variants
- **Outputs:** `vocab_variants/*.jsonl`
- **Status:** Active

---

## export_top_variants.py
- **Phase:** Vocab Search
- **Purpose:** Export best variants
- **Status:** Active

---

# 50 — Tokenizer Evaluation

## dp_tokenizer_lib.py
- **Phase:** Tokenizer Core
- **Purpose:** DP tokenizer implementation
- **Status:** Active

---

## dp_tokenizer_eval.py
- **Phase:** Evaluation
- **Purpose:** Evaluate tokenizer performance
- **Status:** Active

---

## check_tokenizer_parity.py
- **Phase:** Evaluation
- **Purpose:** Verify correctness
- **Status:** Active

---

## compare_baseline_vs_custom.py
- **Phase:** Evaluation
- **Purpose:** Direct comparison
- **Status:** Active

---

## compare_vocab_to_baseline.py
- **Phase:** Evaluation
- **Purpose:** Vocab comparison
- **Status:** Active

---

## analyze_tokenizer_gaps.py
- **Phase:** Evaluation
- **Purpose:** Identify failure cases
- **Outputs:** `tokenizer_gap_report.txt`
- **Status:** Active

---

## analyze_variant_tokenizers.py
- **Phase:** Evaluation
- **Purpose:** Compare variants
- **Status:** Active

---

## inspect_variant_docs.py
- **Phase:** Evaluation
- **Purpose:** Doc-level inspection
- **Status:** Active

---

## score_vocab_candidate.py
- **Phase:** Evaluation
- **Purpose:** Score candidates
- **Status:** Active

---

## rank_variant_scores.py
- **Phase:** Evaluation
- **Purpose:** Rank variants
- **Status:** Active

---

## eval_hybrid_grid.py
- **Phase:** Evaluation
- **Purpose:** Evaluate grid results
- **Status:** Active

---

## compare_vocab_runs.py
- **Phase:** Evaluation
- **Purpose:** Compare runs
- **Status:** Active

---

## compare_run_logs.py
- **Phase:** Evaluation
- **Purpose:** Compare logs
- **Status:** Active

---

# 60 — Dataset Export

## export_custom_dp_dataset_mp.py
- **Phase:** Export
- **Purpose:** Multiprocess export
- **Status:** Deprecated

---

## export_custom_dp_dataset_v2.py
- **Phase:** Export
- **Purpose:** Improved export
- **Status:** Deprecated

---

## export_custom_dp_dataset_v3.py
- **Phase:** Export
- **Purpose:** Further refinement
- **Status:** Deprecated

---

## export_custom_dp_dataset_v4.py
- **Phase:** Export
- **Purpose:** Current export pipeline
- **Outputs:** `.bin shards`
- **Status:** Active

---

# 70 — Training

## train_gpt.py
- **Phase:** Training
- **Purpose:** Baseline training
- **Status:** Active

---

## train_gpt_custom_v1_locked.py
- **Phase:** Training
- **Purpose:** Initial custom training
- **Status:** Deprecated

---

## train_gpt_custom_v2_init_locked.py
- **Phase:** Training
- **Purpose:** Init experiments
- **Status:** Experimental

---

## train_gpt_custom_v3_bigram.py
- **Phase:** Training
- **Purpose:** Bigram-aware training
- **Status:** Active

---

## train_gpt_mlx.py
- **Phase:** Training
- **Purpose:** Alternate backend
- **Status:** Experimental

---

# Final Note

This index exists to prevent:

- duplicated logic
- forgotten scripts
- pipeline confusion

If a script is not referenced here, it should be:
→ added  
or  
→ archived