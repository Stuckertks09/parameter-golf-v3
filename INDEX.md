# 📘 `SYSTEM_INDEX.md` 

````md
# System Index
## Tokenizer system: generation → evaluation → refinement → training

---

## 1. Overview

This document defines all components of the tokenizer system and how they interact.

The system is organized into four stages:

1. Generation — produce candidate token structures  
2. Construction — build vocabularies  
3. Evaluation — measure performance on full validation data  
4. Refinement — correct failure regions  

Final step:
→ Training (only after validated improvements)

---

## 2. System Diagram

```text
Raw Data
   ↓
N-gram Mining
   ↓
Candidate Curation
   ↓
Vocabulary Construction
   ↓
DP Tokenizer
   ↓
Full Validation Evaluation
   ↓
Failure Diagnosis
   ↓
Targeted Refinement
   ↓
Variant Ranking
   ↓
(only if improved)
   ↓
Dataset Export → Training
````

---

## 3. Active Pipeline (V3)

These components define the **current working system**.

### Evaluation & Refinement (Core)

* `compare_full_val_compression.py`
  → evaluates tokenizer on full validation set

* `propose_vocab_fixes_from_worst_docs.py`
  → generates targeted vocab corrections

* `rank_vocab_variants_full_val.py`
  → ranks variants based on compression

---

### Tokenizer & Dataset

* `dp_tokenizer_lib.py`
  → dynamic programming tokenizer implementation

* `export_custom_dp_dataset_v4.py`
  → dataset export pipeline

---

### Training

* `train_gpt_custom_v3_locked.py`
  → training with custom tokenizer

* `train_gpt.py`
  → SentencePiece baseline

---

### Core Artifacts

* `vocab/vocab_best_v3.jsonl`
  → final tokenizer vocabulary

* `data/datasets/fineweb10B_customdp1024_v3/`
  → exported dataset

---

## 4. Generation Pipeline (Historical but Important)

These scripts generate candidate token structures.

### N-gram Mining

* `mine_fineweb_ngrams.py`
* `decode_bigrams.py`
* `decode_trigrams.py`

Artifacts:

* `ngram_top.csv`
* `decoded_bigrams.jsonl`
* `decoded_trigrams.jsonl`

Purpose:
→ identify compressible patterns in data

---

### Merge Savings Estimation

* `estimate_merge_savings.py`
* `estimate_ngram_merge_savings.py`
* `simulate_merge_tokenization.py`

Artifacts:

* `merge_savings_*.json`
* `retokenization_simulation_report.json`

Purpose:
→ estimate realistic compression gains

---

## 5. Candidate Curation

* `filter_bigram_candidates_v2.py`
* `filter_merge_candidates_v2.py`
* `split_merge_candidates.py`
* `merge_candidate_files.py`
* `auto_curate_subwords_v2.py`

Artifacts:

* filtered candidate sets
* curated subword rankings

Purpose:
→ remove noise and enforce quality constraints

---

## 6. Vocabulary Construction

* `build_hybrid_vocab_v3.py`
* `generate_vocab_variants.py`
* `build_hybrid_grid.py`
* `build_hybrid_local_search.py`

Artifacts:

* `custom_vocab_full_*.jsonl`
* `vocab_variants/*.jsonl`

Purpose:
→ construct and explore tokenizer design space

---

## 7. Tokenizer Evaluation (Pre-Refinement)

* `compare_baseline_vs_custom.py`
* `compare_vocab_to_baseline.py`
* `analyze_tokenizer_gaps.py`
* `inspect_variant_docs.py`

Artifacts:

* `tokenizer_gap_report.txt`
* comparison logs

Purpose:
→ identify general weaknesses prior to full-val pipeline

---

## 8. Evaluation & Refinement (Critical System)

This is the **most important part of the system**.

### Full Validation Evaluation

* `compare_full_val_compression.py`

Outputs:

* `full_val_summary_*.json`
* `full_val_worst_*.csv`
* `full_val_delta_histogram_*.csv`

Purpose:
→ measure real tokenizer performance

---

### Failure Diagnosis

Artifacts:

* worst-doc reports
* gap analysis

Purpose:
→ identify structured failure regions

---

### Targeted Refinement

* `propose_vocab_fixes_from_worst_docs.py`

Outputs:

* candidate additions/removals
* vocab swap proposals
* new vocab variants

Purpose:
→ correct failures directly

---

### Variant Ranking

* `rank_vocab_variants_full_val.py`

Outputs:

* ranked vocab variants

Purpose:
→ select only compression-improving tokenizers

---

## 9. Dataset Export

* `export_custom_dp_dataset_v4.py`

Outputs:

* `fineweb_train_*.bin`
* `fineweb_val_*.bin`

Purpose:
→ produce training-ready dataset using custom tokenizer

---

## 10. Training

### Baseline

* `train_gpt.py`

### Custom

* `train_gpt_custom_v3_locked.py`
* `train_gpt_custom_v3_bigram.py` (experimental)
* `train_gpt_custom_v2_init_locked.py` (experimental)

Outputs:

* training logs
* validation metrics

Purpose:
→ evaluate downstream impact of tokenizer

---

## 11. Minimal Reproduction (V3)

To reproduce the final system:

1. `vocab/vocab_best_v3.jsonl`
2. `compare_full_val_compression.py`
3. `propose_vocab_fixes_from_worst_docs.py`
4. `rank_vocab_variants_full_val.py`
5. `export_custom_dp_dataset_v4.py`
6. `train_gpt_custom_v3_locked.py`
7. dataset: `fineweb10B_customdp1024_v3`

---

## 12. Deprecated / Low-Priority Components

These exist for historical context but are not part of the active pipeline:

* early SentencePiece shaping scripts
* initial vocab builders (`build_vocab_draft`, etc.)
* multiprocessing export variants (v1–v3)
* exploratory training variants without stable results

---

## 13. Key Insight

This system is not a linear pipeline.

It is a **feedback loop**:

```text
tokenizer → evaluation → failure → correction → improved tokenizer
```

The transition from:

* broad generation
  to
* targeted correction

is what enabled stable progress and led to Tokenizer V3.

---

## 14. Practical Guidance

To work within this system:

**Do:**

* start from `vocab_best_v3.jsonl`
* evaluate on full validation
* apply small, targeted fixes
* validate before training

**Do not:**

* rebuild vocab from scratch repeatedly
* rely on sampled evaluation
* train unvalidated variants

---

## 15. Conclusion

The repository contains many scripts and artifacts, but only a subset forms the active system.

Understanding the distinction between:

* generation (historical exploration)
* refinement (active methodology)

is essential for using the system effectively.

The tokenizer is now fixed. The system is ready for model-side exploration.

```

