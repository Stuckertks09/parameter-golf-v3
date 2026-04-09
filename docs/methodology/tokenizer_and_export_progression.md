# Tokenizer Evaluation and Dataset Export

## Overview

This stage evaluates constructed vocabularies under realistic tokenization conditions and converts them into training-ready datasets. It bridges the gap between static vocabulary design and downstream model training.

The implementation is organized across two script directories:

* `scripts/50_tokenizer_eval/`
* `scripts/60_dataset_export/`

This stage introduces measurable performance metrics and produces the final tokenized datasets used during training.

---

## Tokenizer Evaluation (`50_tokenizer_eval`)

### Purpose

Evaluate vocabulary variants by applying them to real text and comparing tokenization efficiency against the baseline SentencePiece tokenizer.

---

### Core Scripts

* `evaluate_tokenizers.py`
* `compare_tokenizers.py`
* `analyze_tokenization_diff.py`

---

### Evaluation Process

1. Load vocabulary variant (custom or hybrid)
2. Tokenize evaluation dataset (typically 50k documents)
3. Tokenize same dataset with baseline SentencePiece
4. Compute comparative metrics

---

### Metrics

#### Token Count Metrics

* `sp_tokens`: total tokens using baseline tokenizer
* `custom_tokens`: total tokens using candidate tokenizer
* `token_delta_sp_minus_custom`: difference in token counts

Interpretation:

* Negative delta → custom tokenizer uses fewer tokens (compression gain)
* Positive delta → custom tokenizer is worse than baseline

---

#### Ratio Metric

* `ratio_custom_over_sp`

Interpretation:

* < 1.0 → custom tokenizer is more efficient
* > 1.0 → baseline tokenizer is more efficient

---

#### Fallback Metrics

* `fallback_tokens`
* `fallback_share`

Fallback tokens represent cases where the tokenizer cannot match higher-level tokens and must revert to byte-level or minimal units.

Interpretation:

* Lower fallback share indicates better vocabulary coverage
* High fallback share indicates poor token allocation

---

#### Document-Level Metrics

* `docs_custom_beats_sp`
* `docs_sp_beats_custom`

These metrics count how many documents each tokenizer performs better on.

Additional outputs include:

* largest per-document wins for each tokenizer

---

### Outputs

Evaluation results are written as CSV files, including:

* `variant_eval_sp_only_50k.csv`
* `variant_eval_sp_shaped_ranked_50k.csv`
* `variant_eval_sp_shaped_safe_50k.csv`

Each row represents a vocabulary variant with full metric breakdown.

---

### Interpretation of Results

Evaluation focuses on three dimensions:

1. **Compression efficiency** (token count and ratio)
2. **Coverage quality** (fallback share)
3. **Distribution robustness** (document-level wins)

These metrics determine whether a vocabulary is suitable for dataset export and training.

---

## Dataset Export (`60_dataset_export`)

### Purpose

Convert text datasets into tokenized binary shards using a selected tokenizer.

This produces training data compatible with the training pipeline.

---

### Core Scripts

* `export_custom_dp_dataset.py`
* `export_custom_dp_dataset_v2.py`
* `export_custom_dp_dataset_v3.py`
* multiprocessing variants

---

### Export Process

1. Load source dataset (`docs_selected.jsonl`)
2. Apply tokenizer (custom or baseline)
3. Convert text into token ID sequences
4. Chunk tokens into fixed-size shards
5. Write shards to disk

---

### Shard Format

Each shard follows a fixed structure:

* 1024-token header
* sequence of token IDs
* total shard size ≈ 100,000,000 tokens

Shards are stored as binary files.

---

### Configuration Parameters

Typical export parameters include:

* `--max-train-shards`
* `--workers`
* `--batch-docs`

These control:

* total dataset size
* parallelism
* batching behavior

---

### Dataset Variants

Examples include:

* `fineweb10B_sp1024`
* `fineweb10B_customdp1024_v*`

Each dataset variant corresponds to a tokenizer configuration.

---

### Validation

Export correctness is verified by:

* matching shard counts (e.g., 80 shards)
* matching token counts
* ensuring consistent document coverage

---

## Relationship Between Evaluation and Export

Evaluation determines which vocabulary variants are worth exporting.

Export converts those validated vocabularies into datasets used for training.

This ensures that:

* only high-performing vocabularies are used in training
* dataset structure matches tokenizer assumptions

---

## Summary

This stage introduces measurable evaluation and produces training-ready datasets.

It performs two primary functions:

1. evaluates tokenizer efficiency using token count, fallback, and document-level metrics
2. exports tokenized datasets into fixed-format shards for training

The result is a set of datasets aligned with specific tokenizer variants, enabling direct comparison during model training.
