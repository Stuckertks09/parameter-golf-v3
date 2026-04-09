
# `TOKENIZER_PIPELINE.md`

````md
# Tokenizer Refinement Methodology
## A failure-driven, full-validation pipeline for tokenizer optimization

---

## 1. Overview

This document defines the methodology used to develop and refine the custom tokenizer system.

The system evolved from early **generative approaches** (building better tokenizers from corpus statistics) into a **corrective, failure-driven pipeline** grounded in full validation data.

The key shift was methodological:

> Tokenizers should not be optimized based on proxy metrics or intuition, but through direct measurement and correction of real failures on the evaluation distribution.

---

## 2. Problem Framing

Traditional tokenizer development relies on:

- frequency-based merges (e.g., BPE / SentencePiece)
- local heuristics (e.g., n-gram frequency)
- indirect evaluation (sampled text, training loss)

These approaches have two limitations:

1. **They do not directly optimize for evaluation metrics**  
   Improvements in frequency or local compression do not necessarily translate to better performance on the validation set.

2. **They obscure failure modes**  
   It is difficult to determine *where* and *why* a tokenizer is underperforming.

---

## 3. Methodological Shift

This work replaces indirect optimization with a **closed-loop refinement system**:

```text
generate → evaluate → diagnose → correct → validate → (train)
````

The pipeline enforces a strict ordering:

> Compression must improve on the full validation set before training is considered.

---

## 4. System Architecture

```text
        ┌────────────────────┐
        │  Candidate Generation │
        │ (ngrams, subwords) │
        └─────────┬──────────┘
                  ↓
        ┌────────────────────┐
        │  Vocab Construction │
        └─────────┬──────────┘
                  ↓
        ┌────────────────────┐
        │ Full-Val Evaluation │
        └─────────┬──────────┘
                  ↓
        ┌────────────────────┐
        │ Failure Diagnosis   │
        └─────────┬──────────┘
                  ↓
        ┌────────────────────┐
        │ Targeted Refinement │
        └─────────┬──────────┘
                  ↓
        ┌────────────────────┐
        │ Variant Ranking     │
        └─────────┬──────────┘
                  ↓
        ┌────────────────────┐
        │ Training (optional) │
        └────────────────────┘
```

---

## 5. Pipeline Stages

---

### 5.1 Candidate Generation

**Purpose:**
Identify compressible structures in the dataset.

**Methods:**

* n-gram mining (bigrams, trigrams)
* subword extraction
* frequency analysis

**Artifacts:**

* `ngram_top.csv`
* `decoded_bigrams.jsonl`
* `decoded_trigrams.jsonl`

**Key insight:**

> Compression opportunity is measurable, but naive estimates are misleading without overlap-aware evaluation.

---

### 5.2 Vocabulary Construction

**Purpose:**
Build candidate token sets from curated inputs.

**Methods:**

* filtering noisy or redundant candidates
* balancing token classes:

  * phrases
  * subwords
  * byte fallback
* constructing fixed-size vocabularies (1024 tokens)

**Artifacts:**

* `custom_vocab*.jsonl`
* `vocab_variants/*.jsonl`

**Key insight:**

> Vocabulary design is a constrained allocation problem, not a frequency ranking problem.

---

### 5.3 Full Validation Evaluation (Critical Step)

**Script:**

* `compare_full_val_compression.py`

**Purpose:**
Measure tokenizer performance on the **actual evaluation distribution** (50k validation documents).

**Metrics:**

* total token count
* delta_tokens vs SentencePiece
* delta_tpb (tokens per byte)
* fallback counts
* fallback run counts

**Outputs:**

* `full_val_summary_*.json`
* `full_val_worst_*.csv`
* `full_val_delta_histogram_*.csv`

**Key insight:**

> Proxy evaluations can be misleading; only full-validation compression reflects true performance.

---

### 5.4 Failure Diagnosis

**Purpose:**
Identify where the tokenizer underperforms.

**Method:**

* analyze worst-performing documents from full validation output

**Observed failure regions:**

* structured documents (headers, transcripts)
* OCR-like or noisy text
* newline-heavy formatting
* punctuation-dense regions

**Artifacts:**

* `full_val_worst_*.csv`
* `tokenizer_gap_report.txt`

**Key insight:**

> Tokenizer failures are not uniform; they are concentrated in specific structural regions.

---

### 5.5 Targeted Refinement

**Script:**

* `propose_vocab_fixes_from_worst_docs.py`

**Purpose:**
Generate **small, structured vocabulary updates** based on real failures.

**Behavior:**

* proposes token additions aligned to document structure
* favors:

  * line-aligned spans
  * structured phrases
  * punctuation-aware sequences
* rejects:

  * arbitrary substrings
  * low-signal fragments

**Outputs:**

* candidate additions/removals
* swap proposals
* updated vocab variants

**Key insight:**

> Improvements come from fixing specific failure cases, not globally redesigning the tokenizer.

---

### 5.6 Variant Ranking

**Script:**

* `rank_vocab_variants_full_val.py`

**Purpose:**
Evaluate multiple vocab variants directly on full validation data.

**Metrics:**

* delta_tpb
* delta_tokens

**Outputs:**

* ranked variant lists

**Constraint:**

> Only variants that improve compression are considered for training.

---

### 5.7 Training (Deferred)

Training is only performed after a validated compression improvement.

**Rule:**

```text
No compression gain → no training
```

**Purpose:**

* isolate tokenizer improvements from training noise
* reduce unnecessary experiments

---

## 6. System-Level Insight

The central insight of this work is the transition from:

### Generative approach

* build better tokenizers from corpus statistics

to:

### Corrective approach

* fix tokenizer failures observed on real evaluation data

This transforms the pipeline into a **feedback loop**:

```text
tokenizer → evaluation → failure → correction → improved tokenizer
```

---

## 7. Strategic Constraints

To maintain stability and progress:

**Do not:**

* rely on sampled evaluation
* rebuild vocabularies from scratch repeatedly
* train unvalidated variants

**Do:**

* anchor all decisions to full validation compression
* apply small, targeted changes
* validate improvements before training

---

## 8. Outcome

This methodology enabled:

* identification of true failure modes
* stable incremental improvements
* elimination of misleading signals
* convergence to a tokenizer (V3) that:

  * improves compression over SentencePiece
  * is reproducible and stable

---

## 9. Limitations

* compression improvements do not directly translate to improved training performance
* the model may struggle to learn efficiently from more compressed token distributions
* tokenizer optimization alone is insufficient under strict training constraints

---

## 10. Conclusion

Tokenizer optimization is not purely a design problem.

It is a **measurement and feedback problem**.

This pipeline demonstrates that:

* real progress requires evaluation on the actual distribution
* failure-driven refinement is more effective than global search
* tokenizer quality must be considered separately from model learning dynamics

The tokenizer workstream concludes with V3 as a stable baseline, and the focus shifts to model-side optimization.

```

