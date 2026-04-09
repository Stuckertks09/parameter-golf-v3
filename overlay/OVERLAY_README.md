# Parameter Golf — Corpus-Aware Tokenizer with Failure-Driven Refinement

## Overview

This repository explores whether a custom tokenizer can outperform a standard 1024-token SentencePiece tokenizer under the constraints of the Parameter Golf challenge:

- fixed dataset (FineWeb-derived)
- strict training budget (~10 minutes on 8×H100)
- submission size limit (~16MB)

The work evolved from baseline replication into a full tokenizer system, including:

- corpus analysis and shard characterization  
- n-gram mining and compression estimation  
- curated vocabulary construction  
- dynamic-programming (DP) tokenization  
- dataset export aligned to tokenizer behavior  
- failure-driven refinement using full validation data  

---

## Key Result

A custom tokenizer (**V3**) achieves a **measurable compression improvement** over SentencePiece on the full validation dataset.

However:

- SentencePiece still achieves **lower final validation loss**
- The custom tokenizer does **not converge as efficiently within the time budget**

### Interpretation

The results separate cleanly into two effects:

- **Representation**  
  The custom tokenizer encodes the data more efficiently.

- **Optimization**  
  The model learns more effectively from SentencePiece tokens under constrained training.

This indicates that the remaining gap is **model-side**, not tokenizer-side.

---

## Core Contribution

The primary contribution of this work is not a single tokenizer design, but a **methodology for tokenizer optimization**:

> Tokenizer improvements must be driven by full-validation compression and failure analysis, not proxy metrics or intuition.

This led to a structured refinement loop:

```text
generate → evaluate → diagnose → correct → validate → train
````

This replaces earlier approaches based on:

* sampled evaluation
* heuristic vocab adjustments
* training results without isolating tokenizer quality

---

## System Overview

The tokenizer system operates as a closed-loop pipeline:

```text
        ┌───────────────┐
        │   Generation  │
        │ (ngrams etc.) │
        └──────┬────────┘
               ↓
        ┌───────────────┐
        │  Construction │
        │   (vocab)     │
        └──────┬────────┘
               ↓
        ┌───────────────┐
        │  Evaluation   │
        │ (full-val)    │
        └──────┬────────┘
               ↓
        ┌───────────────┐
        │  Refinement   │
        │ (worst-docs)  │
        └──────┬────────┘
               ↓
        ┌───────────────┐
        │   Training    │
        └───────────────┘
```

The critical shift in this project was moving from:

> generating better tokenizers

to:

> correcting tokenizer failures observed on real validation data

---

## Repository Structure

### Core Documents

* `PROJECT_JOURNEY.md`
  Chronological reconstruction of the system’s evolution and decision-making process

* `TOKENIZER_PIPELINE.md`
  Formal definition of the refinement methodology and evaluation loop

* `TOKENIZER_V3_FINAL.md`
  Specification and results of the final frozen tokenizer

* `SYSTEM_INDEX.md`
  Complete mapping of scripts, artifacts, and pipeline stages

---

## Current Status

* Tokenizer V3 is **frozen as the baseline for further research**
* Dataset pipeline is **stable and reproducible**
* Tokenizer-side gains have reached **diminishing returns**

All further work is focused on:

> improving model learning efficiency on top of the fixed tokenizer

---

## Reproducing the System

### 1. Evaluate tokenizer compression

```bash
python analysis/Eval\ Scripts/compare_full_val_compression.py \
  --docs-jsonl data/docs_selected.jsonl \
  --num-val-docs 50000 \
  --sp-model data/tokenizers/fineweb_1024_bpe.model \
  --vocab-jsonl vocab/vocab_best_v3.jsonl
```

---

### 2. Generate targeted refinement candidates

```bash
python analysis/Eval\ Scripts/propose_vocab_fixes_from_worst_docs.py \
  --worst-docs-csv analysis/full_val_worst_*.csv \
  --docs-jsonl data/docs_selected.jsonl \
  --vocab-jsonl vocab/vocab_best_v3.jsonl
```

---

### 3. Rank vocab variants

```bash
python analysis/Eval\ Scripts/rank_vocab_variants_full_val.py \
  --docs-jsonl data/docs_selected.jsonl \
  --variant-dir analysis
```

---

### 4. Export dataset

```bash
python scripts/export_custom_dp_dataset_v4.py ...
```

---

### 5. Train

```bash
torchrun --standalone --nproc_per_node=8 \
scripts/70_training/train_gpt_custom_v3_locked.py
```

---

## Final Position

This project demonstrates:

* custom tokenization can outperform SentencePiece on compression
* evaluation methodology is critical for meaningful progress
* compression improvements do not guarantee training improvements

### Key takeaway

The remaining challenge is:

> how to train efficiently on a more compressed and structured token representation

```

