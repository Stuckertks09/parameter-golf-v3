# Parameter Golf — Custom Tokenizer System (V3)

## Overview

This repository implements a full pipeline for tokenizer optimization under the constraints of the OpenAI Parameter Golf challenge.

The system evolves beyond standard frequency-based tokenization into a **full-validation, failure-driven refinement pipeline**.

The final result is **Tokenizer V3**, which:

* achieves **better compression than SentencePiece**
* is **fully reproducible**
* separates:

  * **representation efficiency (tokenizer)**
  * **optimization efficiency (model training)**

---

## Key Result

```
delta_tokens = -32,149
delta_tpb   = -0.00021279
```

Tokenizer V3 out-compresses SentencePiece on full validation.

However:

```
Final val_bpb (V3) ≈ 1.2419
Final val_bpb (SP) ≈ 1.2329
```

---

## Core Insight

The problem decomposes into:

1. Representation → improved by custom tokenizer
2. Optimization → still better with SentencePiece

---

## Pipeline

```
Dataset → Mining → Curation → Vocab → DP Tokenizer
→ Full-Val Eval → Failure Analysis → Refinement
→ Ranking → Export → Training
```

---

## Minimal Reproduction

### 1. Clone baseline

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

### 2. Apply overlay

```bash
git clone https://github.com/Stuckertks09/parameter-golf-v1.git
rsync -av --exclude='.git' parameter-golf-v1/ parameter-golf/
```

### 3. Export dataset

```bash
python scripts/60_dataset_export/export_custom_dp_dataset_v4.py \
  --docs data/docs_selected.jsonl \
  --vocab vocab/vocab_best_v3.jsonl \
  --output data/datasets/fineweb10B_customdp1024_v3 \
  --max-train-shards 80
```

### 4. Validate compression

```bash
python Eval Scripts/compare_full_val_compression.py \
  --vocab vocab/vocab_best_v3.jsonl
```

### 5. Train baseline

```bash
DATA_PATH=data/datasets/fineweb10B_sp1024 \
TOKENIZER_KIND=sentencepiece \
TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 scripts/70_training/train_gpt.py
```

### 6. Train custom

```bash
DATA_PATH=data/datasets/fineweb10B_customdp1024_v3 \
TOKENIZER_KIND=custom_jsonl \
TOKENIZER_PATH=vocab/vocab_best_v3.jsonl \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
scripts/70_training/train_gpt_custom_v3_locked.py
```

---

## Failure Patterns

* structured documents
* OCR/noisy text
* newline-heavy formatting
* punctuation-dense regions

---

## Status

* Tokenizer: frozen (V3)
* Dataset: fixed
* Focus: model-side optimization
