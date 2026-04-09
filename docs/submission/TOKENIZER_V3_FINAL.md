# Tokenizer V3

## Frozen tokenizer baseline for model-side research

This document defines **Tokenizer V3**, the final tokenizer configuration used as the baseline for subsequent model-side experiments.

The tokenizer, dataset, and supporting training path are now treated as **frozen for the remainder of this work**. While further tokenizer improvements may still be possible, recent iterations have shown diminishing or negative returns. As a result, the focus of the project shifts to **model-side optimization on top of a fixed tokenizer**.

---

## 1. Overview

Tokenizer V3 is a custom tokenization system developed to compete with a standard 1024-token SentencePiece baseline under the constraints of the Parameter Golf challenge:

* fixed dataset source (FineWeb-derived)
* strict wallclock training limit (~10 minutes on 8×H100)
* submission size constraint (~16MB)

The system consists of:

1. A fixed JSONL vocabulary (1024 tokens)
2. A dynamic-programming (DP) tokenizer
3. A reproducible dataset export pipeline
4. A training setup that supports tokenizer-aware evaluation

The goal was to improve **representation efficiency (compression)** without introducing instability or excessive complexity in training.

---

## 2. Vocabulary

Tokenizer V3 uses a fixed vocabulary stored as:

```text
vocab/vocab_best_v3.jsonl
```

This vocabulary is now **frozen**.

### Vocabulary structure

The token set includes:

* byte fallback tokens (`<0xXX>`) for full coverage
* subword tokens
* phrase tokens (multi-token merges such as common n-grams)
* space-prefixed tokens
* structurally meaningful tokens (e.g., formatting patterns, document artifacts)

These token types are also used during training for selective gradient handling.

### Important note

The vocabulary is not assumed to be globally optimal. There are likely still a small number of slots that could be improved. However:

* recent attempts to refine individual tokens have tended to degrade results
* improvements at the tokenizer level no longer translate reliably to better training outcomes

Because of this, the vocabulary is treated as **stable enough to freeze** for the purposes of this work.

---

## 3. Tokenization method (DP encoder)

Tokenizer V3 uses a **dynamic-programming segmentation algorithm** rather than greedy matching.

### Objective function

For each input string, segmentation is chosen to optimize:

1. minimum token count
2. minimum fallback token count
3. minimum fallback runs
4. minimum boundary fallback penalty
5. preference for longer matches

This differs from standard BPE/SentencePiece approaches, which rely on greedy longest-match behavior.

### Boundary awareness

The DP objective applies additional penalties to fallback tokens occurring at structural boundaries, including:

* spaces
* newlines
* tabs

This encourages:

* cleaner token boundaries
* fewer fragmented byte sequences
* better structural alignment in tokenization

---

## 4. Dataset

Tokenizer V3 is paired with a fully exported dataset:

```text
data/datasets/fineweb10B_customdp1024_v3
```

### Dataset properties

* 80 training shards
* validation set derived from first 50,000 documents
* shard size: 100,000,000 tokens
* binary format compatible with baseline training pipeline

The dataset differs from the baseline only in **tokenization**, not in source content.

### Reproducibility

The export process records metadata including:

* tokenizer type (`custom_dp_jsonl`)
* scoring mode (`boundary_aware_v1`)
* shard configuration and sizes

This ensures the tokenizer and dataset are tightly coupled and reproducible.

---

## 5. Training support

A modified training script is used to support the custom tokenizer.

### Additions over baseline

The training setup includes:

* support for `custom_jsonl` tokenizers
* tokenizer-aware byte accounting for `val_bpb`
* token classification into:

  * byte
  * phrase
  * space
  * structural
* token-class gradient scaling for embeddings
* optional compositional initialization (disabled in final runs)

### Final tokenizer-side configuration

The reference V3 run used:

* tokenizer: custom JSONL vocab
* dataset: DP-exported dataset
* training shards: 80
* gradient scaling:

  * byte = 0.70
  * phrase = 0.90
  * space = 0.98
  * structural = 0.95
* compositional initialization: off

This configuration is treated as the **locked tokenizer-side baseline**.

---

## 6. Compression result

Tokenizer V3 achieved a measurable compression improvement over the SentencePiece baseline:

```text
delta_tpb = -0.000212794
delta_tokens = -32,149
```

### Interpretation

* the tokenizer is **more efficient in representation**
* compression improvements are **real and validated on full evaluation data**

This result is the primary justification for freezing the tokenizer.

---

## 7. Training results (8×H100)

Two full training runs were used for comparison:

* custom tokenizer (V3)
* SentencePiece baseline

---

### 7.1 Custom tokenizer (V3)

Key checkpoints:

```text
step 0:     val_bpb = 4.1039
step 1000:  val_bpb = 1.3929
final:      val_bpb = 1.24194503
```

---

### 7.2 SentencePiece baseline

Key checkpoints:

```text
step 0:     val_bpb = 4.1077
step 1000:  val_bpb = 1.3842
final:      val_bpb = 1.23295442
```

---

### 7.3 Comparison

| Metric    |  Custom V3 |   Baseline |
| --------- | ---------: | ---------: |
| Step 0    |     4.1039 |     4.1077 |
| Step 1000 |     1.3929 |     1.3842 |
| Final     | 1.24194503 | 1.23295442 |

---

## 8. Interpretation

The results separate cleanly into two effects:

### Representation (tokenizer)

* V3 provides **better compression than SentencePiece**
* the tokenizer is not underperforming in terms of encoding efficiency

### Optimization (training)

* the model learns **faster and more effectively on SentencePiece tokens**
* the custom tokenizer does not translate its compression advantage into training efficiency within the time constraint

---

## 9. Why the tokenizer is frozen

Tokenizer V3 is frozen **not because it is perfect**, but because:

* it has already achieved a real compression improvement
* further vocab tuning has become unstable and inconsistent
* small changes often degrade downstream performance
* gains at the tokenizer level no longer map cleanly to final loss

This indicates the project has entered a **diminishing returns regime on tokenizer work**.

---

## 10. What remains open

The remaining gap is primarily **model-side**.

Areas of focus going forward:

* improving how embeddings adapt to phrase and structural tokens
* better learning dynamics under compressed token distributions
* lightweight architectural or optimization changes that improve early learning speed
* methods that preserve step time while improving convergence

---

## 11. Conclusion

Tokenizer V3 is the **frozen tokenizer baseline** for the remainder of this work.

It demonstrates that:

* a custom vocabulary + DP tokenizer can outperform SentencePiece on compression
* the tokenizer and dataset pipeline are stable and reproducible
* compression improvements alone are not sufficient to win under constrained training

### Final position

* tokenizer-side work: **paused (not exhausted, but no longer highest leverage)**
* dataset: **fixed**
* training baseline: **locked for comparison**
* remaining problem: **model-side learning efficiency**

All further work proceeds from this point on top of Tokenizer V3.
