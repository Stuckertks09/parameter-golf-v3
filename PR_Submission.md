## Tokenizer V3: Full-validation compression win with controlled A/B showing limits of tokenizer improvements under fixed compute

---

## Summary

This PR introduces a custom tokenizer pipeline (V3) and a controlled experimental study evaluating its impact under the Parameter Golf constraints.

The main result is not just a tokenizer improvement, but a **clear boundary**:

> Improving tokenization quality (compression) does not translate into improved training performance under a fixed compute budget.

---

## What this PR adds

### 1. Custom tokenizer (V3)

* 1024-token vocabulary
* phrase + boundary-aware subwords
* full byte fallback coverage
* dynamic programming segmentation (non-greedy)

---

### 2. Full-validation evaluation

Tokenizer quality is measured on the **full validation set (50,000 docs)**, not samples.

Result:

* **32,149 fewer tokens vs SentencePiece**
* **-0.00021279 tokens-per-byte**

This confirms the tokenizer is a **real representation improvement**.

---

### 3. Controlled A/B training

47 filtered experiments were run across 1×H100 and 8×H100:

* same architecture
* same hyperparameters
* tokenizer is the only variable

Key results:

* Custom tokenizer can beat naive baseline
  → **1.2206 vs ~1.2244 (1024)**

* But does not beat tuned SentencePiece
  → ~**1.242 vs ~1.233**

---

### 4. 4096 sequence-length experiments

Sequence length was evaluated as a separate axis:

Top runs:

* **1.2069 — SentencePiece, 4096**
* **1.2206 — Custom, 4096**
* **1.2228 — Custom, 4096**

Findings:

* 4096 > 1024 (clear improvement)
* Custom 4096 > baseline 1024
* Best SentencePiece 4096 still leads

---

## What didn’t work

All tested under controlled A/B conditions.

None produced asymmetric benefit for the custom tokenizer:

* DP segmentation without vocab redesign
* compositional embedding initialization
* token-class gradient scaling (~0.0002 bpb)
* MLP expansion
* additional layers
* H-net input modulation
* longer sequences (helped both, favored SP)

---

## Key insight

The core result of this work is:

> **Tokenization quality and learning efficiency are separable.**

You can:

* reduce token count
* improve segmentation
* improve structure

…but still fail to improve final model performance.

---

## Interpretation

This PR shows:

* tokenizer improvements are real and measurable
* they can improve baseline performance
* but they **do not dominate** in this regime

Instead:

> **Training efficiency and context length are the primary levers once compression is “good enough.”**

---

## Why this is useful

Most submissions optimize for score.

This work identifies:

* where tokenizer improvements stop helping
* which changes do *not* transfer
* and where future effort should be focused

---

## Takeaway

* Tokenizer V3 improves representation
* It reaches competitive (baseline-beating) performance
* But does not outperform optimized SentencePiece

The bottleneck has shifted from tokenization to learning efficiency.

---

## Repo / Reproduction

See README for:

* dataset export
* full-validation comparison
* training commands

---

## One-line summary

> Better tokenization improves compression, but under fixed compute, learning efficiency and sequence length dominate final performance.

---