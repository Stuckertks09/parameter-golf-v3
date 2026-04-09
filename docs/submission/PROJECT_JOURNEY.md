# `PROJECT_JOURNEY.md`

````md
# Project Journey
## From heuristic tokenization to failure-driven refinement

---

## 1. Overview

This project began as an attempt to improve tokenizer efficiency over a standard SentencePiece baseline.  

It evolved into a structured system for **data-driven tokenizer optimization**, culminating in a custom tokenizer (V3) that improves compression on the evaluation distribution but remains behind in training performance.

The most important outcome of this work is not a specific tokenizer, but a **methodological shift**:

> from intuition-driven experimentation  
> to full-validation, failure-driven refinement

---

## 2. Initial Objective

The original goal was straightforward:

> Can a custom tokenizer outperform SentencePiece under strict constraints?

These constraints included:

- fixed dataset (FineWeb-derived)
- limited training time (~10 minutes on 8×H100)
- fixed vocabulary size (1024 tokens)
- strict model size budget

The problem was initially approached as a **tokenization design problem**.

---

## 3. Phase Progression

The system evolved through a sequence of phases, each addressing limitations of the previous approach.

```text
Baseline → Analysis → Mining → Curation → Vocab → DP → Export → Search → Evaluation → Refinement → V3
````

---

## 4. Phase 0 — Baseline Establishment

A standard SentencePiece tokenizer was trained and evaluated.

Purpose:

* establish correctness (roundtrip encoding/decoding)
* measure baseline compression and training performance
* define reference metrics

**Outcome:**

A stable baseline was established, providing a consistent point of comparison.

**Limitation:**

No understanding of where improvements might come from.

---

## 5. Phase 1 — Dataset Awareness

The dataset was analyzed at the shard level.

Key artifacts:

* shard-level statistics
* shard scoring and ranking

**Insight:**

> The dataset is not uniform.

Different shards exhibit different characteristics:

* structured vs unstructured text
* clean vs noisy (OCR-like)
* varying boundary patterns

**Implication:**

Tokenizer performance may depend on dataset structure, not just token frequency.

---

## 6. Phase 2 — N-gram Mining

Frequent n-grams were extracted and analyzed.

**Findings:**

* bigrams provide the majority of compression opportunity (~6–7%)
* trigrams add smaller incremental gains
* naive frequency-based estimates overstate actual gains

**Key shift:**

> Compression opportunity is real, but must be measured carefully.

---

## 7. Phase 3 — Candidate Curation

Raw n-gram candidates were filtered and refined.

This introduced:

* removal of low-signal merges
* separation of phrase vs subword candidates
* ranking and scoring mechanisms

**Insight:**

> Frequency alone is not a sufficient selection criterion.

**Limitation:**

Still unclear how curated vocab changes impact real evaluation performance.

---

## 8. Phase 4 — Vocabulary Engineering

Custom vocabularies were constructed explicitly.

Approach:

* balance phrases, subwords, and fallback tokens
* experiment with allocation strategies
* generate multiple variants

**Outcome:**

First viable alternatives to SentencePiece were produced.

**Limitation:**

Evaluation remained indirect and often misleading.

---

## 9. Phase 5 — Dynamic Programming Tokenizer

A custom tokenizer was implemented using dynamic programming.

Unlike greedy approaches, the DP tokenizer optimized segmentation globally using:

* token count
* fallback count
* fallback runs
* boundary penalties
* match length

**Insight:**

> Tokenization quality depends on global segmentation, not just local merges.

**Outcome:**

Performance approached SentencePiece, but did not surpass it.

---

## 10. Phase 6 — Dataset Export

The tokenizer was integrated into a full dataset export pipeline.

This enabled:

* training on custom tokenization
* direct comparison with baseline training runs

**Transition:**

> From analysis → real training impact

**Limitation:**

Still unclear where performance differences originated.

---

## 11. Phase 7 — Variant Search

Vocabulary exploration became more systematic.

Approach:

* generate many variants
* evaluate and rank
* compare results across runs

**Outcome:**

Broader exploration of design space.

**Limitation:**

Results became noisy and inconsistent.

Progress slowed.

---

## 12. Phase 8 — Evaluation Shift (Critical Turning Point)

This phase introduced the most important change in the project.

### Problem with earlier approaches

Evaluation relied on:

* sampled text
* proxy metrics
* training outcomes without isolating tokenizer quality

This led to:

* misleading signals
* wasted experiments
* unclear failure modes

---

### The shift

A new evaluation pipeline was introduced:

1. measure compression on full validation dataset (50k docs)
2. identify worst-performing documents
3. analyze failure patterns
4. generate targeted vocabulary fixes
5. rank variants on full validation compression
6. train only validated improvements

---

### Key insight

> Tokenizer performance must be evaluated directly on the real distribution.

This eliminated:

* reliance on intuition
* dependence on partial metrics

---

## 13. Phase 9 — Failure-Driven Refinement

Using full validation outputs, the focus shifted to **specific failure regions**.

Observed failure patterns:

* structured documents (headers, transcripts)
* OCR/noisy text
* newline-heavy formatting
* punctuation-dense regions

Instead of global changes, the approach became:

> apply small, targeted fixes to known failure cases

This stabilized progress.

---

## 14. Phase 10 — V3 Stabilization

Through iterative refinement, Tokenizer V3 emerged.

### Result

* achieves a real compression improvement over SentencePiece
* reduces total token count on validation data
* improves bits-per-byte (bpb)

---

### Training behavior

However:

* SentencePiece learns faster
* SentencePiece maintains a lead in final validation loss

---

### Interpretation

The problem separates into two layers:

1. **Representation**

   * V3 encodes data more efficiently

2. **Optimization**

   * the model learns more effectively from SentencePiece tokens

---

## 15. Final Position

Tokenizer V3 is treated as a **frozen baseline** for further research.

This decision is based on:

* confirmed compression improvement
* stable and reproducible pipeline
* diminishing returns from further vocab tuning
* instability from small changes

---

### Important clarification

The tokenizer is not assumed to be globally optimal.

Further improvements may still be possible, including:

* small slot-level refinements
* additional structural tokens

However:

* recent changes have produced inconsistent results
* improvements do not reliably translate to better training performance

---

## 16. What Remains

The remaining gap is **model-side**.

The key question is no longer:

> how to compress the data

but:

> how to learn efficiently from a compressed representation under strict constraints

---

## 17. Conclusion

This project evolved from heuristic experimentation into a structured optimization system.

The most important shift was not a specific technique, but a change in approach:

> from generating better tokenizers
> to correcting real failures observed on the evaluation distribution

Tokenizer V3 represents the outcome of this process:

* improved compression over SentencePiece
* a reproducible and stable pipeline
* a clear separation between representation and optimization

From this point forward, the tokenizer is held fixed, and the focus shifts to model-side improvements.

```
`
