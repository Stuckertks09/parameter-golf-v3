# Failure-Driven Tokenizer Optimization Under Full-Validation Constraints

## Summary

This work explores a different axis of optimization in the Parameter Golf challenge:

> What happens if the tokenizer is not treated as fixed?

The baseline assumes a 1024-token SentencePiece vocabulary. Most approaches optimize the model under that constraint. This work instead treats **tokenization itself—vocabulary and segmentation—as the optimization surface**.

The result is a custom tokenizer (V3) that:

- achieves **better compression than SentencePiece on the full validation set**
- introduces a **failure-driven refinement pipeline grounded in real evaluation data**
- reveals a clear gap between **representation efficiency and learning efficiency**

---

## Motivation

Tokenizer design is typically driven by:

- frequency-based merges (BPE / SentencePiece)
- local heuristics
- indirect evaluation (samples, training loss)

These approaches assume:

> better local compression → better downstream performance

In practice, this doesn’t hold.

Early experiments showed:

- improvements on sampled text did not carry over to full validation  
- training results were noisy and hard to interpret  
- failure modes were invisible  

The core issue was not lack of ideas—it was lack of **reliable measurement**.

This led to a shift:

> tokenizer optimization should be treated as a measured system, not a heuristic process

---

## Approach

The system evolved into a closed-loop pipeline:

```

generate → evaluate → diagnose → correct → validate → train

```

With one strict rule:

> **No tokenizer is trained unless it improves compression on the full validation set.**

This removes training noise from tokenizer evaluation and forces all progress to be grounded in real distribution performance.

---

## Full-Validation Evaluation

All tokenizer variants are evaluated on:

- 50,000 validation documents  
- total token count  
- tokens-per-byte (tpb)  
- fallback usage  

Outputs include:

- exact compression deltas vs SentencePiece  
- ranked worst-performing documents  

This replaces proxy metrics with direct measurement.

---

## Failure-Driven Refinement

Rather than rebuilding vocabularies globally, improvements are targeted:

1. identify worst-performing documents  
2. inspect their structure  
3. propose small token swaps aligned with those failures  
4. re-evaluate on full validation  

This revealed a consistent pattern:

Tokenizer failures are not uniform—they concentrate in:

- structured text (headers, transcripts, dialogue)
- OCR / noisy content
- newline-heavy formatting
- punctuation-dense regions

Improvements came from **fixing specific failure regions**, not global redesign.

---

## Tokenizer Design (V3)

The final tokenizer consists of:

- a fixed 1024-token vocabulary  
- dynamic programming (DP) segmentation (not greedy)  
- boundary-aware fallback penalties  
- balanced allocation of:
  - phrase tokens  
  - word/subword tokens  
  - byte fallback tokens  

Segmentation is chosen globally using:

- token count  
- fallback count and runs  
- boundary penalties  
- match length  

---

## Results

### Compression (Full Validation)

```

delta_tokens = -32,149
delta_tpb   = -0.00021279

```

The custom tokenizer **out-compresses SentencePiece** on the actual evaluation distribution.

---

### Training (8×H100, ~10 minutes)

| Step | Custom V3 | SentencePiece |
|------|----------|---------------|
| 0    | 4.1039   | 4.1077        |
| 1000 | 1.3929   | 1.3842        |
| Final| 1.2419   | 1.2329        |

Despite better compression, SentencePiece achieves better final loss.

---

## Key Finding

The problem separates cleanly into two components:

### 1. Representation

- Custom tokenizer improves compression  
- More efficient encoding of text  

### 2. Optimization

- SentencePiece enables faster learning  
- Better convergence under strict time constraints  

---

## Interpretation

Improving compression changes the token distribution:

- reduces redundancy  
- introduces longer, more structured tokens  
- alters frequency balance  

The gap appears early:

> Step 1000 already shows divergence (V3: 1.3929 vs SP: 1.3842)

This indicates the issue is **learning speed**, not final capacity.

The compressed representation likely:

- weakens early gradient signal  
- reduces redundancy the model relies on  
- slows convergence under fixed training time  

---

## Contribution

This work introduces:

### 1. Full-validation tokenizer evaluation

Tokenizer quality is measured directly on the real evaluation distribution.

---

### 2. Failure-driven refinement

Improvements are derived from:

- worst-case documents  
- observable failure modes  
- targeted corrections  

---

### 3. Tokenization as an optimization surface

Vocabulary and segmentation are treated as tunable system components, not fixed inputs.

---

### 4. Representation vs optimization separation

The results show that:

> better compression does not imply better learning under constrained training

---

## Limitations (as Findings)

- Compression gains are small (~0.0002 tpb), but consistent  
- Improvements are uneven across document types  
- Gains do not translate to training wins under strict constraints  

These define the boundary of tokenizer-only optimization.

---

## Conclusion

This work shows that:

- tokenizer compression can be improved beyond SentencePiece  
- full-validation evaluation is necessary for meaningful progress  
- failure-driven refinement is more reliable than global search  

Most importantly, it surfaces an open question:

> How should models be adapted to learn efficiently from more compressed representations?

The tokenizer is no longer the bottleneck.

The remaining problem is learning.
