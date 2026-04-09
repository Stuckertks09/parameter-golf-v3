# Dynamic Programming Tokenization

## Overview

The dynamic programming (DP) tokenization stage defines how a constructed vocabulary is applied to text. It replaces greedy segmentation with a global optimization strategy that selects token sequences based on overall sequence efficiency rather than local decisions.

This stage operates between vocabulary construction and tokenizer evaluation. It determines how tokens are selected during encoding and directly impacts token counts, fallback behavior, and downstream training performance.

---

## Motivation

Standard greedy tokenization selects the longest or highest-priority token at each position without considering future context. This can lead to suboptimal segmentations when multiple valid tokenizations exist.

Dynamic programming introduces a global view of the sequence, allowing the tokenizer to:

* evaluate multiple segmentation paths
* minimize total token count
* reduce fallback usage
* prefer structurally coherent token boundaries

---

## Implementation Context

DP tokenization is used with custom vocabularies defined in JSONL format and is enabled through:

* `TOKENIZER_KIND=custom_jsonl`

This tokenizer operates during:

* evaluation
* dataset export
* training

It replaces SentencePiece segmentation with a custom encoding strategy.

---

## Core Mechanism

The DP tokenizer treats tokenization as a sequence optimization problem.

Given an input string, it constructs possible token matches at each position and computes the optimal segmentation using dynamic programming.

### Objective

Minimize total cost over the sequence.

Typical cost components include:

* token count (primary objective)
* penalties for fallback tokens
* structural preferences (optional)

---

## DP Formulation

Let the input be a sequence of characters of length `N`.

Define:

* `dp[i]`: minimum cost to tokenize the prefix ending at position `i`

For each position `i`, the tokenizer considers all tokens that match starting at `i` and updates:

```
dp[j] = min(dp[j], dp[i] + cost(token_i_j))
```

where `j` is the end position of the token.

The optimal segmentation is recovered by backtracking from `dp[N]`.

---

## Candidate Matching

At each position, the tokenizer identifies all tokens in the vocabulary that match the current substring.

These include:

* phrase tokens
* word tokens
* subword tokens
* byte fallback tokens

Matching is typically implemented using prefix lookup over the vocabulary.

---

## Fallback Handling

When no higher-level token matches, the tokenizer falls back to byte-level tokens.

Fallback tokens ensure full coverage but are penalized in the DP objective.

This encourages the tokenizer to:

* prefer higher-level tokens when available
* avoid long sequences of byte tokens

---

## Tie-Breaking and Scoring

When multiple segmentations have equal token counts, additional scoring rules may be applied.

These can include:

* preferring longer tokens
* penalizing consecutive fallback tokens
* favoring tokens with better boundary alignment

Tie-breaking rules influence structural coherence without changing the primary objective.

---

## Differences from Greedy Tokenization

| Aspect            | Greedy         | Dynamic Programming         |
| ----------------- | -------------- | --------------------------- |
| Decision scope    | Local          | Global                      |
| Optimality        | Not guaranteed | Optimal under cost function |
| Fallback handling | Reactive       | Penalized and minimized     |
| Boundary quality  | Variable       | Controlled via scoring      |

DP eliminates cases where greedy selection leads to locally optimal but globally suboptimal tokenization.

---

## Impact on Tokenization Behavior

### Token Count

DP reduces total token count by selecting globally optimal segmentations.

### Fallback Usage

Fallback token frequency and run length are reduced due to explicit penalties.

### Structural Consistency

DP produces more consistent token boundaries across similar contexts.

---

## Integration with Pipeline

DP tokenization is used in three stages:

1. **Evaluation**

   * compute token counts and fallback metrics

2. **Dataset export**

   * generate training shards using DP segmentation

3. **Training**

   * model consumes DP-tokenized sequences

---

## Relationship to Vocabulary Design

Vocabulary construction defines available tokens, while DP determines how effectively those tokens are used.

A vocabulary with strong candidates can still perform poorly under greedy segmentation. DP enables the tokenizer to fully exploit:

* phrase tokens
* multi-token compositions
* overlapping candidate structures

---

## Summary

Dynamic programming tokenization replaces greedy segmentation with a global optimization framework.

It:

* selects token sequences that minimize total cost
* reduces fallback usage
* improves structural consistency
* enables more effective use of custom vocabularies

This stage is critical for realizing the benefits of vocabulary construction and directly influences downstream evaluation and training results.
