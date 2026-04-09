# Vocabulary Construction and Variant Evaluation

## Overview

The vocabulary construction stage converts curated n-gram candidates into complete 1024-token vocabularies. This stage is implemented across two primary locations in the repository:

* `analysis/`
* `scripts/40_vocab_building/`

The `analysis/` directory contains vocabulary artifacts, intermediate outputs, and evaluation results. The `scripts/40_vocab_building/` directory contains the construction logic used to generate and vary these vocabularies.

This stage represents the transition from candidate discovery to explicit token allocation.

---

## Analysis Directory (`analysis/`)

The `analysis/` directory stores all vocabulary artifacts and evaluation outputs produced during this stage.

### Baseline references

* `baseline_vocab.jsonl`
* `baseline_vocab_all.jsonl`
* `baseline_vocab_clean.jsonl`

These files represent the baseline SentencePiece vocabulary in different forms, including cleaned and filtered variants used for comparison and hybrid construction.

---

### Custom vocabulary outputs

* `custom_vocab.jsonl`
* `custom_vocab_full.jsonl`
* `custom_vocab_full_v2.jsonl`
* `custom_vocab_full_v3.jsonl`
* `custom_vocab_full_v5.jsonl`
* `custom_vocab_full_v6.jsonl`
* `custom_vocab_full_v7.jsonl`

These files represent successive iterations of fully custom vocabularies derived from curated candidate pools.

Each version reflects changes in:

* candidate filtering rules
* allocation strategies
* inclusion or exclusion of specific token classes

---

### Hybrid vocabulary outputs

* `hybrid_vocab_1024.jsonl`

Hybrid vocabularies combine baseline tokens with curated candidate merges. These vocabularies explicitly divide the token budget between preserved baseline pieces and newly introduced tokens.

---

### Variant search outputs

* `grid_vocabs/`
* `grid_vocabs_v2/`
* `vocabs/`

These directories contain collections of generated vocabulary variants produced through grid-based and search-based construction processes.

---

### Evaluation outputs

* `variant_eval_sp_only_50k.csv`
* `variant_eval_sp_shaped_ranked_50k.csv`
* `variant_eval_sp_shaped_safe_50k.csv`

These files evaluate vocabulary variants on a fixed 50,000-document slice.

Each evaluation includes:

* `sp_tokens`
* `custom_tokens`
* `token_delta_sp_minus_custom`
* `ratio_custom_over_sp`
* fallback token counts
* fallback share
* document-level win/loss counts

These metrics are used to compare custom vocabularies against the baseline tokenizer.

---

### Control inputs

* `boost_terms.txt`
* `forced_phrases.jsonl`
* `spm_boost.txt`
* `spm_training.txt`

These files define constraints and inputs used during vocabulary construction, including explicitly prioritized terms and phrases.

---

## Vocabulary Construction Scripts (`scripts/40_vocab_building/`)

This directory contains the scripts used to build, expand, and evaluate vocabularies.

### Script list

* `10_build_vocab_draft.py`
* `11_build_full_vocab.py`
* `14_build_hybrid_vocab.py`
* `15_build_hybrid_vocab_v2.py`
* `16_build_hybrid_vocab_v3.py`
* `build_hybrid_grid.py`
* `build_hybrid_local_search.py`
* `generate_vocab_variants.py`
* `export_top_variants.py`

---

## Draft Vocabulary Construction

### `10_build_vocab_draft.py`

This script creates an initial vocabulary draft and writes it to:

* `analysis/custom_vocab.jsonl`

The draft vocabulary is composed of predefined groups:

* phrase-level tokens
* full-word candidates
* subword fragments
* character and punctuation coverage

Each token is assigned:

* a `kind`
* a `priority`

Duplicate entries are resolved by retaining the highest-priority version.

This stage establishes a seed vocabulary rather than a complete allocation.

---

## Full Vocabulary Assembly

### `11_build_full_vocab.py`

This script expands the draft vocabulary into a complete 1024-token vocabulary:

* `analysis/custom_vocab_full.jsonl`

Inputs:

* draft vocabulary
* n-gram candidate file (`merge_candidates_bigram_plus_trigram_v2.jsonl`)

Candidate inclusion is restricted to specific classes:

* `keep_allowlist`
* `keep_short_phrase`
* `keep_full_word`
* `keep_contraction`
* `curated_subword_auto_v2`

The script also enforces:

* ASCII coverage
* Unicode punctuation coverage
* inclusion of punctuation tokens
* byte fallback coverage
* reserved token padding

The resulting vocabulary is a structured combination of:

* curated candidates
* required coverage tokens
* fallback tokens

---

## Hybrid Vocabulary Construction

### `14_build_hybrid_vocab.py`

This script constructs a hybrid vocabulary:

* `analysis/hybrid_vocab_1024.jsonl`

Inputs:

* `baseline_vocab_clean.jsonl`
* merge candidate file

The token budget is divided between:

* baseline tokens
* phrase candidates
* word/subword candidates
* byte fallback tokens

This approach preserves a portion of the baseline vocabulary while introducing selected candidate merges.

---

### `16_build_hybrid_vocab_v3.py`

This script extends the hybrid approach with configurable allocation parameters.

Output example:

* `analysis/custom_vocab_full_b-priority_p-combined_w-combined.jsonl`

Configuration includes:

* number of baseline tokens
* number of phrase tokens
* number of word/subword tokens
* sorting methods for each category

Additional elements include:

* required Unicode punctuation
* pinned token lists
* byte fallback coverage
* reserved token padding

---

## Variant Generation and Search

### Scripts

* `build_hybrid_grid.py`
* `build_hybrid_local_search.py`
* `generate_vocab_variants.py`
* `export_top_variants.py`

These scripts generate multiple vocabulary variants using:

* grid-based parameter sweeps
* local search strategies

Outputs are written to directories such as:

* `analysis/grid_vocabs/`
* `analysis/grid_vocabs_v2/`

These variants are later evaluated using the evaluation pipeline.

---

## Evaluation Metrics

Evaluation files compare vocabulary variants against the baseline tokenizer.

Key metrics include:

* total token count
* token difference vs baseline
* ratio of custom tokens to baseline tokens
* fallback token usage
* fallback share
* document-level performance comparisons

These metrics quantify the effectiveness of each vocabulary under realistic tokenization conditions.

---

## Summary

The vocabulary construction stage converts curated n-gram candidates into complete tokenizer vocabularies through:

1. baseline vocabulary reference
2. draft vocabulary creation
3. full vocabulary assembly
4. hybrid vocabulary construction
5. variant generation and search
6. evaluation on held-out data

The output of this stage is a collection of fully specified 1024-token vocabularies with measurable performance characteristics.
