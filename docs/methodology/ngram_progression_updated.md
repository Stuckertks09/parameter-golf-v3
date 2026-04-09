# N-gram Mining and Candidate Extraction Pipeline

## Overview

The n-gram mining stage defines the process used to extract, interpret, and evaluate candidate token merges from the baseline SentencePiece token stream. This stage converts raw token adjacency patterns into structured candidate sets that inform downstream vocabulary construction.

The implementation is organized across three script directories:

* `scripts/00_baseline_sp`
* `scripts/10_dataset_analysis`
* `scripts/20_ngram_mining`

Together, these components form a reproducible pipeline that produces the artifacts contained in the `ngram/` folder.

---

## Baseline Vocabulary Reference (`00_baseline_sp`)

### Relevant script

* `13_dump_baseline_vocab.py`

### Description

This script exports the baseline SentencePiece vocabulary into a normalized, human-readable format. The whitespace marker (`▁`) is converted into a standard space representation, and special or byte-level tokens can be separated from standard vocabulary entries.

### Role in the pipeline

The baseline vocabulary serves as the reference surface for interpreting mined n-grams. All candidate merges identified in later stages are evaluated relative to the structures already represented in the baseline tokenizer.

This ensures that:

* candidate merges are not redundant with existing token coverage
* decoded n-grams can be compared directly to baseline token units
* vocabulary gaps can be identified explicitly

---

## Dataset Analysis (`10_dataset_analysis`)

### Scripts

* `analyze_fineweb_shards.py`
* `score_shards.py`

### Description

This stage evaluates the FineWeb training shards to characterize their structure, quality, and content distribution.

#### `analyze_fineweb_shards.py`

Processes shards and computes:

* token counts
* unique-token ratios
* ASCII ratios
* URL and HTML occurrence rates
* repeated-character patterns
* decoded sample previews

#### `score_shards.py`

Consumes the analysis outputs and derives ranking metrics:

* `score_diversity`
* `score_noise`
* `score_clean_prose`
* `score_priority`

Outputs include ranked shard orderings such as:

* `top_first_order.txt`
* `bottom_first_order.txt`

### Role in the pipeline

This stage establishes a structured understanding of the dataset prior to mining. It provides a basis for interpreting n-gram statistics in the context of shard quality and variability.

---

## N-gram Mining (`20_ngram_mining`)

### Scripts

* `mine_fineweb_ngrams.py`
* `decode_ngrams.py`
* `decode_trigrams.py`
* `estimate_ngram_merge_savings.py`
* `estimate_merge_savings.py`
* `simulate_merge_tokenization.py`
* `08_phrase_usage_report.py`
* `09_phrase_efficiency.py`
* `extract_sample_text.py`

---

### N-gram Extraction

#### `mine_fineweb_ngrams.py`

Scans training shards and extracts:

* bigrams
* trigrams

Each n-gram record includes:

* token ID sequence
* frequency
* shard occurrence
* left/right context information

The output of this step corresponds to early artifacts such as:

* `ngram_top.csv`
* `ngram_candidates.jsonl`

---

### Decoding

#### `decode_ngrams.py`

#### `decode_trigrams.py`

These scripts convert token ID sequences into text using the baseline tokenizer.

Outputs include:

* `decoded_bigrams.jsonl`
* `decoded_trigrams.jsonl`

Decoding enables direct inspection of mined sequences and supports classification and filtering in later stages.

---

### Candidate Filtering

Filtered candidate sets are generated from decoded outputs and stored as:

* `filtered_bigram_candidates_v1.jsonl`
* `filtered_bigram_candidates_v2.jsonl`
* `filtered_trigram_candidates.jsonl`
* `filtered_trigram_candidates_v2.jsonl`

Preview CSVs provide tabular views of these sets.

Each candidate is associated with:

* decoded text
* normalized text
* frequency
* classification label

Classification labels include:

* phrase
* single word / full word
* subword
* contraction
* allowlisted candidate

---

### Candidate Categorization

Filtered candidates are further organized into category-specific pools:

* `merge_candidates_phrases.jsonl`
* `merge_candidates_subwords.jsonl`
* `merge_candidates_contractions.jsonl`

Each category has corresponding preview CSVs for inspection.

These pools separate candidate types for independent evaluation and later recombination.

---

### Subword Curation

Artifacts:

* `subword_auto_curation_v1.csv`
* `subword_auto_curation_v2.csv`
* curated and rejected JSONL files

Each candidate is evaluated and assigned:

* a score
* an action (`keep`, `review_keep`, `drop`)
* explanatory reasons

This stage formalizes selection criteria for subword and word-like tokens.

---

### Merge Savings Estimation

#### Scripts

* `estimate_ngram_merge_savings.py`
* `estimate_merge_savings.py`

These scripts evaluate candidate merges on sampled shards.

Metrics include:

* naive token savings
* greedy token savings

Outputs include:

* `merge_savings_report.json`
* `filtered_merge_savings_report.json`
* `filtered_merge_savings_report_v2.json`
* `merge_savings_subwords_curated_v1.json`
* `merge_savings_subwords_curated_v2.json`

Greedy savings reflects overlap-aware estimates and is used as the primary metric.

---

### Merge Simulation

#### `simulate_merge_tokenization.py`

Simulates tokenization behavior under proposed merges to evaluate:

* interaction between overlapping candidates
* realistic token count reductions

Outputs include:

* `retokenization_simulation_*.json`

---

### Phrase Analysis

#### `08_phrase_usage_report.py`

#### `09_phrase_efficiency.py`

These scripts analyze phrase-level candidates by:

* usage frequency
* contribution to token reduction
* efficiency relative to other candidates

They support prioritization of phrase-based merges.

---

### Sample Inspection

#### `extract_sample_text.py`

Extracts decoded text samples for manual inspection and validation of candidate behavior within real data contexts.

---

## Final Candidate Consolidation

Final outputs include:

* `merge_candidates_final_v1.jsonl`
* `merge_savings_final_v1.json`

These combine:

* phrase candidates
* subword and word candidates
* contraction candidates

into a unified candidate set with associated savings estimates.

---

## Artifact Interpretation

### CSV Files

Preview CSVs provide human-readable summaries of candidate sets.

Common columns include:

* `freq`: occurrence count
* `normalized_text`: standardized text form
* `class`: candidate category

These files are inspection surfaces derived from JSONL artifacts.

---

### JSON / JSONL Files

* JSONL files store structured candidate records
* JSON files store aggregated metrics and evaluation outputs

These formats preserve the full pipeline state for reproducibility.

---

## Summary

The n-gram mining pipeline converts baseline token sequences into structured candidate vocabularies through:

1. baseline vocabulary inspection
2. dataset analysis
3. n-gram extraction and decoding
4. filtering and classification
5. category-based organization
6. subword curation
7. merge savings estimation and simulation
8. final candidate consolidation

The resulting artifacts form the basis for subsequent vocabulary construction and refinement stages.
