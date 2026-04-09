# ARTIFACT INDEX

## Overview

This document maps all major artifacts produced throughout the tokenizer workstream.

Each artifact is linked to:
- The script(s) that produced it
- Its purpose
- Whether it is still relevant

---

# Legend

- **Active** → used in current or recent pipeline
- **Reference** → kept for comparison or baseline
- **Deprecated** → replaced or no longer needed
- **Intermediate** → temporary / debugging / exploration

---

# 1. Tokenizer Artifacts

## fineweb_1024_bpe.model
- **Produced by:** `04_train_tokenizer.py`
- **Type:** SentencePiece model
- **Purpose:** Baseline tokenizer used for comparison
- **Status:** Reference

---

## fineweb_1024_bpe.vocab
- **Produced by:** `04_train_tokenizer.py`
- **Type:** Vocabulary file
- **Purpose:** Baseline vocab for SP tokenizer
- **Status:** Reference

---

## baseline_vocab*.jsonl
- **Produced by:** `13_dump_baseline_vocab.py`
- **Purpose:** JSONL representation of SP vocab
- **Used for:**
  - comparisons
  - variant generation
- **Status:** Active

---

# 2. Dataset Analysis Artifacts

## shard_report.json
- **Produced by:** `analyze_fineweb_shards.py`
- **Purpose:** Per-shard statistics (token density, ratios, etc.)
- **Status:** Active

---

## shard_scores.csv
- **Produced by:** `score_shards.py`
- **Purpose:** Ranked shard quality scores
- **Used for:** training order experiments
- **Status:** Active

---

## top_first_order.txt / bottom_first_order.txt
- **Produced by:** shard scoring pipeline
- **Purpose:** Suggested curriculum ordering
- **Status:** Experimental

---

# 3. N-gram Mining Artifacts

## ngram_top.csv
- **Produced by:** `mine_fineweb_ngrams.py`
- **Purpose:** Frequency table of top n-grams
- **Status:** Active

---

## ngram_candidates.jsonl
- **Produced by:** `mine_fineweb_ngrams.py`
- **Purpose:** Raw candidate merge list
- **Status:** Intermediate

---

## decoded_bigrams.jsonl
- **Produced by:** `decode_bigrams.py`
- **Purpose:** Human-readable bigram candidates
- **Status:** Active

---

## decoded_trigrams.jsonl
- **Produced by:** `decode_trigrams.py`
- **Purpose:** Human-readable trigram candidates
- **Status:** Active

---

## mine_ngrams.log
- **Produced by:** mining pipeline
- **Purpose:** Debug/log output
- **Status:** Intermediate

---

# 4. Merge Savings & Simulation

## merge_savings_bigrams_v*.json
- **Produced by:** `estimate_merge_savings.py`
- **Purpose:** Estimated savings from bigram merges
- **Status:** Reference

---

## merge_savings_bigram_plus_trigram*.json
- **Produced by:** combined savings estimation
- **Purpose:** Combined merge evaluation
- **Status:** Reference

---

## merge_savings_subwords*.json
- **Produced by:** subword evaluation
- **Purpose:** Savings from subword candidates
- **Status:** Reference

---

## merge_savings_report.json
- **Produced by:** `estimate_merge_savings.py`
- **Purpose:** Consolidated savings analysis
- **Status:** Active

---

## filtered_merge_savings_report*.json
- **Produced by:** filtered pipelines
- **Purpose:** Post-filter savings estimates
- **Status:** Active

---

## retokenization_simulation_report.json
- **Produced by:** `simulate_merge_tokenization.py`
- **Purpose:** Simulated tokenizer behavior
- **Status:** Active

---

## retokenization_simulation_large.json
- **Produced by:** extended simulation
- **Purpose:** Large-scale simulation
- **Status:** Reference

---

# 5. Candidate Curation Artifacts

## filtered_bigram_candidates_*.jsonl
- **Produced by:** `filter_bigram_candidates_v2.py`
- **Purpose:** Cleaned bigram candidates
- **Status:** Active

---

## filtered_trigram_candidates.jsonl
- **Produced by:** trigram filtering
- **Purpose:** Cleaned trigram candidates
- **Status:** Active

---

## merge_candidates_phrases.jsonl
- **Produced by:** candidate splitting
- **Purpose:** Phrase-based candidates
- **Status:** Active

---

## merge_candidates_subwords.jsonl
- **Produced by:** candidate splitting
- **Purpose:** Subword candidates
- **Status:** Active

---

## merge_candidates_subwords_curated*.jsonl
- **Produced by:** `auto_curate_subwords_v2.py`
- **Purpose:** Final curated subwords
- **Status:** Active

---

## merge_candidates_subwords_reject*.jsonl
- **Produced by:** curation pipeline
- **Purpose:** Rejected candidates
- **Status:** Intermediate

---

## subword_auto_curation_v*.csv
- **Produced by:** `auto_curate_subwords_v2.py`
- **Purpose:** Scoring + ranking
- **Status:** Active

---

# 6. Vocabulary Artifacts

## custom_vocab_full_v*.jsonl
- **Produced by:** `build_hybrid_vocab_v*`
- **Purpose:** Full vocab variants
- **Status:** Reference (keep best version)

---

## custom_vocab.jsonl
- **Produced by:** final selection
- **Purpose:** Current working vocab
- **Status:** Active

---

## hybrid_vocab_1024.jsonl
- **Produced by:** hybrid builder
- **Purpose:** balanced vocab
- **Status:** Active

---

## vocab_v2.jsonl
- **Produced by:** earlier pipeline
- **Status:** Deprecated

---

# 7. Vocab Variant Artifacts

## vocab_variants/*.jsonl
- **Produced by:** `generate_vocab_variants.py`
- **Purpose:** candidate vocab variants
- **Status:** Active

---

## analysis/grid_vocabs/
## analysis/grid_vocabs_v2/
- **Produced by:** grid evaluation
- **Purpose:** structured experiment results
- **Status:** Active

---

# 8. Tokenizer Evaluation Artifacts

## tokenizer_gap_report.txt
- **Produced by:** `analyze_tokenizer_gaps.py`
- **Purpose:** identify failure cases
- **Status:** Active

---

## comparison logs
- **Produced by:** multiple eval scripts
- **Purpose:** debugging and analysis
- **Status:** Intermediate

---

# 9. Dataset Artifacts

## fineweb_train_*.bin
- **Produced by:** dataset export scripts
- **Purpose:** training data (custom tokenizer)
- **Status:** Active

---

## fineweb_val_*.bin
- **Produced by:** dataset export scripts
- **Purpose:** validation data
- **Status:** Active

---

## dataset export logs
- **Purpose:** validation/debugging
- **Status:** Intermediate

---

# 10. Temporary / Debug Artifacts

## *_preview.csv
- **Purpose:** manual inspection
- **Status:** Intermediate → archive candidate

---

## *_reject*.json*
- **Purpose:** rejected candidates
- **Status:** Intermediate → archive candidate

---

## *_temp*.json*
- **Purpose:** experimental outputs
- **Status:** Intermediate → archive candidate

---

# Final Note

If an artifact is:

- not referenced by any script
- not used in evaluation
- not needed for reproducibility

→ it should be archived or deleted.

This file exists to ensure:
- traceability
- reproducibility
- clarity of pipeline outputs