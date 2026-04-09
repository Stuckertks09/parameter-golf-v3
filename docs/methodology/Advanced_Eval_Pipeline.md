# Advanced Evaluation Pipeline

## Overview

The advanced evaluation pipeline extends basic tokenizer evaluation into a refinement loop grounded in full-validation compression and document-level failure analysis. In the repository, this stage is implemented across two directories:

* `Eval Scripts/`
* `Doc_analysis/`

The `Eval Scripts/` directory contains the executable refinement pipeline: `AnalysisReadMe.md`, `compare_full_val_compression.py`, `fill_reserved_slots.py`, `propose_vocab_fixes_from_worst_docs.py`, `propose_vocab_fixes_structured_bias.py`, and `rank_vocab_variants_full_val.py`. ([github.com](https://github.com/Stuckertks09/parameter-golf-v1/tree/main/Eval%20Scripts))

The `Doc_analysis/` directory stores outputs from that pipeline, including `full_val_summary_20260405_005155.json`, `full_val_worst_20260405_005155.csv`, `full_val_delta_histogram_20260405_005155.csv`, `worst_docs_add_candidates_20260405_005420.csv`, `worst_docs_remove_candidates_20260405_005420.csv`, and `vocab_best_worstdocs_fix_20260405_005420.jsonl`. ([github.com](https://github.com/Stuckertks09/parameter-golf-v1/tree/main/Doc_analysis))

---

## Pipeline Structure

`AnalysisReadMe.md` defines the evaluation order explicitly:

1. compare full-validation compression
2. generate vocabulary fixes from worst documents
3. rank vocabulary variants on the full validation set
4. train only the winner

The stated purpose of this ordering is to enforce `compression win first → training second`, so that vocabulary refinement is driven by real validation compression before any dataset export or training run is performed. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Stuckertks09/parameter-golf-v1/main/Eval%20Scripts/AnalysisReadMe.md))

---

## Full-Validation Compression

### Script

* `compare_full_val_compression.py`

### Role

This script measures whether a custom vocabulary is better or worse than the baseline SentencePiece tokenizer on the full 50,000 validation documents. `AnalysisReadMe.md` identifies its primary outputs as:

* `full_val_summary_*.json`
* `full_val_worst_*.csv`
* `full_val_delta_histogram_*.csv`

and identifies the key metrics as:

* `delta_tpb`
* `delta_tokens`
* worst offending documents

These outputs turn tokenizer evaluation into a full-validation compression study rather than a proxy-score estimate. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Stuckertks09/parameter-golf-v1/main/Eval%20Scripts/AnalysisReadMe.md))

### Recorded result

The stored summary file `full_val_summary_20260405_005155.json` reports the following full-validation outcome for `vocab_best_v3.jsonl`:

* `docs`: 50,000
* `bytes`: 151,080,645
* `sp_tokens`: 61,971,846
* `custom_tokens`: 61,939,697
* `delta_tokens`: -32,149
* `delta_tpb`: -0.00021279363746429247
* `custom_fallback_tokens`: 15,057,908
* `custom_fallback_runs`: 8,674,710

A negative `delta_tokens` and negative `delta_tpb` indicate that the custom vocabulary out-compressed SentencePiece on the full validation set in this run. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Stuckertks09/parameter-golf-v1/main/Doc_analysis/full_val_summary_20260405_005155.json))

---

## Worst-Document Analysis

### Script

* `propose_vocab_fixes_from_worst_docs.py`
* `propose_vocab_fixes_structured_bias.py`

### Role

`AnalysisReadMe.md` defines this stage as a mechanism for generating small, structured vocabulary corrections from the documents where the custom tokenizer loses most heavily to SentencePiece. The README states that these scripts favor:

* newline-prefixed spans
* headers
* structured phrases
* punctuation-aware multiword spans

and reject:

* arbitrary substrings
* glue fragments
* low-quality junk fragments

This reframes vocabulary refinement as targeted repair rather than full vocabulary reconstruction. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Stuckertks09/parameter-golf-v1/main/Eval%20Scripts/AnalysisReadMe.md))

### Stored worst-document outputs

The file `full_val_worst_20260405_005155.csv` ranks the largest custom losses by document. The highest-loss examples include:

* `doc_idx 38634`, with `delta_tokens = 2431`, containing Shakespeare-style speaker formatting from *A Midsummer Night’s Dream*
* `doc_idx 1455`, with `delta_tokens = 1298`, containing narrative prose with dense dialogue and scene formatting
* `doc_idx 4976`, with `delta_tokens = 1000`, containing OCR-like historical newspaper text
* multiple additional entries dominated by newspaper OCR, structured headers, legal terms pages, transcript-like layouts, and formatting-heavy web text

The worst-document set shows that tokenizer losses are concentrated in structured, noisy, header-heavy, and OCR-like material rather than ordinary prose. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Stuckertks09/parameter-golf-v1/main/Doc_analysis/full_val_worst_20260405_005155.csv))

---

## Candidate Generation From Worst Documents

### Stored proposal artifacts

* `worst_docs_add_candidates_20260405_005420.csv`
* `worst_docs_remove_candidates_20260405_005420.csv`
* `worst_docs_vocab_report_20260405_005420.json`
* `vocab_best_worstdocs_fix_20260405_005420.jsonl`

The vocabulary report records the exact refinement configuration used for the stored proposal:

* input vocabulary: `vocab_best_v3.jsonl`
* top documents analyzed: `300`
* swaps requested: `10`
* minimum piece length: `4`
* maximum piece length: `24`
* minimum document hits: `3`
* candidates scored: `48,205`

It also records a 10-swap preview that replaces tokens such as `the first`, `should be`, `about the`, `with the`, `from the`, and `going to` with candidates such as `has been`, `one of the`, `, however`, `United States`, `I don't`, and `and a`. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Stuckertks09/parameter-golf-v1/main/Doc_analysis/worst_docs_vocab_report_20260405_005420.json))

The add-candidate file shows that the highest-scoring additions are dominated by structured and high-reuse phrases rather than isolated fragments. Top entries include `has been`, `one of the`, `, however`, `is not`, `United States`, `I don't`, `of this`, `and a`, `as the`, and `there is`. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Stuckertks09/parameter-golf-v1/main/Doc_analysis/worst_docs_add_candidates_20260405_005420.csv))

The remove-candidate file shows that the proposed removals come primarily from existing phrase slots and reserved tail positions. Examples include `the first`, `should be`, `about the`, `, and the`, `with the`, `from the`, `that the`, `going to`, `the same`, and `you have`, along with reserved IDs `984` through `991`. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Stuckertks09/parameter-golf-v1/main/Doc_analysis/worst_docs_remove_candidates_20260405_005420.csv))

---

## Variant Ranking

### Script

* `rank_vocab_variants_full_val.py`

### Role

`AnalysisReadMe.md` defines this script as the batch-ranking step that tests multiple proposed vocabularies directly on the full 50,000 validation documents without training. Its outputs are:

* `rank_vocab_variants_*.json`
* `rank_vocab_variants_*.csv`

and the ranking criterion is `delta_tpb`. The intent is to keep only variants that improve over the current parent vocabulary before any training is run. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Stuckertks09/parameter-golf-v1/main/Eval%20Scripts/AnalysisReadMe.md))

---

## Reserved-Slot Completion

### Script

* `fill_reserved_slots.py`

### Role

This script appears in the evaluation pipeline directory as part of the same refinement toolchain. Its placement indicates that vocabulary repair and full-validation ranking are performed on complete vocabularies rather than partially specified candidate sets. ([github.com](https://github.com/Stuckertks09/parameter-golf-v1/tree/main/Eval%20Scripts))

---

## Document Analysis as a Research Artifact

The `Doc_analysis/` directory preserves the full state of a refinement cycle:

* a full-validation summary
* a worst-document loss table
* a delta histogram
* add/remove candidate lists
* a structured vocabulary report
* a generated repaired vocabulary

This turns tokenizer refinement into a traceable analysis workflow. Instead of describing tokenizer quality with a single aggregate number, the pipeline records the distribution of failures, the document classes where losses occur, the candidate tokens proposed to repair those losses, and the exact vocabulary changes advanced to the next evaluation round. ([github.com](https://github.com/Stuckertks09/parameter-golf-v1/tree/main/Doc_analysis))

---

## Summary

The advanced evaluation pipeline is a full-validation refinement loop built around real compression outcomes rather than sampled heuristics.

It proceeds by:

1. measuring compression on all 50,000 validation documents
2. identifying the worst failing documents
3. proposing small, structured token swaps from those documents
4. ranking repaired vocabularies on full-validation compression
5. advancing only the best-performing variants to training

The stored outputs in `Doc_analysis/` show that this process was used to move from a slightly losing custom vocabulary toward a full-validation compression win, while also localizing the remaining weaknesses to structured, OCR-like, and formatting-heavy documents. ([raw.githubusercontent.com](https://raw.githubusercontent.com/Stuckertks09/parameter-golf-v1/main/Eval%20Scripts/AnalysisReadMe.md))
