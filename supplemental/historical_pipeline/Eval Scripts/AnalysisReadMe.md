Here’s the updated pipeline, aligned to the current approach: **real-val compression first, worst-doc-driven fixes second, batch ranking third, training last**.

---

# Full Tokenizer Refinement Pipeline

## 1. Compare full validation compression

**Script:**
[`analysis/Eval Scripts/compare_full_val_compression.py`](analysis/Eval%20Scripts/compare_full_val_compression.py)

**Purpose:**
Measure whether the current custom vocab is actually better or worse than SP on the **real 50k validation docs**.

**Run:**

```bash
python "analysis/Eval Scripts/compare_full_val_compression.py" \
  --docs-jsonl data/docs_selected.jsonl \
  --num-val-docs 50000 \
  --sp-model data/tokenizers/fineweb_1024_bpe.model \
  --vocab-jsonl vocab/vocab_best.jsonl \
  --top-k 200 \
  --output-dir analysis
```

**Outputs:**

* `full_val_summary_*.json`
* `full_val_worst_*.csv`
* `full_val_delta_histogram_*.csv`

**What matters:**

* `delta_tpb`
* `delta_tokens`
* worst offending docs

**Current real result:**

```text
docs                 50000
bytes                151080645
sp_tokens            61971846
custom_tokens        62070660
delta_tokens         98814
delta_tpb            +0.000654048
custom_fallbacks     15220593
custom_fallback_runs 8754977
```

That means the current custom vocab is **slightly worse than SP on real validation compression**, even though it looked competitive on smaller proxy tests.

---

## 2. Generate vocab fixes from worst docs

**Script:**
[`analysis/Eval Scripts/propose_vocab_fixes_from_worst_docs.py`](analysis/Eval%20Scripts/propose_vocab_fixes_from_worst_docs.py)

**Purpose:**
Turn the worst failing docs into **small, structured vocab corrections**.

**Current behavior:**

* only proposes **line/token-aligned candidates**
* favors:

  * newline-prefixed spans
  * headers
  * structured phrases
  * punctuation-aware multiword spans
* rejects:

  * arbitrary substrings
  * glue fragments
  * junk like `e t`, `s a`, `n th`

**Run:**

```bash
python "analysis/Eval Scripts/propose_vocab_fixes_from_worst_docs.py" \
  --worst-docs-csv analysis/full_val_worst_20260403_125718.csv \
  --docs-jsonl data/docs_selected.jsonl \
  --vocab-jsonl vocab/vocab_best.jsonl \
  --top-k-docs 200 \
  --num-swaps 20 \
  --output-dir analysis

  python "analysis/Eval Scripts/propose_vocab_fixes_structured_bias.py" \
  --worst-docs-csv analysis/full_val_worst_20260403_125718.csv\
  --docs-jsonl data/docs_selected.jsonl \
  --vocab-jsonl vocab/vocab_best_v2.jsonl \
  --top-k-docs 200 \
  --num-swaps 20 \
  --output-dir analysis

  python "analysis/Eval Scripts/propose_vocab_fixes_from_worst_docs.py" \
  --worst-docs-csv analysis/full_val_worst_20260403_125718.csv \
  --vocab-jsonl vocab/vocab_best_v2.jsonl \
  --output-dir analysis

  python "workspace/parameter-golf/analysis/Eval Scripts/propose_vocab_fixes_from_worst_docs.py" \
  --worst-docs-csv workspace/parameter-golf/analysis/full_val_worst_20260405_005155.csv \
  --docs-jsonl workspace/parameter-golf/data/docs_selected.jsonl \
  --vocab-jsonl workspace/parameter-golf/vocab/vocab_best_v3.jsonl \
  --top-k-docs 300 \
  --num-swaps 10 \
  --output-dir workspace/parameter-golf/analysis
```

**Outputs:**

* `worst_docs_add_candidates_*.csv`
* `worst_docs_remove_candidates_*.csv`
* `worst_docs_vocab_swaps_*.csv`
* `worst_docs_vocab_report_*.json`
* `vocab_best_worstdocs_fix_*.jsonl`

**Goal:**
Propose **small token swaps** that specifically patch the structured/noisy regions where custom loses to SP.

---

## 3. Rank vocab variants on full validation docs

**Script:**
[`analysis/Eval Scripts/rank_vocab_variants_full_val.py`](analysis/Eval%20Scripts/rank_vocab_variants_full_val.py)

**Purpose:**
Batch-test multiple proposed vocab variants directly on the full 50k validation docs, without training.

**Run:**

```bash
python "analysis/Eval Scripts/rank_vocab_variants_full_val.py" \
  --docs-jsonl data/docs_selected.jsonl \
  --num-val-docs 50000 \
  --sp-model data/tokenizers/fineweb_1024_bpe.model \
  --variant-dir analysis \
  --glob "vocab_best_worstdocs_fix_*.jsonl" \
  --output-dir analysis
```

**Outputs:**

* `rank_vocab_variants_*.json`
* `rank_vocab_variants_*.csv`

**What matters:**

* rank by `delta_tpb`
* keep only variants that improve over current `vocab_best`

This avoids exporting datasets and training bad variants.

---

## 4. Only train the winner

Once a variant improves `delta_tpb`, then:

1. export shards with that vocab
2. run training
3. compare against baseline and current custom path

This is the filter:

```text
compression win first → training second
```

---

# Why this pipeline exists

Earlier, the project drifted sideways because tokenizer progress was being judged on:

* sampled texts
* partial diagnostics
* training outcomes without isolating compression quality

The current pipeline fixes that.

It forces this order:

```text
real eval compression
→ identify real failures
→ generate small targeted fixes
→ rank variants
→ only then train
```

That keeps the custom path disciplined.

---

# What this pipeline is solving

The main failure mode is **not broad natural language**.

The custom vocab is mostly losing on:

* structured documents
* OCR/noisy newspaper text
* newline-heavy formatting
* headers / caps / transcript-style layouts

Examples from worst docs included:

* Shakespeare/play formatting
* newspaper OCR pages
* header-heavy scraped pages
* structured legal / terms pages

So the right move is **not** rebuilding vocab from scratch.
It is **small, data-driven correction** around `vocab_best.jsonl`.

---

# Strategic rule

Do **not**:

* reopen giant vocab search
* randomly tweak SP
* retrain every candidate

Do:

* keep `vocab_best` as parent
* make small worst-doc-driven variants
* rank them on full-val compression
* only train clear improvements

---

# Current status

* `vocab_best` is close, but still slightly behind SP on real validation compression
* the gap is small enough to attack surgically
* init experiments did not help
* the next real leverage is **vocab correction on real eval failures**

---

# MD version

````md
# Tokenizer Refinement Pipeline

## Why this exists

The custom tokenizer looked competitive on smaller proxy tests, but real full-validation compression showed it was still slightly worse than SentencePiece.

Current real result:

```text
docs                 50000
bytes                151080645
sp_tokens            61971846
custom_tokens        62070660
delta_tokens         98814
delta_tpb            +0.000654048
custom_fallbacks     15220593
custom_fallback_runs 8754977
````

This means the custom vocab is close, but still behind on the actual evaluation distribution.

The pipeline below exists to fix that with **small, targeted vocab updates** instead of broad random experimentation.

---

## 1. Compare Compression

**Script:**
[compare_full_val_compression.py](analysis/Eval%20Scripts/compare_full_val_compression.py)

**Run:**

```bash
python "analysis/Eval Scripts/compare_full_val_compression.py" \
  --docs-jsonl data/docs_selected.jsonl \
  --num-val-docs 50000 \
  --sp-model data/tokenizers/fineweb_1024_bpe.model \
  --vocab-jsonl vocab/vocab_best.jsonl \
  --top-k 200 \
  --output-dir analysis
```

**What it does:**

* compares SP vs current custom vocab on the real 50k validation docs
* outputs:

  * summary JSON
  * worst docs CSV
  * delta histogram CSV

**Why it matters:**
This is the real ground-truth tokenizer evaluation.

---

## 2. Generate Vocab Fixes

**Script:**
[propose_vocab_fixes_from_worst_docs.py](analysis/Eval%20Scripts/propose_vocab_fixes_from_worst_docs.py)

**Run:**

```bash
python "analysis/Eval Scripts/propose_vocab_fixes_from_worst_docs.py" \
  --worst-docs-csv analysis/full_val_worst_YYYYMMDD_HHMMSS.csv \
  --docs-jsonl data/docs_selected.jsonl \
  --vocab-jsonl vocab/vocab_best.jsonl \
  --top-k-docs 200 \
  --num-swaps 20 \
  --output-dir analysis
```

**What it does:**

* reads the worst failing docs
* proposes structured, token-aligned additions
* avoids junk substring proposals
* outputs:

  * add candidates CSV
  * remove candidates CSV
  * swap proposal CSV
  * new vocab JSONL

**Why it matters:**
This turns failure cases into small, concrete vocab corrections.

---

## 3. Rank Vocab Variants

**Script:**
[rank_vocab_variants_full_val.py](analysis/Eval%20Scripts/rank_vocab_variants_full_val.py)

**Run:**

```bash
python "analysis/Eval Scripts/rank_vocab_variants_full_val.py" \
  --docs-jsonl data/docs_selected.jsonl \
  --num-val-docs 50000 \
  --sp-model data/tokenizers/fineweb_1024_bpe.model \
  --variant-dir analysis \
  --glob "vocab_best_worstdocs_fix_*.jsonl" \
  --output-dir analysis
```

**What it does:**

* batch-tests proposed vocab variants on the real validation docs
* ranks them by:

  * `delta_tpb`
  * `delta_tokens`

**Why it matters:**
Only the variants that improve compression should move forward to export/training.

---

## 4. Train only the winner

Once a variant improves full-val compression:

1. export dataset
2. train
3. compare against baseline and prior custom runs

Rule:

```text
compression win first → training second
```

---

## Strategy

Do **not**:

* rebuild vocab from scratch
* run large random searches
* train every variant

Do:

* keep `vocab_best.jsonl` as the parent
* make small worst-doc-driven fixes
* rank them on real val docs
* only train clear improvements

This keeps the custom tokenizer path disciplined and anchored to the actual evaluation distribution.

```

If you want, I can also turn this into a polished `TOKENIZER_PIPELINE.md` file block with a short intro and section headers ready to paste.
```
