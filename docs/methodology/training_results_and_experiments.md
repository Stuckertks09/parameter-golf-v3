# Training, Model Variants, and Experimental Results

## Overview

The training stage evaluates whether tokenizer improvements survive model training under the challenge constraints.  
Earlier stages produce candidate vocabularies, dynamic-programming segmentation, full-validation compression measurements, and repaired variants. The training stage tests whether those tokenizer-side gains translate into better learning.

In the refactored repository, this work is organized under:

- `scripts/70_training/`

The directory contains:

- `train_gpt.py`
- `train_gpt_custom_v1_locked.py`
- `train_gpt_custom_v1_locked_token_class_grad (1).py`
- `train_gpt_custom_v2_init_locked.py`
- `train_gpt_custom_v3_bigram.py`
- `train_gpt_mlx.py`

---

## Baseline Training Script

### `train_gpt.py`

This is the baseline training script and the reference point for later comparisons.

The script defines a default configuration with:

- 9 transformer blocks
- width 512
- 8 attention heads
- 4 KV heads (GQA)
- MLP multiplier 2
- vocabulary size 1024
- sequence length 1024
- tied embeddings
- 524,288 training tokens per step
- 20,000 nominal iterations
- 600-second wallclock cap

It also computes a tokenizer-agnostic validation metric, `val_bpb`, on the full validation split.

This script establishes:

1. the model architecture
2. the optimizer and logging defaults
3. the evaluation contract used for tokenizer experiments

---

## Locked Custom Training

### `train_gpt_custom_v1_locked.py`

This script is the direct custom-tokenizer counterpart to the baseline run.

Its role is to keep the training setup as close as possible to baseline while swapping in a custom tokenizer. It introduces `TOKENIZER_KIND`, allowing the run to switch between:

- `sentencepiece`
- `custom_jsonl`

It validates that the tokenizer vocabulary size matches `VOCAB_SIZE`, builds tokenizer-specific byte and boundary lookup tables, and logs the dataset and tokenizer configuration at runtime.

This script is the cleanest apples-to-apples comparator for:

- SP1024 baseline runs
- custom JSONL vocabulary runs
- DP-tokenized exported datasets

---

## Token-Class Gradient Scaling

### `train_gpt_custom_v1_locked_token_class_grad (1).py`

This variant extends the locked custom trainer with token-class-aware embedding optimization.

Its purpose is to treat token classes differently during training. In project usage, this was applied to classes such as:

- byte fallback tokens
- phrase tokens
- space-sensitive tokens
- structural tokens

This script represents a shift from:

> train the same model on a different tokenizer

to:

> train the model in a way that respects the token classes introduced by that tokenizer

---

## Custom Embedding Initialization

### `train_gpt_custom_v2_init_locked.py`

This script adds controlled initialization for custom-tokenizer embeddings.

It introduces:

- `CUSTOM_INIT_MODE`
- `CUSTOM_INIT_BLEND`
- `CUSTOM_INIT_SCALE`
- `CUSTOM_INIT_MIN_PARTS`
- `CUSTOM_INIT_MAX_PARTS`
- `CUSTOM_INIT_ALLOW_BYTE_FALLBACK`
- `CUSTOM_INIT_LOG_SAMPLES`

The intended use is compositional initialization, especially for phrase-like tokens built from smaller pieces. This allows newly introduced custom tokens to begin training closer to meaningful regions of embedding space instead of being fully random.

---

## Bigram Hash Augmentation

### `train_gpt_custom_v3_bigram.py`

This script extends the custom trainer with a bigram-hash augmentation path.

It adds:

- `BIGRAM_HASH`
- `BIGRAM_HASH_BUCKETS`
- `BIGRAM_HASH_SCALE`
- `BIGRAM_HASH_INIT_STD`
- `BIGRAM_HASH_DIM`

This stage explores whether the model can recover additional short-range compositional structure at runtime, even when the tokenizer already includes phrase-heavy and custom-merged tokens.

---

## MLX Port

### `train_gpt_mlx.py`

This is the MLX-side training variant.  
Within the tokenizer workstream, it is not the main experimental path, but it remains part of the repository’s training layer.

---

## Shared Training Design

Across the training scripts, several design features remain stable.

### Architecture and optimizer structure

The scripts retain the same small-model challenge setup:

- compact transformer
- tied embeddings
- grouped-query attention
- Muon for matrix parameters
- Adam for embeddings, scalar parameters, and untied heads when present

### Validation metric

All runs are measured using tokenizer-agnostic `val_bpb`, not just token loss. This is essential because different tokenizers change token counts, byte counts, and sequence structure.

### Export format awareness

The scripts assume dataset shards produced by the export pipeline and validate against the full exported validation split.

### Submission-aware postprocessing

The baseline script includes post-training int8 quantization with zlib compression to fit the challenge artifact-size limit.

---

## Experimental Questions Tested

The training stage was used to answer several distinct questions.

### 1. Baseline reference

How strong is the stock SP1024 training setup under the challenge budget?

### 2. Pure tokenizer substitution

If the model is held fixed, does replacing SentencePiece with the custom JSONL tokenizer help or hurt final `val_bpb`?

### 3. Initialization effects

If custom phrase and merge tokens are initialized compositionally rather than randomly, does early learning improve?

### 4. Token-class-aware optimization

Does scaling gradients differently by token class improve training behavior for byte, phrase, or structural tokens?

### 5. Runtime short-range structure

Can a bigram-hash path recover useful local interactions that are not fully captured by the tokenizer alone?

---

## Observed Experimental Progression

The project record across training runs shows a consistent progression.

### Stage 1: Tokenizer-side work outpaced training gains

The tokenizer and evaluation pipeline eventually produced a full-validation compression win over SentencePiece on the 50,000-document validation set. However, early training comparisons still showed that better compression alone did not guarantee better learning curves.

### Stage 2: Locked custom runs approached parity but did not immediately surpass baseline

Smoke-test comparisons on 1×H100 repeatedly suggested that a strong custom tokenizer configuration was close to baseline but still behind it in final training score. In the project record, this gap was typically discussed as being on the order of a few thousandths of `val_bpb`, often roughly `0.006–0.007` behind the SP reference in short smoke tests.

### Stage 3: Initialization and structured optimization were explored to close the gap

Because tokenizer compression alone appeared insufficient, later model variants introduced:

- phrase-compositional initialization
- token-class gradient scaling
- bigram-hash augmentation

These were attempts to turn tokenizer-side structural gains into model-side learning gains.

### Stage 4: Speed became part of the model-quality tradeoff

As the tokenizer and model variants became more specialized, run speed became a limiting factor. Reduced throughput directly lowers the number of optimization steps completed before the 600-second limit.

---

## Selected Results from the Project Record

The repository captures the training scripts, but many concrete run comparisons were recorded in the surrounding experiment threads. The following points summarize the training-side findings that informed the project direction.

### Baseline SP reference runs

Across the project record, the 8×H100 SP baseline was treated as the main benchmark. Typical outcomes discussed in the experiment notes were:

- final step counts in the low- to mid-11k range under the 600-second cap
- `val_bpb` commonly in the low `1.22x` range
- step times around the ~50 ms range on strong baseline runs

A baseline value around `1.2311` was repeatedly used as a comparison point in later analysis.

### 1×H100 smoke tests

Single-GPU smoke tests were used for rapid iteration before expensive 8×H100 confirmations.

These runs indicated:

- the custom tokenizer was within striking distance of SP
- the gap narrowed materially relative to early versions
- tokenizer compression improvements did not automatically become training wins

### Full-validation tokenizer win before training win

A major milestone in the project record was the tokenizer-only validation result for `vocab_best_v3.jsonl`, which beat SentencePiece on the full 50,000 validation documents:

- `delta_tokens = -32,149`
- `delta_tpb = -0.0002127936`

This proved that the vocabulary and DP segmentation work had crossed the compression threshold, even if model training still lagged.

### Structured failure analysis

Worst-document analysis showed that remaining tokenizer losses were concentrated in:

- OCR-heavy documents
- header-heavy pages
- transcript-like layouts
- formatting-heavy or structured text
- Shakespeare/dialogue-style formatting

This explained why training improvements were harder to realize than the aggregate tokenizer win alone might suggest.

### Speed penalties from later variants

In the project record, some later custom-vocab and training variants showed measurable step-time regression relative to earlier custom configurations and to baseline SP. This mattered because even a slightly better tokenizer can lose the overall training comparison if the model completes fewer steps before the wallclock cap.

---

## Interpretation

The training stage established a central result of the project:

> Better tokenizer compression is necessary, but not sufficient.

The work shows three distinct layers of difficulty:

1. **Compression layer**  
   Build a vocabulary and segmentation method that beats SP on tokenization efficiency.

2. **Training layer**  
   Preserve or improve model learning dynamics under the same wallclock and size constraints.

3. **Systems layer**  
   Retain enough throughput that model improvements are not erased by slower step times.

The project achieved a genuine tokenizer-side compression win and built several model-side mechanisms intended to translate that win into training performance. The remaining gap, where present, was the narrower problem of converting compression gains into end-to-end model advantage under strict runtime constraints.

---

## Role of This Stage in the Full Pipeline

The training stage sits at the end of the research loop:

1. mine recurrent structure
2. build vocabulary variants
3. apply DP tokenization
4. evaluate full-validation compression
5. repair weaknesses using worst-document analysis
6. export datasets
7. train under challenge constraints

This is the point where all earlier decisions are tested together.

---

## Summary

The `scripts/70_training/` directory contains the model-side progression of the project:

- baseline SP training
- locked custom-tokenizer substitution
- token-class-aware optimization
- custom embedding initialization
- bigram-hash augmentation

These scripts were used to test whether tokenizer-side gains could be converted into lower `val_bpb` under the challenge budget.

The experimental record shows that:

- the custom tokenizer and DP pipeline eventually achieved a real full-validation compression win over SentencePiece
- training remained the harder problem
- later work focused on initialization, token-class-aware optimization, and runtime structure to close the remaining model-side gap
- throughput and wallclock efficiency remained critical constraints throughout
