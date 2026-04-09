import os
import json
import sentencepiece as spm

INPUT = "analysis/spm_training.txt"
MODEL_PREFIX = "tokenizers/merge_spm"
PHRASE_FILE = "analysis/forced_phrases.jsonl"

os.makedirs("tokenizers", exist_ok=True)

placeholders = []
with open(PHRASE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        placeholders.append(row["placeholder"])

print(f"Using {len(placeholders)} placeholder symbols")

spm.SentencePieceTrainer.train(
    input=INPUT,
    model_prefix=MODEL_PREFIX,
    vocab_size=1024,
    model_type="bpe",
    character_coverage=1.0,
    bos_id=-1,
    eos_id=-1,
    user_defined_symbols=placeholders,
    add_dummy_prefix=False,
    treat_whitespace_as_suffix=True,
)

print("Tokenizer trained -> tokenizers/merge_spm.model")