import argparse
import random
from pathlib import Path
import numpy as np
import sentencepiece as spm


MAGIC = 20240520
HEADER_INTS = 256
HEADER_DTYPE = np.dtype("<i4")
TOKEN_DTYPE = np.dtype("<u2")
HEADER_BYTES = HEADER_INTS * HEADER_DTYPE.itemsize


def load_shard_tokens(path: Path):
    header = np.fromfile(path, dtype=HEADER_DTYPE, count=HEADER_INTS)
    if int(header[0]) != MAGIC:
        raise ValueError(f"Bad shard: {path}")

    n = int(header[2])
    tokens = np.fromfile(path, dtype=TOKEN_DTYPE, count=n, offset=HEADER_BYTES)
    return tokens.astype(np.uint16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--output", type=str, default="sample_text_large.txt")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    shard_paths = sorted(dataset_dir.glob("fineweb_train_*.bin"))

    if not shard_paths:
        raise ValueError("No shards found")

    sp = spm.SentencePieceProcessor()
    if not sp.load(args.tokenizer):
        raise ValueError("Failed to load tokenizer")

    print(f"Using {len(shard_paths)} shards")

    all_lines = []

    for _ in range(args.num_samples):
        shard = random.choice(shard_paths)
        tokens = load_shard_tokens(shard)

        if len(tokens) < args.seq_len:
            continue

        start = random.randint(0, len(tokens) - args.seq_len)
        chunk = tokens[start:start + args.seq_len]

        try:
            text = sp.decode(chunk.tolist())
        except Exception:
            continue

        text = text.strip()
        if len(text) > 20:
            all_lines.append(text)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line.replace("\n", " ") + "\n")

    print(f"Wrote {len(all_lines)} lines to {output_path}")


if __name__ == "__main__":
    main()