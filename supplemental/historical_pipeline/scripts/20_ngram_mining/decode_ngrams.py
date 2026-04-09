import argparse
import json
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n", type=int, choices=[2, 3], required=True)
    parser.add_argument("--top-n", type=int, default=5000)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    if not sp.load(args.tokenizer):
        raise ValueError(f"Failed to load tokenizer: {args.tokenizer}")

    count = 0
    with open(args.input, "r") as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            if row.get("n") != args.n:
                continue
            ids = row["ids"]
            freq = row["freq"]

            try:
                text = sp.decode(ids)
            except Exception:
                text = "<decode_error>"

            f_out.write(json.dumps({
                "n": args.n,
                "ids": ids,
                "freq": freq,
                "text": text,
            }, ensure_ascii=False) + "\n")

            count += 1
            if count >= args.top_n:
                break

    print(f"Decoded {count} {args.n}-grams to {args.output}")


if __name__ == "__main__":
    main()