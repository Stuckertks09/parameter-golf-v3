import json
import argparse

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output", type=str, default="decoded_trigrams.jsonl")
    parser.add_argument("--top-n", type=int, default=5000)

    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    count = 0

    with open(args.input, "r") as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            if count >= args.top_n:
                break

            row = json.loads(line)

            if row.get("n") != 3:
                continue  # only trigrams

            ids = row["ids"]
            freq = row["freq"]

            try:
                text = sp.decode(ids)
            except Exception:
                text = "<decode_error>"

            out = {
                "ids": ids,
                "freq": freq,
                "text": text
            }

            f_out.write(json.dumps(out) + "\n")
            count += 1

            if count % 500 == 0:
                print(f"decoded {count}")

    print(f"\nDone. Wrote {count} trigrams to {args.output}")


if __name__ == "__main__":
    main()