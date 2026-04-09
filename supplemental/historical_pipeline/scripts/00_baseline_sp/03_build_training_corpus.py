from phrase_utils import load_phrase_rows, compile_phrase_patterns, replace_phrases

BOOST = "analysis/spm_boost.txt"
SOURCE = "sample_text_large.txt"
OUTPUT = "analysis/spm_training.txt"

rows = load_phrase_rows()
compiled = compile_phrase_patterns(rows)

written = 0

with open(OUTPUT, "w", encoding="utf-8") as out:
    with open(BOOST, "r", encoding="utf-8") as f:
        for line in f:
            out.write(line.rstrip("\n") + "\n")
            written += 1

    with open(SOURCE, "r", encoding="utf-8") as f:
        for line in f:
            transformed = replace_phrases(line.rstrip("\n"), compiled)
            out.write(transformed + "\n")
            written += 1

print(f"Wrote training corpus -> {OUTPUT} ({written} lines)")