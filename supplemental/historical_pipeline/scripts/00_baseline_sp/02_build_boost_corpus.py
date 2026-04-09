from phrase_utils import load_phrase_rows, compile_phrase_patterns, replace_phrases

BOOST_INPUT = "analysis/boost_terms.txt"
OUTPUT = "analysis/spm_boost.txt"
BOOST_FACTOR = 80

rows = load_phrase_rows()
compiled = compile_phrase_patterns(rows)

with open(BOOST_INPUT, "r", encoding="utf-8") as f:
    terms = [line.strip() for line in f if line.strip()]

with open(OUTPUT, "w", encoding="utf-8") as f:
    for term in terms:
        transformed = replace_phrases(term, compiled)
        for _ in range(BOOST_FACTOR):
            f.write(transformed + "\n")

print(f"Wrote boost corpus: {len(terms)} terms x {BOOST_FACTOR}")