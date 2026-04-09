import sentencepiece as spm
from phrase_utils import load_phrase_rows, compile_phrase_patterns, replace_phrases, restore_placeholders

sp = spm.SentencePieceProcessor()
sp.load("tokenizers/merge_spm.model")

rows = load_phrase_rows()
compiled = compile_phrase_patterns(rows)

tests = [
    "of the",
    "in the",
    "to the",
    "going to",
    "as well as",
    "one of the",
    "this is a test of the system",
]

for t in tests:
    transformed = replace_phrases(t, compiled)
    pieces = sp.encode(transformed, out_type=str)
    decoded = sp.decode(sp.encode(transformed, out_type=int))
    restored = restore_placeholders(decoded, rows)

    print("TEXT:       ", t)
    print("XFORM:      ", transformed)
    print("PIECES:     ", pieces)
    print("ROUNDTRIP:  ", restored)
    print()