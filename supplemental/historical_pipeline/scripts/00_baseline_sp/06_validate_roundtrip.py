import sentencepiece as spm
from phrase_utils import load_phrase_rows, compile_phrase_patterns, replace_phrases, restore_placeholders

sp = spm.SentencePieceProcessor()
sp.load("tokenizers/merge_spm.model")

rows = load_phrase_rows()
compiled = compile_phrase_patterns(rows)

samples = [
    "this is a test",
    "of the system",
    "going to the store",
    "as well as possible",
    "one of the best examples",
]

for s in samples:
    x = replace_phrases(s, compiled)
    ids = sp.encode(x, out_type=int)
    dec = sp.decode(ids)
    restored = restore_placeholders(dec, rows)

    if restored != s:
        print("FAIL:", s, "->", restored)
    else:
        print("OK:", s)