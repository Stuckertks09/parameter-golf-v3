import json
import re
from typing import List, Tuple

PHRASE_FILE = "analysis/forced_phrases.jsonl"


def load_phrase_rows(path: str = PHRASE_FILE):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda x: (-len(x["text"]), -x["freq"]))
    return rows


def compile_phrase_patterns(rows) -> List[Tuple[re.Pattern, str, str]]:
    compiled = []
    for row in rows:
        text = row["text"]
        placeholder = row["placeholder"]

        # Match either:
        # 1. start-of-string + phrase
        # 2. whitespace + phrase
        #
        # We consume the leading whitespace when present so SentencePiece
        # doesn't emit a separate ▁ piece.
        pattern = re.compile(
            rf"(^|\s+)({re.escape(text)})(?=\s+|$|[.,!?;:])",
            flags=re.IGNORECASE,
        )
        compiled.append((pattern, text, placeholder))
    return compiled


def replace_phrases(text: str, compiled_patterns) -> str:
    out = text
    for pattern, original, placeholder in compiled_patterns:
        def repl(match):
            prefix = match.group(1)
            # If phrase is at start, no space before placeholder
            if prefix == "":
                return placeholder
            # If phrase had leading whitespace, collapse it into the placeholder
            return " " + placeholder

        out = pattern.sub(repl, out)
    return out


def restore_placeholders(text: str, rows) -> str:
    out = text
    for row in rows:
        placeholder = row["placeholder"]
        original = row["text"]
        out = out.replace(placeholder, original)
    return out