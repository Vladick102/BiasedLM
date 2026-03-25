#!/usr/bin/env python3
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List


ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTINEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(text: Any) -> str:
    """
    Normalize text for QA dataset creation.

    What it does:
    - cast to string
    - Unicode normalization
    - remove BOM / zero-width chars
    - unify apostrophes and quotes
    - unify dashes
    - normalize line breaks
    - collapse excessive spaces
    - trim outer whitespace
    """
    if text is None:
        return ""

    text = str(text)

    # Normalize unicode compatibility forms
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove BOM / zero-width chars
    text = ZERO_WIDTH_RE.sub("", text)

    # Unify apostrophes
    apostrophe_map = {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201B": "'",  # single high-reversed-9 quote
        "\u2032": "'",  # prime
        "\u02BC": "'",  # modifier letter apostrophe
        "\uFF07": "'",  # fullwidth apostrophe
        "`": "'",       # grave accent -> apostrophe
        "´": "'",       # acute accent -> apostrophe
    }
    for src, dst in apostrophe_map.items():
        text = text.replace(src, dst)

    # Unify double quotes
    quote_map = {
        "\u201C": '"',  # left double quote
        "\u201D": '"',  # right double quote
        "\u201E": '"',  # double low-9 quote
        "\u201F": '"',  # double high-reversed-9 quote
        "\u2033": '"',  # double prime
        "\uFF02": '"',  # fullwidth quote
    }
    for src, dst in quote_map.items():
        text = text.replace(src, dst)

    # Unify dashes
    dash_map = {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
    }
    for src, dst in dash_map.items():
        text = text.replace(src, dst)

    # Replace non-breaking spaces etc.
    text = text.replace("\u00A0", " ")
    text = text.replace("\u202F", " ")
    text = text.replace("\u2009", " ")

    # Clean whitespace around newlines
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

    # Collapse repeated spaces
    text = MULTISPACE_RE.sub(" ", text)

    # Collapse too many blank lines
    text = MULTINEWLINE_RE.sub("\n\n", text)

    return text.strip()


def strip_accidental_outer_quotes(text: str) -> str:
    """
    Remove accidental wrapping quotes only when the whole string
    is wrapped once, e.g. '"hello"' -> 'hello'

    Does NOT remove meaningful internal quotes.
    """
    if len(text) >= 2:
        if (text[0] == text[-1]) and text[0] in {"'", '"'}:
            return text[1:-1].strip()
    return text


def clean_qa_field(text: Any) -> str:
    text = normalize_text(text)
    text = strip_accidental_outer_quotes(text)
    return text


def iter_json_objects(text: str) -> Iterable[Dict[str, Any]]:
    """
    Robust parser for:
    1) a JSON array of objects
    2) a single JSON object
    3) concatenated JSON objects / JSONL-like content
    """
    text = text.strip()
    if not text:
        return

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            yield parsed
            return
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    yield item
            return
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)

    while idx < n:
        while idx < n and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= n:
            break

        obj, next_idx = decoder.raw_decode(text, idx)
        if isinstance(obj, dict):
            yield obj
        idx = next_idx


def flatten_pairs(records: Iterable[Dict[str, Any]], deduplicate: bool = True) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()

    for record in records:
        pairs = record.get("pairs", [])
        if not isinstance(pairs, list):
            continue

        for pair in pairs:
            if not isinstance(pair, dict):
                continue

            question = clean_qa_field(pair.get("question", ""))
            answer = clean_qa_field(pair.get("answer", ""))

            if not question or not answer:
                continue

            item = {
                "question": question,
                "answer": answer,
            }

            if deduplicate:
                key = (question, answer)
                if key in seen:
                    continue
                seen.add(key)

            rows.append(item)

    return rows


def save_jsonl(rows: List[Dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer"],
            quoting=csv.QUOTE_ALL,  #quotes all
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten nested article->pairs dataset into clean QA dataset."
    )
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--output", required=True, help="Path to output file")
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output format. Recommended: jsonl",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate question-answer pairs",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    records = list(iter_json_objects(text))
    rows = flatten_pairs(records, deduplicate=not args.keep_duplicates)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "jsonl":
        save_jsonl(rows, output_path)
    else:
        save_csv(rows, output_path)

    print(f"Parsed top-level records: {len(records)}")
    print(f"Flattened QA rows: {len(rows)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()