"""Build citations.json from explicit article cross-references in chunks.json.

Scans every chunk for patterns like「第三十六条」and resolves them to
(src_chunk_id, target_article_id) pairs saved to data/chunks/citations.json.

Also handles 「前款」 as intra-article paragraph references by pointing to
the same article_id (CitationGraph will expand to all sibling paragraphs).

Usage:
    python scripts/build_citations.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CHUNKS_PATH = Path("data/chunks/chunks.json")
OUTPUT_PATH = Path("data/chunks/citations.json")

# ---------------------------------------------------------------------------
# Chinese numeral → Arabic integer
# ---------------------------------------------------------------------------

_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn_to_int(text: str) -> int | None:
    """Convert a Chinese numeral string (e.g. '三十六') to int, or None if it's
    already an Arabic digit string."""
    if re.fullmatch(r"\d+", text):
        return int(text)

    result = 0
    unit = 1
    # Process right-to-left
    i = len(text) - 1
    while i >= 0:
        ch = text[i]
        if ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if i == 0:
                # Leading 十 means 1×10
                result += unit
        elif ch in _CN_DIGIT:
            result += _CN_DIGIT[ch] * unit
            unit = 1
        else:
            return None
        i -= 1
    return result if result > 0 else None


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

# Matches 「第X条」where X is Chinese or Arabic numerals
_ART_REF = re.compile(r"第([零一二三四五六七八九十百千\d]+)条")

# Matches 「前款」— intra-article paragraph reference
_PREV_PARA = re.compile(r"前款")


def extract_citations(chunks: list[dict]) -> list[tuple[str, str]]:
    """Return list of (src_chunk_id, target_article_id_str) pairs."""
    # Build set of valid article_ids per document for target validation
    valid_ids: dict[str, set[str]] = {}
    for chunk in chunks:
        doc = chunk["doc_name"]
        valid_ids.setdefault(doc, set()).add(chunk["article_id"])

    citations: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        src_id = chunk["chunk_id"]
        doc = chunk["doc_name"]
        own_article = chunk["article_id"]
        text = chunk["article_text"]

        # 1. Explicit 第X条 references
        for match in _ART_REF.finditer(text):
            raw = match.group(1)
            target_int = cn_to_int(raw)
            if target_int is None:
                continue
            target_str = str(target_int)
            # Skip self-reference
            if target_str == own_article:
                continue
            # Only keep if target exists in same document
            if target_str not in valid_ids.get(doc, set()):
                continue
            edge = (src_id, target_str)
            if edge not in seen:
                seen.add(edge)
                citations.append(edge)

        # 前款/本条 are intra-paragraph references — skipped intentionally.
        # They don't add cross-article graph edges and would cause noisy expansion.

    return citations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CHUNKS_PATH.exists():
        print(f"ERROR: {CHUNKS_PATH} not found", file=sys.stderr)
        sys.exit(1)

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    citations = extract_citations(chunks)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(citations, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(citations)} citation edges written to {OUTPUT_PATH}")

    # Print summary grouped by source doc
    from collections import Counter
    doc_counts: Counter[str] = Counter()
    for src, _ in citations:
        doc_counts[src.split("_")[0]] += 1
    for doc, count in doc_counts.most_common():
        print(f"  {doc}: {count} edges")

    # Show all edges for verification
    print("\nAll edges:")
    for src, tgt in citations:
        print(f"  {src} -> 第{tgt}条")


if __name__ == "__main__":
    main()
