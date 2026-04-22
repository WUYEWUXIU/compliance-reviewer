"""Verification script for HybridSearch."""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.hybrid_search import HybridSearch


def main() -> int:
    print("=" * 60)
    print("HybridSearch Verification")
    print("=" * 60)

    search_engine = HybridSearch()
    query = "保本保息，年化收益5%"
    print(f"\nQuery: {query}\n")

    result = search_engine.search(query)

    # 1. Check return type
    print("[1] Return type check")
    print(f"    result.top_chunks is list: {isinstance(result.top_chunks, list)}")
    print(f"    result.reference_chunks is list: {isinstance(result.reference_chunks, list)}")

    # 2. Check top-5 has content
    print("\n[2] Top-5 content check")
    print(f"    len(top_chunks): {len(result.top_chunks)}")
    if len(result.top_chunks) == 0:
        print("    FAIL: top_chunks is empty")
        return 1

    # 3. Check each result has required fields
    print("\n[3] Field completeness check")
    required_attrs = ("chunk_id", "score", "text")
    all_ok = True
    for i, chunk in enumerate(result.top_chunks):
        missing = [a for a in required_attrs if not hasattr(chunk, a)]
        if missing:
            print(f"    FAIL: top_chunks[{i}] missing {missing}")
            all_ok = False
        else:
            print(
                f"    top_chunks[{i}] chunk_id={chunk.chunk_id}, "
                f"score={chunk.score:.6f}, text_len={len(chunk.text)}"
            )
    if not all_ok:
        return 1

    # 4. Check reference_chunks structure
    print("\n[4] Reference chunks check")
    print(f"    len(reference_chunks): {len(result.reference_chunks)}")
    for i, ref in enumerate(result.reference_chunks):
        print(
            f"    reference_chunks[{i}] chunk_id={ref.chunk_id}, "
            f"score={ref.score:.6f}, text_len={len(ref.text)}"
        )

    print("\n" + "=" * 60)
    print("All checks PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
