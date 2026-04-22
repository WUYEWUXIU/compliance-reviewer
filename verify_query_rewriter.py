"""Verification script for QueryRewriter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retrieval.query_rewriter import QueryRewriter


def main() -> int:
    rewriter = QueryRewriter()

    test_cases = [
        (
            "本产品保本保息，年化收益5%稳稳到手",
            ["V01", "V02"],
        ),
        (
            "本产品不保本，收益不确定",
            ["V01", "V02"],
        ),
        (
            "这是最好的保险产品",
            ["V03"],
        ),
    ]

    passed = 0
    failed = 0

    for text, expected_vids in test_cases:
        print(f"\n输入: {text}")
        results = rewriter.rewrite(text)

        # Show all results
        for r in results:
            print(f"  [{r.violation_type_id}] query_text={r.query_text!r} keywords={r.keywords}")

        # Check expected violation IDs are present
        actual_vids = {r.violation_type_id for r in results if r.violation_type_id != "V00"}
        missing = set(expected_vids) - actual_vids
        extra = actual_vids - set(expected_vids)

        if missing:
            print(f"  [FAIL] 缺少期望的 violation_type_id: {missing}")
            failed += 1
            continue
        if extra:
            print(f"  [FAIL] 出现意外的 violation_type_id: {extra}")
            failed += 1
            continue

        print(f"  [PASS]")
        passed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
