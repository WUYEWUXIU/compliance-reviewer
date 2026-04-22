"""Retrieval evaluation for the hybrid search pipeline.

Run directly:   python tests/test_retrieval.py
Run via pytest: pytest tests/test_retrieval.py -s
"""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, "")

from src.evaluation.retrieval_metrics import (
    RetrievalCaseResult,
    compute_retrieval_metrics,
)
from src.retrieval.hybrid_search import HybridSearch
from tests.golden_set import GOLDEN_SET

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_retrieval() -> None:
    """Run hybrid search against golden cases and compute retrieval metrics."""
    hs = HybridSearch()
    case_results: list[RetrievalCaseResult] = []

    for case in GOLDEN_SET:
        result = hs.search(case["text"])
        retrieved = [r.chunk_id for r in result.top_chunks]
        relevant = set(case.get("relevant_chunk_ids", []))

        case_results.append(
            RetrievalCaseResult(
                case_id=case["id"],
                relevant_chunks=relevant,
                retrieved_chunks=retrieved,
            )
        )

    eval_result = compute_retrieval_metrics(case_results)

    # Console report
    print("\n" + "=" * 50)
    print("检索评估报告 — Hybrid Search")
    print("=" * 50)
    print(f"总样本数:      {eval_result.total_cases}")
    print(f"有标注样本数:  {eval_result.cases_with_relevant}")
    print(f"平均相关chunk: {eval_result.mean_relevant:.1f}")
    print()
    print("指标:")
    for k, recall in sorted(eval_result.recall_at_k.items()):
        print(f"  Recall@{k}:    {recall*100:.1f}%")
    print(f"  MRR:          {eval_result.mrr:.3f}")
    print(f"  HitRate@10:   {eval_result.hit_rate*100:.1f}%")
    print()

    # Per-type breakdown
    print("按违规类型 (有标注的case):")
    from collections import defaultdict
    from src.config.violation_types import VIOLATION_TYPES

    type_results: dict[str, list[RetrievalCaseResult]] = defaultdict(list)
    for cr in case_results:
        case = next(c for c in GOLDEN_SET if c["id"] == cr.case_id)
        for v in case["expected_violations"]:
            type_results[v].append(cr)

    for vid in sorted(type_results):
        results = type_results[vid]
        # Only include cases that have relevant chunks for this type
        valid = [r for r in results if r.relevant_chunks]
        if not valid:
            continue
        sub = compute_retrieval_metrics(valid, k_values=[1, 3, 5, 10])
        name = VIOLATION_TYPES.get(vid, {}).get("name", vid)
        r1 = sub.recall_at_k.get(1, 0) * 100
        r5 = sub.recall_at_k.get(5, 0) * 100
        hr = sub.hit_rate * 100
        print(f"  {vid} {name[:10]:<10}  R@1={r1:5.1f}%  R@5={r5:5.1f}%  HR@10={hr:5.1f}%  n={len(valid)}")
    print()

    # Failure cases
    failures = [cr for cr in case_results if cr.relevant_chunks and not (cr.relevant_chunks & set(cr.retrieved_chunks[:10]))]
    if failures:
        print(f"未命中案例 (HitRate=0): {len(failures)} 个")
        for cr in failures[:10]:
            print(f"  {cr.case_id}: relevant={len(cr.relevant_chunks)}, retrieved={len(cr.retrieved_chunks)}")
    else:
        print("所有有标注case均至少命中一个相关chunk")
    print()

    # Assertions
    assert eval_result.total_cases == 38
    assert eval_result.hit_rate >= 0.0


if __name__ == "__main__":
    test_retrieval()
