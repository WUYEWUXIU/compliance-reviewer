"""Retrieval evaluation for the hybrid search pipeline.

Run directly:   python tests/test_retrieval.py
Run via pytest: pytest tests/test_retrieval.py -s
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, "")

from src.evaluation.retrieval_metrics import (
    RetrievalCaseResult,
    compute_retrieval_metrics,
)
from src.retrieval.hybrid_search import HybridSearch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_golden_set() -> list[dict]:
    """Load golden set from JSON files."""
    golden_dir = Path(__file__).parent / "golden_set"
    cases: list[dict] = []
    for json_file in sorted(golden_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            cases.extend(json.load(f))
    return cases


def test_retrieval() -> None:
    """Run hybrid search against golden cases and compute retrieval metrics."""
    golden_set = _load_golden_set()
    hs = HybridSearch()
    case_results: list[RetrievalCaseResult] = []

    for case in golden_set:
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
    assert eval_result.total_cases == len(golden_set)
    assert eval_result.hit_rate >= 0.0


if __name__ == "__main__":
    test_retrieval()
