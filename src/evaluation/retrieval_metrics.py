"""Retrieval evaluation metrics: Recall@K, MRR, HitRate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass(frozen=True)
class RetrievalCaseResult:
    """Retrieval result for a single case."""

    case_id: str
    relevant_chunks: Set[str]
    retrieved_chunks: List[str]  # ordered by retrieval rank


@dataclass
class RetrievalEvaluationResult:
    """Aggregated retrieval metrics."""

    case_results: List[RetrievalCaseResult]
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate: float = 0.0
    mean_relevant: float = 0.0

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def cases_with_relevant(self) -> int:
        return sum(1 for cr in self.case_results if cr.relevant_chunks)


def compute_retrieval_metrics(
    case_results: List[RetrievalCaseResult],
    k_values: List[int] | None = None,
) -> RetrievalEvaluationResult:
    """Compute Recall@K, MRR, and HitRate.

    Only cases with at least one relevant chunk contribute to metrics.
    Cases with no relevant chunks are counted in total but skipped in
    denominator to avoid skewing metrics.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    result = RetrievalEvaluationResult(case_results=case_results)

    valid_cases = [cr for cr in case_results if cr.relevant_chunks]
    if not valid_cases:
        return result

    result.mean_relevant = sum(len(cr.relevant_chunks) for cr in valid_cases) / len(valid_cases)

    # Recall@K
    for k in k_values:
        recall_sum = 0.0
        for cr in valid_cases:
            retrieved_set = set(cr.retrieved_chunks[:k])
            hits = len(cr.relevant_chunks & retrieved_set)
            recall_sum += hits / len(cr.relevant_chunks)
        result.recall_at_k[k] = recall_sum / len(valid_cases)

    # MRR (Mean Reciprocal Rank)
    mrr_sum = 0.0
    for cr in valid_cases:
        rr = 0.0
        for rank, chunk_id in enumerate(cr.retrieved_chunks, start=1):
            if chunk_id in cr.relevant_chunks:
                rr = 1.0 / rank
                break
        mrr_sum += rr
    result.mrr = mrr_sum / len(valid_cases)

    # HitRate (at least one relevant chunk in top-10)
    hit_sum = 0
    for cr in valid_cases:
        retrieved_set = set(cr.retrieved_chunks[:10])
        if cr.relevant_chunks & retrieved_set:
            hit_sum += 1
    result.hit_rate = hit_sum / len(valid_cases)

    return result
