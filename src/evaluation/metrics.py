"""Core evaluation metrics for compliance review."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from src.config.violation_types import VIOLATION_TYPES


@dataclass(frozen=True)
class CaseResult:
    """Comparison between prediction and ground truth for a single case."""

    case_id: str
    expected_compliant: str
    predicted_compliant: str
    expected_violations: Set[str]
    predicted_violations: Set[str]
    difficulty: str
    note: str


@dataclass(frozen=True)
class PerTypeMetrics:
    """Precision / Recall / F1 for a single violation type."""

    violation_type: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvaluationResult:
    """Aggregated evaluation results across all golden cases."""

    case_results: List[CaseResult]
    per_type: Dict[str, PerTypeMetrics] = field(default_factory=dict)
    exact_match_accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    compliant_accuracy: float = 0.0

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def correct_cases(self) -> int:
        return sum(1 for cr in self.case_results if cr.expected_violations == cr.predicted_violations)

    def failures(self) -> List[CaseResult]:
        return [cr for cr in self.case_results if cr.expected_violations != cr.predicted_violations]

    def failures_by_difficulty(self, difficulty: str) -> List[CaseResult]:
        return [cr for cr in self.case_results if cr.difficulty == difficulty and cr.expected_violations != cr.predicted_violations]


def compute_metrics(case_results: List[CaseResult]) -> EvaluationResult:
    """Compute precision, recall, F1 per violation type + aggregate metrics."""
    result = EvaluationResult(case_results=case_results)

    if not case_results:
        return result

    # Exact-match accuracy: violation sets must match exactly
    correct = sum(1 for cr in case_results if cr.expected_violations == cr.predicted_violations)
    result.exact_match_accuracy = correct / len(case_results)

    # Compliant-level accuracy (yes/no/unknown)
    compliant_correct = sum(
        1 for cr in case_results if cr.expected_compliant == cr.predicted_compliant
    )
    result.compliant_accuracy = compliant_correct / len(case_results)

    # Per-type metrics
    vtype_ids = [vid for vid in VIOLATION_TYPES if vid != "V00"]
    per_type_list: List[PerTypeMetrics] = []

    for vid in vtype_ids:
        tp = sum(1 for cr in case_results if vid in cr.predicted_violations and vid in cr.expected_violations)
        fp = sum(1 for cr in case_results if vid in cr.predicted_violations and vid not in cr.expected_violations)
        fn = sum(1 for cr in case_results if vid not in cr.predicted_violations and vid in cr.expected_violations)
        support = sum(1 for cr in case_results if vid in cr.expected_violations)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_type_list.append(
            PerTypeMetrics(
                violation_type=vid,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            )
        )

    result.per_type = {m.violation_type: m for m in per_type_list}

    # Macro averages (over types with at least one expected occurrence)
    supported = [m for m in per_type_list if m.support > 0]
    if supported:
        result.macro_precision = sum(m.precision for m in supported) / len(supported)
        result.macro_recall = sum(m.recall for m in supported) / len(supported)
        result.macro_f1 = sum(m.f1 for m in supported) / len(supported)

    return result
