"""Core evaluation metrics for compliance review."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class CaseResult:
    """Comparison between prediction and ground truth for a single case."""

    case_id: str
    expected_compliant: str
    predicted_compliant: str


@dataclass
class EvaluationResult:
    """Aggregated evaluation results across all golden cases."""

    case_results: List[CaseResult] = field(default_factory=list)
    compliant_accuracy: float = 0.0

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def correct_cases(self) -> int:
        return sum(
            1 for cr in self.case_results if cr.expected_compliant == cr.predicted_compliant
        )

    def failures(self) -> List[CaseResult]:
        return [
            cr for cr in self.case_results if cr.expected_compliant != cr.predicted_compliant
        ]


def compute_metrics(case_results: List[CaseResult]) -> EvaluationResult:
    """Compute compliant-level accuracy only."""
    result = EvaluationResult(case_results=case_results)

    if not case_results:
        return result

    correct = sum(
        1 for cr in case_results if cr.expected_compliant == cr.predicted_compliant
    )
    result.compliant_accuracy = correct / len(case_results)

    return result
