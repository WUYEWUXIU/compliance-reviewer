"""Golden set runner — executes pipeline against annotated cases."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from src.evaluation.metrics import CaseResult, EvaluationResult, compute_metrics
from src.pipeline import CompliancePipeline
from tests.golden_set import GOLDEN_SET, GoldenCase

logger = logging.getLogger(__name__)


class GoldenSetRunner:
    """Run the compliance pipeline against all golden-set cases and compute metrics."""

    def __init__(self, pipeline: CompliancePipeline | None = None) -> None:
        self.pipeline = pipeline or CompliancePipeline()

    def run(
        self,
        cases: List[GoldenCase] | None = None,
        progress_every: int = 10,
    ) -> EvaluationResult:
        """Evaluate pipeline against golden cases.

        Args:
            cases: Subset of golden cases to evaluate. Defaults to full GOLDEN_SET.
            progress_every: Log progress every N cases.

        Returns:
            EvaluationResult with per-case and aggregated metrics.
        """
        cases = cases if cases is not None else GOLDEN_SET
        case_results: List[CaseResult] = []

        for idx, case in enumerate(cases, start=1):
            result = self._evaluate_case(case)
            case_results.append(result)
            if idx % progress_every == 0 or idx == len(cases):
                logger.info("Evaluated %d/%d cases", idx, len(cases))

        return compute_metrics(case_results)

    def _evaluate_case(self, case: GoldenCase) -> CaseResult:
        """Run pipeline on a single case and return comparison result."""
        try:
            prediction: Dict[str, Any] = self.pipeline.review(case["text"])
        except Exception as exc:
            logger.error("Pipeline failed for case %s: %s", case["id"], exc)
            prediction = {
                "compliant": "unknown",
                "violations": [],
                "positive_compliance": [],
            }

        predicted_violations: Set[str] = set()
        for v in prediction.get("violations", []):
            vid = v.get("violation_type_id")
            if vid:
                predicted_violations.add(vid)

        return CaseResult(
            case_id=case["id"],
            expected_compliant=case["expected_compliant"],
            predicted_compliant=prediction.get("compliant", "unknown"),
            expected_violations=set(case["expected_violations"]),
            predicted_violations=predicted_violations,
            difficulty=case["difficulty"],
            note=case["note"],
        )
