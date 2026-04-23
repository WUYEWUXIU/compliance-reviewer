"""Evaluation report generators (console and JSON)."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, TextIO

from src.evaluation.metrics import CaseResult, EvaluationResult


class ConsoleReporter:
    """Print evaluation summary to console with Unicode box-drawing."""

    def report(self, result: EvaluationResult, out: TextIO = sys.stdout) -> None:
        """Emit full console report."""
        self._header(result, out)
        self._overall(result, out)
        self._failures(result, out)
        self._footer(out)

    @staticmethod
    def _header(result: EvaluationResult, out: TextIO) -> None:
        out.write("┌" + "─" * 58 + "┐\n")
        out.write("│" + " 合规评估报告 — Golden Set ".center(56) + "│\n")
        out.write("├" + "─" * 58 + "┤\n")
        out.write(f"│  总样本数: {result.total_cases:>3}                                │\n")
        out.write("└" + "─" * 58 + "┘\n\n")

    @staticmethod
    def _overall(result: EvaluationResult, out: TextIO) -> None:
        out.write("┌" + "─" * 58 + "┐\n")
        out.write("│" + " 整体指标 ".center(56) + "│\n")
        out.write("├" + "─" * 58 + "┤\n")
        out.write(f"│  合规判断准确率:   {result.compliant_accuracy*100:>6.1f}%                          │\n")
        out.write("└" + "─" * 58 + "┘\n\n")

    @staticmethod
    def _failures(result: EvaluationResult, out: TextIO) -> None:
        failures = result.failures()
        if not failures:
            out.write("✅ 所有 case 均判断正确！\n")
            return

        out.write("┌" + "─" * 58 + "┐\n")
        out.write(f"│" + f" 失败案例详情 (共 {len(failures)} 个) ".center(56) + "│\n")
        out.write("├" + "─" * 58 + "┤\n")

        for cr in failures[:20]:  # cap at 20 for readability
            expected = "合规" if cr.expected_compliant == "yes" else "违规"
            predicted = "合规" if cr.predicted_compliant == "yes" else "违规" if cr.predicted_compliant == "no" else "未知"
            out.write(f"│  {cr.case_id:<20}  预期={expected} 预测={predicted:<10}│\n")
        out.write("└" + "─" * 58 + "┘\n")

    @staticmethod
    def _footer(out: TextIO) -> None:
        out.write("\n")


class JsonReporter:
    """Serialize evaluation result to JSON."""

    def report(self, result: EvaluationResult) -> Dict[str, Any]:
        """Return a plain dict suitable for json.dumps."""
        failures_json: List[Dict[str, Any]] = []
        for cr in result.failures():
            failures_json.append(
                {
                    "case_id": cr.case_id,
                    "expected_compliant": cr.expected_compliant,
                    "predicted_compliant": cr.predicted_compliant,
                }
            )

        return {
            "summary": {
                "total_cases": result.total_cases,
                "correct_cases": result.correct_cases,
                "compliant_accuracy": round(result.compliant_accuracy, 4),
            },
            "failures": failures_json,
        }

    def report_to_string(self, result: EvaluationResult, indent: int = 2) -> str:
        return json.dumps(self.report(result), ensure_ascii=False, indent=indent)
