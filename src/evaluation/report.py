"""Evaluation report generators (console and JSON)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any, Dict, List, TextIO

from src.evaluation.metrics import CaseResult, EvaluationResult, PerTypeMetrics
from src.config.violation_types import VIOLATION_TYPES


class ConsoleReporter:
    """Print evaluation summary to console with Unicode box-drawing."""

    def report(self, result: EvaluationResult, out: TextIO = sys.stdout) -> None:
        """Emit full console report."""
        self._header(result, out)
        self._overall(result, out)
        self._per_type(result, out)
        self._per_difficulty(result, out)
        self._failures(result, out)
        self._footer(out)

    @staticmethod
    def _header(result: EvaluationResult, out: TextIO) -> None:
        out.write("┌" + "─" * 58 + "┐\n")
        out.write("│" + " 合规评估报告 — Golden Set ".center(56) + "│\n")
        out.write("├" + "─" * 58 + "┤\n")
        out.write(f"│  总样本数: {result.total_cases:>3}                                │\n")
        out.write(f"│  精确匹配正确: {result.correct_cases:>3} ({result.exact_match_accuracy*100:.1f}%)                    │\n")
        out.write("└" + "─" * 58 + "┘\n\n")

    @staticmethod
    def _overall(result: EvaluationResult, out: TextIO) -> None:
        out.write("┌" + "─" * 58 + "┐\n")
        out.write("│" + " 整体指标 ".center(56) + "│\n")
        out.write("├" + "─" * 58 + "┤\n")
        out.write(f"│  合规判断准确率:   {result.compliant_accuracy*100:>6.1f}%                          │\n")
        out.write(f"│  违规类型精确匹配: {result.exact_match_accuracy*100:>6.1f}%                          │\n")
        out.write(f"│  Macro Precision:  {result.macro_precision*100:>6.1f}%                          │\n")
        out.write(f"│  Macro Recall:     {result.macro_recall*100:>6.1f}%                          │\n")
        out.write(f"│  Macro F1:         {result.macro_f1*100:>6.1f}%                          │\n")
        out.write("└" + "─" * 58 + "┘\n\n")

    @staticmethod
    def _per_type(result: EvaluationResult, out: TextIO) -> None:
        out.write("┌" + "─" * 58 + "┐\n")
        out.write("│" + " 按违规类型 (有 support 的类型) ".center(56) + "│\n")
        out.write("├" + "─" * 58 + "┤\n")
        out.write("│  类型   名称              P%    R%    F1%   样本 │\n")
        out.write("├" + "─" * 58 + "┤\n")

        sorted_types = sorted(result.per_type.values(), key=lambda m: m.support, reverse=True)
        for m in sorted_types:
            if m.support == 0:
                continue
            name = VIOLATION_TYPES.get(m.violation_type, {}).get("name", "")[:10]
            out.write(
                f"│  {m.violation_type}  {name:<10}  "
                f"{m.precision*100:>5.1f} {m.recall*100:>5.1f} {m.f1*100:>5.1f}  {m.support:>3}  │\n"
            )
        out.write("└" + "─" * 58 + "┘\n\n")

    @staticmethod
    def _per_difficulty(result: EvaluationResult, out: TextIO) -> None:
        out.write("┌" + "─" * 58 + "┐\n")
        out.write("│" + " 按难度分布 ".center(56) + "│\n")
        out.write("├" + "─" * 58 + "┤\n")

        for difficulty in ["easy", "medium", "hard"]:
            total = sum(1 for cr in result.case_results if cr.difficulty == difficulty)
            if total == 0:
                continue
            correct = sum(
                1
                for cr in result.case_results
                if cr.difficulty == difficulty and cr.expected_violations == cr.predicted_violations
            )
            out.write(f"│  {difficulty:<8}  {correct:>3}/{total:<3}  准确率 {correct/total*100:.1f}%               │\n")
        out.write("└" + "─" * 58 + "┘\n\n")

    @staticmethod
    def _failures(result: EvaluationResult, out: TextIO) -> None:
        failures = result.failures()
        if not failures:
            out.write("✅ 所有 case 均精确匹配！\n")
            return

        out.write("┌" + "─" * 58 + "┐\n")
        out.write(f"│" + f" 失败案例详情 (共 {len(failures)} 个) ".center(56) + "│\n")
        out.write("├" + "─" * 58 + "┤\n")

        for cr in failures[:20]:  # cap at 20 for readability
            miss = sorted(cr.expected_violations - cr.predicted_violations)
            extra = sorted(cr.predicted_violations - cr.expected_violations)
            out.write(f"│  {cr.case_id:<20}  [{cr.difficulty:<6}]           │\n")
            if miss:
                out.write(f"│    漏检: {', '.join(miss):<40}│\n")
            if extra:
                out.write(f"│    误报: {', '.join(extra):<40}│\n")
        out.write("└" + "─" * 58 + "┘\n")

    @staticmethod
    def _footer(out: TextIO) -> None:
        out.write("\n")


class JsonReporter:
    """Serialize evaluation result to JSON."""

    def report(self, result: EvaluationResult) -> Dict[str, Any]:
        """Return a plain dict suitable for json.dumps."""
        per_type_json: Dict[str, Any] = {}
        for vid, m in result.per_type.items():
            per_type_json[vid] = {
                "name": VIOLATION_TYPES.get(vid, {}).get("name", ""),
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "f1": round(m.f1, 4),
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
                "support": m.support,
            }

        failures_json: List[Dict[str, Any]] = []
        for cr in result.failures():
            failures_json.append(
                {
                    "case_id": cr.case_id,
                    "difficulty": cr.difficulty,
                    "expected_compliant": cr.expected_compliant,
                    "predicted_compliant": cr.predicted_compliant,
                    "expected_violations": sorted(cr.expected_violations),
                    "predicted_violations": sorted(cr.predicted_violations),
                    "missed": sorted(cr.expected_violations - cr.predicted_violations),
                    "extra": sorted(cr.predicted_violations - cr.expected_violations),
                    "note": cr.note,
                }
            )

        return {
            "summary": {
                "total_cases": result.total_cases,
                "correct_cases": result.correct_cases,
                "exact_match_accuracy": round(result.exact_match_accuracy, 4),
                "compliant_accuracy": round(result.compliant_accuracy, 4),
                "macro_precision": round(result.macro_precision, 4),
                "macro_recall": round(result.macro_recall, 4),
                "macro_f1": round(result.macro_f1, 4),
            },
            "per_type": per_type_json,
            "per_difficulty": {
                difficulty: {
                    "total": sum(1 for cr in result.case_results if cr.difficulty == difficulty),
                    "correct": sum(
                        1
                        for cr in result.case_results
                        if cr.difficulty == difficulty
                        and cr.expected_violations == cr.predicted_violations
                    ),
                }
                for difficulty in ["easy", "medium", "hard"]
            },
            "failures": failures_json,
        }

    def report_to_string(self, result: EvaluationResult, indent: int = 2) -> str:
        return json.dumps(self.report(result), ensure_ascii=False, indent=indent)
