"""Evaluation script — run golden set and print metrics report.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --golden-set tests/golden_set/short.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging BEFORE any imports that may trigger jieba initialization
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# jieba prints initialization messages directly to stderr AND resets its own
# logger level to DEBUG on import. We redirect stderr during import, then
# clamp the jieba logger level immediately after.
_old_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
from src.evaluation.runner import GoldenSetRunner
from src.pipeline import CompliancePipeline
sys.stderr = _old_stderr
logging.getLogger("jieba").setLevel(logging.WARNING)


def load_golden_set(path: str | None = None) -> list[dict]:
    """Load golden set from JSON file."""
    if path:
        p = Path(path)
        if not p.exists():
            print(f"Error: golden set file not found: {path}")
            sys.exit(1)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    # Default: load all JSON files in tests/golden_set/
    golden_dir = Path("tests/golden_set")
    cases: list[dict] = []
    if golden_dir.exists():
        for json_file in sorted(golden_dir.glob("*.json")):
            with json_file.open("r", encoding="utf-8") as f:
                cases.extend(json.load(f))
    return cases


def _build_markdown_report(
    cases: list[dict],
    result,
    golden_set_path: str | None,
) -> str:
    """Build markdown evaluation report."""
    lines: list[str] = []
    lines.append("# 保险营销内容智能审核 — 效果评估报告\n")
    lines.append(f"- **运行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **样本总数**: {len(cases)} 条")
    if golden_set_path:
        lines.append(f"- **Golden Set**: `{golden_set_path}`")
    lines.append("")

    # 总体指标
    lines.append("## 总体指标\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 合规二分类准确率 | {result.compliant_accuracy:.1%} |")
    lines.append("")

    # 失败案例
    failures = result.failures()
    if failures:
        lines.append(f"## 失败案例（共 {len(failures)} 条）\n")
        for cr in failures:
            lines.append(f"### [{cr.case_id}]\n")
            lines.append(f"- **预期**: {'合规' if cr.expected_compliant == 'yes' else '违规'}")
            lines.append(f"- **预测**: {'合规' if cr.predicted_compliant == 'yes' else '违规' if cr.predicted_compliant == 'no' else '未知'}")
            lines.append("")
    else:
        lines.append("## 失败案例\n")
        lines.append("全部通过！\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compliance review evaluation")
    parser.add_argument(
        "--golden-set",
        dest="golden_set",
        help="Path to golden set JSON file (default: tests/golden_set/short.json)",
    )
    args = parser.parse_args()

    cases = load_golden_set(args.golden_set)

    print(f"\n{'='*60}")
    print(f"  保险营销内容智能审核 — 效果评估报告")
    print(f"{'='*60}")
    print(f"  样本总数 : {len(cases)} 条")
    print(f"{'='*60}\n")

    pipeline = CompliancePipeline()
    runner = GoldenSetRunner(pipeline)
    result = runner.run(cases)

    # ── 总体指标 ──────────────────────────────────────────────
    print("【总体指标】")
    print(f"  合规二分类准确率  : {result.compliant_accuracy:.1%}  ({result.correct_cases}/{result.total_cases})")

    # ── 失败案例 ──────────────────────────────────────────────
    failures = result.failures()
    if failures:
        print(f"\n【失败案例】共 {len(failures)} 条")
        for cr in failures:
            expected_label = "合规" if cr.expected_compliant == "yes" else "违规"
            predicted_label = "合规" if cr.predicted_compliant == "yes" else "违规" if cr.predicted_compliant == "no" else "未知"
            print(f"  [{cr.case_id}] 预期={expected_label} 预测={predicted_label}")
    else:
        print("\n  全部通过！")

    print(f"\n{'='*60}\n")

    # ── 保存 Markdown 报告 ─────────────────────────────────────
    report_dir = Path("tests/evaluation_report")
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"report_{timestamp}.md"
    report_md = _build_markdown_report(cases, result, args.golden_set)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()
