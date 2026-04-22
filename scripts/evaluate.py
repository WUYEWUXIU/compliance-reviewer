"""Evaluation script — run golden set and print metrics report.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --subset easy      # easy / medium / hard
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.runner import GoldenSetRunner
from src.pipeline import CompliancePipeline
from tests.golden_set import GOLDEN_SET

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compliance review evaluation")
    parser.add_argument(
        "--subset",
        choices=["easy", "medium", "hard"],
        help="Filter golden set by difficulty",
    )
    args = parser.parse_args()

    cases = GOLDEN_SET
    if args.subset:
        cases = [c for c in GOLDEN_SET if c["difficulty"] == args.subset]

    print(f"\n{'='*60}")
    print(f"  保险营销内容智能审核 — 效果评估报告")
    print(f"{'='*60}")
    print(f"  样本总数 : {len(cases)} 条" + (f"（{args.subset}）" if args.subset else ""))
    print(f"{'='*60}\n")

    pipeline = CompliancePipeline()
    runner = GoldenSetRunner(pipeline)
    result = runner.run(cases)

    # ── 总体指标 ──────────────────────────────────────────────
    print("【总体指标】")
    print(f"  合规二分类准确率  : {result.compliant_accuracy:.1%}  ({int(result.compliant_accuracy * len(cases))}/{len(cases)})")
    print(f"  违规类型精确匹配  : {result.exact_match_accuracy:.1%}  ({result.correct_cases}/{result.total_cases})")
    print(f"  宏平均 Precision  : {result.macro_precision:.1%}")
    print(f"  宏平均 Recall     : {result.macro_recall:.1%}")
    print(f"  宏平均 F1         : {result.macro_f1:.1%}")

    # ── 各违规类型指标 ─────────────────────────────────────────
    print("\n【各违规类型指标】")
    print(f"  {'ID':<5} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*38}")
    for vid, m in sorted(result.per_type.items()):
        if m.support == 0:
            continue
        print(f"  {vid:<5} {m.precision:>6.1%} {m.recall:>6.1%} {m.f1:>6.1%} {m.support:>8}")

    # ── 失败案例 ──────────────────────────────────────────────
    failures = result.failures()
    if failures:
        print(f"\n【失败案例】共 {len(failures)} 条")
        for cr in failures[:10]:
            print(f"  [{cr.case_id}] ({cr.difficulty})")
            print(f"    预期: {cr.expected_violations or '合规'}")
            print(f"    预测: {cr.predicted_violations or '合规'}")
            print(f"    备注: {cr.note}")
    else:
        print("\n  全部通过！")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
