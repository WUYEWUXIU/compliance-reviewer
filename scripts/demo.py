"""Demo script for the compliance review RAG pipeline.

Usage:
    python scripts/demo.py "营销文案"
    python scripts/demo.py               # runs 5 built-in test cases
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure src/ is on PYTHONPATH when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import CompliancePipeline

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

BUILTIN_CASES = [
    {
        "name": "明确违规（保本保息）",
        "text": "本产品保本保息，年化收益5%稳稳到手",
    },
    {
        "name": "明确违规（绝对化用语）",
        "text": "这是最好的保险产品，业内第一",
    },
    {
        "name": "否定表述（不应判违规）",
        "text": "本产品不保本，收益不确定",
    },
    {
        "name": "合规文案",
        "text": "本产品由XX保险公司承保，犹豫期15天，退保可能会有损失",
    },
    {
        "name": "边界模糊",
        "text": "购买本产品有惊喜",
    },
]


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


def run_case(pipeline: CompliancePipeline, name: str, text: str) -> None:
    """Run a single test case and print formatted output."""
    print("=" * 72)
    print(f"【案例】{name}")
    print(f"【文案】{text}")
    print("-" * 72)

    result = pipeline.review(text)

    print(f"合规结论 : {result['compliant']}")
    print(f"置信度   : {result['confidence']}")
    print(f"行动建议 : {result['action_advice']}")

    if result.get("warning"):
        print(f"警告     : {result['warning']}")

    print(f"违规项   : {len(result['violations'])} 条")
    for v in result["violations"]:
        print(f"  - [{v.get('violation_type_id', '?')}] {v.get('violation_type_name', '?')}")
        print(f"    原因: {v.get('reason', '')}")
        print(f"    建议: {v.get('directional_advice', '')}")

    print(f"合规正向 : {len(result['positive_compliance'])} 条")
    for p in result["positive_compliance"]:
        print(f"  - [{p.get('tag_id', '?')}] {p.get('tag_name', '?')}")
        print(f"    证据: {p.get('evidence', '')}")

    print(f"检索条文 : {len(result['top_chunks'])} 条")
    for idx, chunk in enumerate(result["top_chunks"], start=1):
        print(f"  [{idx}] {chunk['chunk_id']} (score={chunk['score']:.4f})")
        preview = chunk["text"][:120].replace("\n", " ")
        print(f"      {preview}...")

    if result["reference_chunks"]:
        print(f"参考条文 : {len(result['reference_chunks'])} 条")
        for idx, chunk in enumerate(result["reference_chunks"], start=1):
            print(f"  [R{idx}] {chunk['chunk_id']} (score={chunk['score']:.4f})")

    print("\n【完整 JSON 输出】")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()


def main() -> None:
    pipeline = CompliancePipeline()

    if len(sys.argv) >= 2:
        # Single custom case from command line
        text = sys.argv[1]
        run_case(pipeline, "用户输入", text)
    else:
        # Run all built-in cases
        for case in BUILTIN_CASES:
            run_case(pipeline, case["name"], case["text"])


if __name__ == "__main__":
    main()
