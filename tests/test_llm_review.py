"""Validation tests for the LLM review module.

Tests mock reviewer, output parser fallback, and output validation.
"""

from __future__ import annotations
from src.retrieval.hybrid_search import RerankResult
from src.llm_review.reviewer import LLMReviewer, ReviewResult
from src.llm_review.output_parser import parse_llm_output, validate_output

import json
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _make_chunk(chunk_id: str, text: str, score: float = 0.9) -> RerankResult:
    return RerankResult(chunk_id=chunk_id, score=score, text=text)


def test_mock_violation_detection() -> None:
    """Test that mock reviewer detects a clear violation."""
    reviewer = LLMReviewer(api_key="")
    text = "本产品保本保息，年化收益高达5%，稳赚不赔！"
    chunks = [
        _make_chunk(
            "保险销售行为管理办法_第二十一条_0",
            "保险销售人员不得承诺本金不受损失或者承诺确定收益。",
        ),
        _make_chunk(
            "金融网销管理办法_第八条_0",
            "金融产品营销宣传不得含有保本保息、稳赚等表述。",
        ),
    ]
    result = reviewer.review(text, chunks, [])

    assert result.used_mock is True
    assert result.compliant == "no"
    assert len(result.violations) >= 2

    vids = {v["violation_type_id"] for v in result.violations}
    assert "V01" in vids, f"Expected V01 in violations, got {vids}"
    assert "V02" in vids, f"Expected V02 in violations, got {vids}"

    # Check required fields
    for v in result.violations:
        assert "violation_type_id" in v
        assert "violation_type_name" in v
        assert "article_id" in v
        assert "doc_name" in v
        assert "article_text" in v
        assert "reason" in v
        assert "severity" in v

    print("[PASS] test_mock_violation_detection")


def test_output_parser_valid_json() -> None:
    """Test parsing a well-formed LLM JSON output."""
    raw = json.dumps(
        {
            "compliant": "no",
            "violations": [
                {
                    "violation_type_id": "V02",
                    "violation_type_name": "承诺确定收益",
                    "article_id": "第二十一条",
                    "doc_name": "保险销售行为管理办法",
                    "article_text": "不得承诺确定收益",
                    "reason": "文案承诺年化5%",
                    "severity": "critical"
                }
            ],
        },
        ensure_ascii=False,
    )
    parsed = parse_llm_output(raw)
    assert parsed["compliant"] == "no"
    assert len(parsed["violations"]) == 1
    assert parsed["violations"][0]["violation_type_id"] == "V02"

    errors = validate_output(parsed)
    assert errors == [], f"Unexpected validation errors: {errors}"

    print("[PASS] test_output_parser_valid_json")


def test_output_parser_markdown_fence() -> None:
    """Test parsing JSON wrapped in markdown code fences."""
    inner = json.dumps(
        {
            "compliant": "yes",
            "violations": [],
        },
        ensure_ascii=False,
    )
    raw = f"```json\n{inner}\n```"
    parsed = parse_llm_output(raw)
    assert parsed["compliant"] == "yes"
    assert parsed["violations"] == []

    errors = validate_output(parsed)
    assert errors == [], f"Unexpected validation errors: {errors}"

    print("[PASS] test_output_parser_markdown_fence")


def test_output_parser_fallback() -> None:
    """Test fallback behavior when LLM output is unparseable."""
    raw = "这不是 JSON，是 LLM 的胡言乱语。"
    parsed = parse_llm_output(raw)

    assert parsed["compliant"] == "unknown"
    assert parsed["violations"] == []
    assert parsed["error"] == "llm_output_unparseable"
    assert "raw_output" in parsed

    print("[PASS] test_output_parser_fallback")


def test_output_parser_empty() -> None:
    """Test fallback behavior for empty output."""
    parsed = parse_llm_output("")
    assert parsed["compliant"] == "unknown"
    assert parsed.get("error") is not None

    print("[PASS] test_output_parser_empty")


def test_validate_output_missing_fields() -> None:
    """Test validation catches missing required fields."""
    bad = {"compliant": "no"}  # missing violations
    errors = validate_output(bad)
    assert any("violations" in e for e in errors)

    bad2 = {
        "compliant": "maybe",
        "violations": [],
    }
    errors2 = validate_output(bad2)
    assert any("compliant" in e for e in errors2)

    bad3 = {
        "compliant": "no",
        "violations": [
            {"violation_type_id": "V01"}  # missing many fields
        ],
    }
    errors3 = validate_output(bad3)
    assert len(errors3) > 0

    print("[PASS] test_validate_output_missing_fields")


def test_system_prompt_contains_catalogue() -> None:
    """Test that system prompt embeds the violation catalogue."""
    from src.llm_review.prompt_builder import build_system_prompt

    prompt = build_system_prompt()
    assert "V01" in prompt
    assert "V11" in prompt
    assert "JSON" in prompt

    print("[PASS] test_system_prompt_contains_catalogue")


def test_user_prompt_format() -> None:
    """Test that user prompt assembles chunks correctly."""
    from src.llm_review.prompt_builder import build_user_prompt

    chunks = [
        _make_chunk("docA_第1条_0", "条文A内容", 0.95),
        _make_chunk("docB_第2条_0", "条文B内容", 0.88),
    ]
    refs = [_make_chunk("docA_第3条_0", "参考条文内容", 0.5)]
    prompt = build_user_prompt("测试文案", chunks, refs)

    assert "测试文案" in prompt
    assert "条文A内容" in prompt
    assert "条文B内容" in prompt
    assert "参考条文内容" in prompt
    assert "docA" in prompt
    assert "docB" in prompt
    assert "[1]" in prompt
    assert "[R1]" in prompt

    print("[PASS] test_user_prompt_format")


if __name__ == "__main__":
    print("=" * 60)
    print("Running LLM Review Module Validation Tests")
    print("=" * 60)

    test_mock_violation_detection()
    test_output_parser_valid_json()
    test_output_parser_markdown_fence()
    test_output_parser_fallback()
    test_output_parser_empty()
    test_validate_output_missing_fields()
    test_system_prompt_contains_catalogue()
    test_user_prompt_format()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
