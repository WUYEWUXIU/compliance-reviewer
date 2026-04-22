"""Validation tests for the LLM review module.

Tests mock reviewer, output parser fallback, and output validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.llm_review.output_parser import parse_llm_output, validate_output
from src.llm_review.reviewer import LLMReviewer, ReviewResult
from src.retrieval.hybrid_search import RerankResult


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
        assert "directional_advice" in v

    print("[PASS] test_mock_violation_detection")


def test_mock_negation_aware() -> None:
    """Test that negated keywords are NOT flagged as violations."""
    reviewer = LLMReviewer(api_key="")
    text = "本产品不保本，收益不确定，投资有风险。"
    chunks = [
        _make_chunk(
            "保险销售行为管理办法_第二十一条_0",
            "保险销售人员不得承诺本金不受损失。",
        ),
    ]
    result = reviewer.review(text, chunks, [])

    assert result.used_mock is True
    # "不保本" should not trigger V01; "不确定" should not trigger V02
    vids = {v["violation_type_id"] for v in result.violations}
    assert "V01" not in vids, f"V01 should not fire for negated text, got {vids}"
    assert "V02" not in vids, f"V02 should not fire for negated text, got {vids}"

    print("[PASS] test_mock_negation_aware")


def test_mock_positive_compliance() -> None:
    """Test that mock reviewer detects positive compliance signals and
    does not flag them as violations."""
    reviewer = LLMReviewer(api_key="")
    # Use exact keywords from COMPLIANCE_TAGS to ensure mock matching works
    text = "本产品由XX保险公司承保，犹豫期为15天，退保损失由投保人承担。"
    chunks = []
    result = reviewer.review(text, chunks, [])

    # C01 keywords ("本产品由", "保险公司承保", "犹豫期", "退保损失") overlap
    # with V05/V08 keywords; mock should resolve overlap in favor of compliance.
    assert result.compliant == "yes", (
        f"Expected compliant=yes for risk-disclosure text, got {result.compliant}. "
        f"Violations: {result.violations}"
    )
    assert len(result.positive_compliance) >= 1

    tag_ids = {p["tag_id"] for p in result.positive_compliance}
    assert "C01" in tag_ids, f"Expected C01 in positive_compliance, got {tag_ids}"

    for p in result.positive_compliance:
        assert "tag_id" in p
        assert "tag_name" in p
        assert "evidence" in p

    print("[PASS] test_mock_positive_compliance")


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
                    "severity": "critical",
                    "directional_advice": "建议删除收益承诺",
                }
            ],
            "positive_compliance": [],
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
            "positive_compliance": [
                {
                    "tag_id": "C01",
                    "tag_name": "含法定风险提示",
                    "evidence": "文案含风险提示",
                }
            ],
        },
        ensure_ascii=False,
    )
    raw = f"```json\n{inner}\n```"
    parsed = parse_llm_output(raw)
    assert parsed["compliant"] == "yes"
    assert len(parsed["positive_compliance"]) == 1

    errors = validate_output(parsed)
    assert errors == [], f"Unexpected validation errors: {errors}"

    print("[PASS] test_output_parser_markdown_fence")


def test_output_parser_fallback() -> None:
    """Test fallback behavior when LLM output is unparseable."""
    raw = "这不是 JSON，是 LLM 的胡言乱语。"
    parsed = parse_llm_output(raw)

    assert parsed["compliant"] == "unknown"
    assert parsed["violations"] == []
    assert parsed["positive_compliance"] == []
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
    bad = {"compliant": "no", "violations": []}  # missing positive_compliance
    errors = validate_output(bad)
    assert any("positive_compliance" in e for e in errors)

    bad2 = {
        "compliant": "maybe",
        "violations": [],
        "positive_compliance": [],
    }
    errors2 = validate_output(bad2)
    assert any("compliant" in e for e in errors2)

    bad3 = {
        "compliant": "no",
        "violations": [
            {"violation_type_id": "V01"}  # missing many fields
        ],
        "positive_compliance": [],
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
    assert "C01" in prompt
    assert "C03" in prompt
    assert "不保本" in prompt  # negation pattern
    assert "JSON" in prompt
    assert "positive_compliance" in prompt

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
    test_mock_negation_aware()
    test_mock_positive_compliance()
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
