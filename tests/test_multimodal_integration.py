"""Integration test: real image → VL extraction → compliance pipeline.

Requires DASHSCOPE_API_KEY / BAILIAN_API_KEY in environment.
Skip automatically when no API key is present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HAS_API_KEY = bool(
    os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
)
pytestmark = pytest.mark.skipif(
    not HAS_API_KEY, reason="BAILIAN_API_KEY / DASHSCOPE_API_KEY not set"
)

IMAGE_PATH = Path(__file__).parent / "pics" / "image.png"


@pytest.fixture(scope="module")
def extracted_text() -> str:
    from src.multimodal.image_processor import ImageProcessor

    proc = ImageProcessor()
    return proc.extract([str(IMAGE_PATH)])


# ---------------------------------------------------------------------------
# VL extraction tests
# ---------------------------------------------------------------------------


def test_image_exists() -> None:
    assert IMAGE_PATH.exists(), f"Test image not found: {IMAGE_PATH}"


def test_vl_extraction_returns_text(extracted_text: str) -> None:
    assert "[图片1内容]" in extracted_text
    assert len(extracted_text) > 20


def test_vl_extraction_captures_key_claims(extracted_text: str) -> None:
    """VL model must extract the core violation-triggering phrases."""
    assert "保本保息" in extracted_text, "Expected '保本保息' in extracted text"
    assert "8%" in extracted_text, "Expected '8%' in extracted text"


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_result():
    from src.pipeline import CompliancePipeline

    pipeline = CompliancePipeline()
    return pipeline.review(marketing_text="", images=[str(IMAGE_PATH)])


def test_pipeline_result_non_compliant(pipeline_result: dict) -> None:
    assert pipeline_result["compliant"] == "no", (
        f"Expected 'no', got '{pipeline_result['compliant']}'"
    )


def test_pipeline_detects_v01(pipeline_result: dict) -> None:
    """图片中'保本保息'应触发 V01 承诺本金不受损失。"""
    vids = {v["violation_type_id"] for v in pipeline_result["violations"]}
    assert "V01" in vids, f"V01 not detected; got: {vids}"


def test_pipeline_detects_v02(pipeline_result: dict) -> None:
    """图片中'年化8%'应触发 V02 承诺确定收益。"""
    vids = {v["violation_type_id"] for v in pipeline_result["violations"]}
    assert "V02" in vids, f"V02 not detected; got: {vids}"


def test_pipeline_violations_have_required_fields(pipeline_result: dict) -> None:
    required = {
        "violation_type_id", "violation_type_name",
        "article_id", "doc_name", "reason", "severity",
    }
    for v in pipeline_result["violations"]:
        missing = required - v.keys()
        assert not missing, f"Violation missing fields {missing}: {v}"


def test_pipeline_retrieved_chunks(pipeline_result: dict) -> None:
    assert len(pipeline_result["top_chunks"]) > 0, "No chunks retrieved"


def test_pipeline_not_mock(pipeline_result: dict) -> None:
    assert pipeline_result.get("warning") != "[LLM降级为mock审核]", (
        "Pipeline fell back to mock — check API key and model config"
    )
