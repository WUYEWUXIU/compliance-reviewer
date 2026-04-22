"""Tests for multimodal image processing module.

Covers: ImageProcessor (mock mode), _to_image_url, merge_text_and_images,
and pipeline integration with images parameter.
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.multimodal.image_processor import (
    ImageProcessor,
    _to_image_url,
    merge_text_and_images,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def processor_no_key() -> ImageProcessor:
    """ImageProcessor with no API key — triggers mock mode."""
    return ImageProcessor(api_key="")


@pytest.fixture()
def processor_with_key() -> ImageProcessor:
    """ImageProcessor with a fake API key for patching tests."""
    return ImageProcessor(api_key="fake-key-for-test")


@pytest.fixture()
def tiny_png_path(tmp_path: Path) -> Path:
    """Write a 1×1 white PNG to a temp file and return its path."""
    # Minimal valid PNG bytes (1x1 white pixel)
    png_bytes = bytes([
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
        0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc,
        0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
        0x44, 0xae, 0x42, 0x60, 0x82,
    ])
    p = tmp_path / "test.png"
    p.write_bytes(png_bytes)
    return p


# ---------------------------------------------------------------------------
# _to_image_url tests
# ---------------------------------------------------------------------------


def test_to_image_url_http() -> None:
    url = "http://example.com/img.jpg"
    assert _to_image_url(url) == url


def test_to_image_url_https() -> None:
    url = "https://cdn.example.com/banner.png"
    assert _to_image_url(url) == url


def test_to_image_url_file_path(tiny_png_path: Path) -> None:
    result = _to_image_url(str(tiny_png_path))
    assert result.startswith("data:image/png;base64,")
    # Verify the base64 payload decodes to the original bytes
    b64_payload = result.split(",", 1)[1]
    assert base64.b64decode(b64_payload) == tiny_png_path.read_bytes()


def test_to_image_url_invalid_raises() -> None:
    with pytest.raises(ValueError):
        _to_image_url("/nonexistent/path/image.png")


# ---------------------------------------------------------------------------
# merge_text_and_images tests
# ---------------------------------------------------------------------------


def test_merge_both_present() -> None:
    result = merge_text_and_images("营销文案", "[图片1内容]\n保本保息年化8%")
    assert "营销文案" in result
    assert "[图片1内容]" in result
    assert "\n\n" in result


def test_merge_no_image_text() -> None:
    assert merge_text_and_images("营销文案", "") == "营销文案"


def test_merge_no_marketing_text() -> None:
    assert merge_text_and_images("", "[图片1内容]\n文字") == "[图片1内容]\n文字"


def test_merge_both_empty() -> None:
    assert merge_text_and_images("", "") == ""


# ---------------------------------------------------------------------------
# ImageProcessor mock mode (no API key)
# ---------------------------------------------------------------------------


def test_extract_empty_list(processor_no_key: ImageProcessor) -> None:
    assert processor_no_key.extract([]) == ""


def test_extract_single_image_mock(processor_no_key: ImageProcessor) -> None:
    result = processor_no_key.extract(["http://example.com/img.png"])
    assert "[图片1内容]" in result
    assert "mock" in result.lower()


def test_extract_multiple_images_mock(processor_no_key: ImageProcessor) -> None:
    result = processor_no_key.extract([
        "http://example.com/img1.png",
        "http://example.com/img2.png",
    ])
    assert "[图片1内容]" in result
    assert "[图片2内容]" in result


# ---------------------------------------------------------------------------
# ImageProcessor with API key — patch HTTP call
# ---------------------------------------------------------------------------


def _make_vl_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}]
    }
    return resp


def test_extract_calls_vl_api(processor_with_key: ImageProcessor) -> None:
    extracted = "本产品保本保息，年化收益率8%，零风险。"
    with patch("src.multimodal.image_processor.requests.post") as mock_post:
        mock_post.return_value = _make_vl_response(extracted)
        result = processor_with_key.extract(["http://example.com/banner.jpg"])

    assert "[图片1内容]" in result
    assert extracted in result
    mock_post.assert_called_once()


def test_extract_multiple_images_api(processor_with_key: ImageProcessor) -> None:
    texts = ["图片一文字内容", "图片二文字内容"]
    responses = [_make_vl_response(t) for t in texts]
    with patch("src.multimodal.image_processor.requests.post", side_effect=responses):
        result = processor_with_key.extract([
            "http://example.com/img1.jpg",
            "http://example.com/img2.jpg",
        ])

    assert "[图片1内容]" in result
    assert "[图片2内容]" in result
    assert "图片一文字内容" in result
    assert "图片二文字内容" in result


def test_extract_handles_api_failure(processor_with_key: ImageProcessor) -> None:
    """When VL API fails, image entry shows error but does not raise."""
    with patch(
        "src.multimodal.image_processor.requests.post",
        side_effect=Exception("network error"),
    ):
        result = processor_with_key.extract(["http://example.com/img.jpg"])

    assert "[图片1内容]" in result
    assert "提取失败" in result


def test_extract_with_file_path(
    processor_with_key: ImageProcessor, tiny_png_path: Path
) -> None:
    extracted = "测试图片中的文字"
    with patch("src.multimodal.image_processor.requests.post") as mock_post:
        mock_post.return_value = _make_vl_response(extracted)
        result = processor_with_key.extract([str(tiny_png_path)])

    assert extracted in result
    # Verify the payload used a data URI
    call_payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
    content = call_payload["messages"][0]["content"]
    image_part = next(p for p in content if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_review_with_images() -> None:
    """Pipeline.review() should merge image text and pass it through correctly."""
    from src.pipeline import CompliancePipeline

    pipeline = CompliancePipeline.__new__(CompliancePipeline)

    # Stub out all dependencies
    pipeline.image_processor = MagicMock()
    pipeline.image_processor.extract.return_value = "[图片1内容]\n保本保息年化8%"

    pipeline.hybrid_search = MagicMock()
    pipeline.hybrid_search.search.return_value = MagicMock(
        top_chunks=[], reference_chunks=[]
    )

    pipeline.query_rewriter = MagicMock()
    pipeline.llm_reviewer = MagicMock()

    result = pipeline.review("文字文案", images=["http://example.com/img.jpg"])

    pipeline.image_processor.extract.assert_called_once_with(["http://example.com/img.jpg"])

    # hybrid_search should receive merged text
    search_call_arg = pipeline.hybrid_search.search.call_args[0][0]
    assert "文字文案" in search_call_arg
    assert "保本保息年化8%" in search_call_arg

    assert result["compliant"] == "unknown"
    assert result.get("warning") == "未检索到相关条文"


def test_pipeline_review_text_only() -> None:
    """Pipeline.review() without images should behave as before."""
    from src.pipeline import CompliancePipeline

    pipeline = CompliancePipeline.__new__(CompliancePipeline)
    pipeline.image_processor = MagicMock()
    pipeline.image_processor.extract.return_value = ""

    pipeline.hybrid_search = MagicMock()
    pipeline.hybrid_search.search.return_value = MagicMock(
        top_chunks=[], reference_chunks=[]
    )
    pipeline.query_rewriter = MagicMock()
    pipeline.llm_reviewer = MagicMock()

    result = pipeline.review("仅文字文案")

    # image_processor.extract should NOT be called when no images
    pipeline.image_processor.extract.assert_not_called()

    search_call_arg = pipeline.hybrid_search.search.call_args[0][0]
    assert search_call_arg == "仅文字文案"


if __name__ == "__main__":
    import subprocess
    import sys as _sys
    _sys.exit(subprocess.run(
        ["python", "-m", "pytest", __file__, "-v"], check=False
    ).returncode)
