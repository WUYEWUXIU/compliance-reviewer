"""Image processor for multimodal compliance review.

Accepts image inputs (file path, URL, or base64 string) and uses Bailian's
vision-language model to extract text content for downstream compliance review.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import List, Optional

import requests

from src.config.settings import BAILIAN_API_KEY, BAILIAN_VL_MODEL, LLM_TIMEOUT, MAX_RETRIES

logger = logging.getLogger(__name__)

BAILIAN_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

_EXTRACT_PROMPT = (
    "请提取并转录图片中所有可见的文字内容，保持原有格式。"
    "如果图片包含营销宣传信息，请特别注意提取产品名称、收益描述、风险提示等关键信息。"
    "仅输出提取的文字，不需要分析或评论。"
)

_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def _to_image_url(image_input: str) -> str:
    """Convert image_input to a data URI or pass through HTTP URL."""
    if _URL_PATTERN.match(image_input):
        return image_input

    # File path
    path = Path(image_input)
    if path.exists():
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # Assume raw base64 (already encoded)
    if len(image_input) > 260 and not path.suffix:
        return f"data:image/jpeg;base64,{image_input}"

    raise ValueError(f"Cannot resolve image input: {image_input[:80]}...")


def _extract_text_from_image(image_url: str, api_key: str, model: str) -> str:
    """Call Bailian VL model to extract text from a single image."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": _EXTRACT_PROMPT},
                ],
            }
        ],
        "temperature": 0.0,
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                BAILIAN_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            choices = resp.json().get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                return content.strip()
            return ""
        except Exception as exc:
            last_err = exc
            logger.warning("VL API attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"VL API failed after {MAX_RETRIES} retries: {last_err}")


class ImageProcessor:
    """Extract text content from images using Bailian VL model.

    Falls back to a placeholder when no API key is configured, enabling
    unit tests without real credentials.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else BAILIAN_API_KEY
        self.model = model or BAILIAN_VL_MODEL

    def extract(self, images: List[str]) -> str:
        """Extract and concatenate text from all provided images.

        Args:
            images: List of image inputs — file paths, HTTP URLs, or base64 strings.

        Returns:
            Concatenated extracted text, prefixed with '[图片内容]' marker per image.
        """
        if not images:
            return ""

        if not self.api_key:
            logger.warning("BAILIAN_API_KEY not set; image extraction skipped (mock mode).")
            return self._mock_extract(images)

        parts: List[str] = []
        for idx, img in enumerate(images, start=1):
            try:
                image_url = _to_image_url(img)
                text = _extract_text_from_image(image_url, self.api_key, self.model)
                if text:
                    parts.append(f"[图片{idx}内容]\n{text}")
                else:
                    logger.warning("VL model returned empty text for image %d", idx)
            except Exception as exc:
                logger.error("Failed to process image %d: %s", idx, exc)
                parts.append(f"[图片{idx}内容]\n[提取失败: {exc}]")

        return "\n\n".join(parts)

    @staticmethod
    def _mock_extract(images: List[str]) -> str:
        """Return placeholder text for each image (used when API key absent)."""
        return "\n\n".join(
            f"[图片{i}内容]\n[mock: 图片文字提取占位符]"
            for i in range(1, len(images) + 1)
        )


def merge_text_and_images(marketing_text: str, image_extracted: str) -> str:
    """Merge marketing text with image-extracted content into a single review input."""
    if not image_extracted:
        return marketing_text
    if not marketing_text.strip():
        return image_extracted
    return f"{marketing_text}\n\n{image_extracted}"
