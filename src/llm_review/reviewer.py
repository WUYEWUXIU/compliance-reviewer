"""LLM reviewer for compliance review RAG system.

Orchestrates prompt building, LLM API calling (Bailian), output parsing,
and mock review fallback for development/testing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Set

import requests

from src.config.settings import BAILIAN_API_KEY, BAILIAN_LLM_MODEL, LLM_TIMEOUT, MAX_RETRIES
from src.config.violation_types import COMPLIANCE_TAGS, NEGATION_PATTERNS, VIOLATION_TYPES
from src.llm_review.output_parser import parse_llm_output, validate_output
from src.llm_review.prompt_builder import build_system_prompt, build_user_prompt
from src.retrieval.hybrid_search import RerankResult

logger = logging.getLogger(__name__)

BAILIAN_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewResult:
    """Structured output of a compliance review."""

    compliant: str  # "yes", "no", or "unknown"
    violations: List[Dict[str, Any]]
    positive_compliance: List[Dict[str, Any]]
    confidence: float
    raw_output: str
    validation_errors: List[str]
    used_mock: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class LLMReviewer:
    """Compliance review orchestrator using Bailian LLM."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = BAILIAN_API_KEY if api_key is None else api_key
        self.model = model or BAILIAN_LLM_MODEL

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def review(
        self,
        marketing_text: str,
        top_chunks: List[RerankResult],
        reference_chunks: List[RerankResult],
        confidence_score: float,
    ) -> ReviewResult:
        """Run compliance review on marketing text.

        Args:
            marketing_text: The marketing copy to review.
            top_chunks: Top-5 reranked chunks.
            reference_chunks: Citation-graph expansion chunks.
            confidence_score: Retrieval confidence score (0-1).

        Returns:
            ReviewResult with structured conclusion.
        """
        if not self.api_key:
            logger.warning("BAILIAN_API_KEY not configured; using mock review.")
            return self._mock_review(
                marketing_text, top_chunks, reference_chunks, confidence_score
            )

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(marketing_text, top_chunks, reference_chunks)

        raw_output = self._call_llm_with_retry(system_prompt, user_prompt)
        parsed = parse_llm_output(raw_output)
        validation_errors = validate_output(parsed)

        if validation_errors:
            logger.warning("LLM output validation errors: %s", validation_errors)

        return ReviewResult(
            compliant=parsed.get("compliant", "unknown"),
            violations=parsed.get("violations", []),
            positive_compliance=parsed.get("positive_compliance", []),
            confidence=confidence_score,
            raw_output=raw_output,
            validation_errors=validation_errors,
            used_mock=False,
        )

    # ------------------------------------------------------------------
    # LLM API calling with retry
    # ------------------------------------------------------------------

    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Call Bailian chat API with exponential backoff.

        Args:
            system_prompt: The system prompt.
            user_prompt: The user prompt.

        Returns:
            Raw LLM output string.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    BAILIAN_CHAT_URL,
                    headers=headers,
                    json=payload,
                    timeout=LLM_TIMEOUT,
                )
                response.raise_for_status()
                result = response.json()

                # OpenAI-compatible response format
                choices = result.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if content:
                        return content.strip()

                logger.warning("LLM response missing content: %s", result)
                return json.dumps(result)

            except requests.exceptions.Timeout:
                last_error = requests.exceptions.Timeout("LLM API timeout")
                logger.warning("LLM API attempt %d/%d timed out", attempt, MAX_RETRIES)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                logger.warning(
                    "LLM API attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc
                )

            # Exponential backoff: 1s, 2s, 4s
            if attempt < MAX_RETRIES:
                backoff = 2 ** (attempt - 1)
                logger.info("Retrying in %d seconds...", backoff)
                time.sleep(backoff)

        logger.error("LLM API failed after %d retries: %s", MAX_RETRIES, last_error)
        return json.dumps(
            {
                "compliant": "unknown",
                "violations": [],
                "positive_compliance": [],
                "error": "llm_api_failed",
                "reason": f"API failed after {MAX_RETRIES} retries: {last_error}",
            }
        )

    # ------------------------------------------------------------------
    # Mock review (for dev/test when API key is absent)
    # ------------------------------------------------------------------

    def _mock_review(
        self,
        marketing_text: str,
        top_chunks: List[RerankResult],
        reference_chunks: List[RerankResult],
        confidence_score: float,
    ) -> ReviewResult:
        """Generate a rule-based mock review result.

        Uses keyword matching against VIOLATION_TYPES and COMPLIANCE_TAGS,
        with negation-aware logic and overlap resolution.
        """
        text = marketing_text.strip()
        violations: List[Dict[str, Any]] = []
        positive_compliance: List[Dict[str, Any]] = []

        # --- Negation detection ---
        negated = any(pat in text for pat in NEGATION_PATTERNS)

        # --- Collect all positive keywords and their substrings ---
        all_positive_keywords: Set[str] = set()
        for cinfo in COMPLIANCE_TAGS.values():
            for kw in cinfo.get("keywords", []):
                if kw in text:
                    all_positive_keywords.add(kw)

        # --- Violation detection (keyword matching) ---
        for vid, vinfo in VIOLATION_TYPES.items():
            if vid == "V00":
                continue
            keywords = vinfo.get("keywords", [])
            matched = [kw for kw in keywords if kw in text]
            if not matched:
                continue

            # If negated and the matched keyword is commonly negated, skip
            if negated and _is_likely_negated(text, matched):
                continue

            # Overlap resolution: if a violation keyword is a substring of
            # any positive compliance keyword found in text, skip it (the
            # positive context takes precedence in mock mode).
            filtered_matched = []
            for kw in matched:
                if any(kw in pk and kw != pk for pk in all_positive_keywords):
                    # kw is a strict substring of a positive keyword
                    continue
                if kw in all_positive_keywords:
                    # kw exactly matches a positive keyword
                    continue
                filtered_matched.append(kw)

            if not filtered_matched:
                continue
            matched = filtered_matched

            # Pick the best chunk as evidence
            best_chunk = top_chunks[0] if top_chunks else None
            if best_chunk is None and reference_chunks:
                best_chunk = reference_chunks[0]

            doc_name = "未知文档"
            article_id = "未知条文"
            article_text = ""
            if best_chunk:
                parts = best_chunk.chunk_id.split("_")
                if len(parts) >= 2:
                    doc_name, article_id = parts[0], parts[1]
                article_text = best_chunk.text.strip()[:200]

            violations.append(
                {
                    "violation_type_id": vid,
                    "violation_type_name": vinfo["name"],
                    "article_id": article_id,
                    "doc_name": doc_name,
                    "article_text": article_text,
                    "reason": f"文案中包含关键词：{', '.join(matched)}",
                    "severity": vinfo.get("severity", "warning"),
                    "directional_advice": f"建议删除或修改包含{matched[0]}的表述，确保不违反{vinfo['name']}相关规定。",
                }
            )

        # --- Positive compliance detection ---
        for cid, cinfo in COMPLIANCE_TAGS.items():
            keywords = cinfo.get("keywords", [])
            matched = [kw for kw in keywords if kw in text]
            if matched:
                positive_compliance.append(
                    {
                        "tag_id": cid,
                        "tag_name": cinfo["name"],
                        "evidence": f"文案中包含：{', '.join(matched)}",
                    }
                )

        compliant = "no" if violations else "yes"

        # Build a synthetic raw_output for traceability
        raw_output = json.dumps(
            {
                "compliant": compliant,
                "violations": violations,
                "positive_compliance": positive_compliance,
                "note": "mock_review",
            },
            ensure_ascii=False,
            indent=2,
        )

        return ReviewResult(
            compliant=compliant,
            violations=violations,
            positive_compliance=positive_compliance,
            confidence=confidence_score,
            raw_output=raw_output,
            validation_errors=[],
            used_mock=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_likely_negated(text: str, matched_keywords: List[str]) -> bool:
    """Heuristic: check if matched keywords appear in a negated context.

    Looks for negation prefixes within 6 characters before the keyword.
    """
    negation_prefixes = ["不", "非", "无", "没有", "不存在", "并非", "不等于", "不涉及", "不含"]
    for kw in matched_keywords:
        idx = text.find(kw)
        if idx > 0:
            window = text[max(0, idx - 6) : idx]
            for prefix in negation_prefixes:
                if window.endswith(prefix):
                    return True
    return False
