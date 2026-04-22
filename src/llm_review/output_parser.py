"""Output parser for LLM compliance review responses.

Handles JSON extraction, validation, and graceful fallback when the LLM
returns malformed or non-JSON output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Expected top-level keys in the parsed output
_REQUIRED_TOP_KEYS = {"compliant", "violations", "positive_compliance"}

# Required keys inside each violation object
_REQUIRED_VIOLATION_KEYS = {
    "violation_type_id",
    "violation_type_name",
    "article_id",
    "doc_name",
    "article_text",
    "reason",
    "severity",
    "directional_advice",
}

# Required keys inside each positive_compliance object
_REQUIRED_POSITIVE_KEYS = {
    "tag_id",
    "tag_name",
    "evidence",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_llm_output(raw_output: str) -> Dict[str, Any]:
    """Parse raw LLM output into a structured dictionary.

    Attempts JSON extraction with up to 2 retries (re-prompting the LLM is
    the caller's responsibility; this function focuses on extraction logic).
    If parsing ultimately fails, returns a fallback JSON with error metadata.

    Args:
        raw_output: The raw string returned by the LLM.

    Returns:
        A dictionary conforming to the review output schema, or a fallback
        error dictionary.
    """
    if not raw_output or not raw_output.strip():
        return _build_fallback(raw_output, "LLM 返回空输出")

    # Try direct JSON parse first, then regex extraction
    parsed = _try_extract_json(raw_output)
    if parsed is not None:
        return parsed

    # Final fallback
    return _build_fallback(raw_output, "LLM 输出格式异常，JSON 解析失败")


def validate_output(parsed: Dict[str, Any]) -> List[str]:
    """Validate the structural integrity of a parsed LLM output.

    Args:
        parsed: Dictionary returned by parse_llm_output().

    Returns:
        A list of validation error messages. Empty list means valid.
    """
    errors: List[str] = []

    # Top-level keys
    missing_top = _REQUIRED_TOP_KEYS - set(parsed.keys())
    if missing_top:
        errors.append(f"缺少顶层字段: {sorted(missing_top)}")

    # compliant value
    compliant = parsed.get("compliant")
    if compliant not in ("yes", "no", "unknown"):
        errors.append(f"compliant 字段值非法: {compliant!r}")

    # violations array
    violations = parsed.get("violations")
    if not isinstance(violations, list):
        errors.append("violations 必须是数组")
    else:
        for idx, v in enumerate(violations):
            if not isinstance(v, dict):
                errors.append(f"violations[{idx}] 必须是对象")
                continue
            missing_v = _REQUIRED_VIOLATION_KEYS - set(v.keys())
            if missing_v:
                errors.append(f"violations[{idx}] 缺少字段: {sorted(missing_v)}")
            # severity enum check
            sev = v.get("severity")
            if sev not in ("critical", "warning", "info", None):
                errors.append(
                    f"violations[{idx}].severity 值非法: {sev!r}"
                )

    # positive_compliance array
    positives = parsed.get("positive_compliance")
    if not isinstance(positives, list):
        errors.append("positive_compliance 必须是数组")
    else:
        for idx, p in enumerate(positives):
            if not isinstance(p, dict):
                errors.append(f"positive_compliance[{idx}] 必须是对象")
                continue
            missing_p = _REQUIRED_POSITIVE_KEYS - set(p.keys())
            if missing_p:
                errors.append(
                    f"positive_compliance[{idx}] 缺少字段: {sorted(missing_p)}"
                )

    return errors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_extract_json(raw_output: str) -> Dict[str, Any] | None:
    """Attempt to extract and parse JSON from raw LLM output.

    Tries, in order:
    1. Direct json.loads on the stripped string.
    2. Extract the first JSON object/array via regex.
    3. Extract from markdown code fences.

    Returns:
        Parsed dict if successful, None otherwise.
    """
    text = raw_output.strip()

    # 1. Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Markdown code block
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 3. First JSON object/array via regex (aggressive)
    json_pattern = re.compile(r"(\{[\s\S]*\})")
    match = json_pattern.search(text)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def _build_fallback(raw_output: str, reason: str) -> Dict[str, Any]:
    """Construct a fallback response when parsing fails."""
    return {
        "compliant": "unknown",
        "violations": [],
        "positive_compliance": [],
        "error": "llm_output_unparseable",
        "raw_output": raw_output,
        "confidence": 0,
        "reason": reason,
    }
