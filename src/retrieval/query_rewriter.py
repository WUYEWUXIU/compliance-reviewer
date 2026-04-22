"""Query rewriter for compliance review RAG system.

Transforms raw marketing copy into structured retrieval requests.
Two strategies: rule-based (fast path) + LLM semantic (deep path).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import requests

from src.config.settings import BAILIAN_API_KEY, LLM_TIMEOUT, MAX_RETRIES
from src.config.violation_types import VIOLATION_TYPES

logger = logging.getLogger(__name__)

BAILIAN_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# Lightweight model for intent classification — low latency, low cost.
_REWRITE_MODEL = "qwen-turbo"

_LLM_REWRITE_PROMPT = """\
你是一名金融合规意图分析专家。

请将以下营销文案拆解为【检索意图】：
1. 识别文案中**显式或隐式**的违规意图（即使用了委婉、暗示、历史业绩包装等方式）
2. 对每种意图，生成 1-3 条面向监管条文检索的查询语句
3. 标注每条查询的意图类型，使用以下违规类型 ID：
   V01=承诺保本保息, V02=承诺确定收益, V03=绝对化用语, V04=无资质代言,
   V05=缺失风险提示, V06=误导产品比较, V07=隐瞒淡化费用,
   V08=诱导退保转保, V09=伪造备案信息, V10=不当使用客户信息,
   V11=违规承诺增值服务
4. 如果文案包含**否定词**（不、无、非、未等），请判断该否定是否构成"否定式违规"（如"从未亏损"="隐式保本"）或"合规声明"（如"不保本"）

输出严格的 JSON 格式，不要包含任何其他内容：
{{
  "queries": [
    {{"violation_type_id": "V01", "query": "承诺本金不受损失 保本保息", "confidence": 0.9}}
  ]
}}

如果文案无任何违规意图，输出：{{"queries": []}}

待审核文案：{marketing_text}
"""


@dataclass(frozen=True)
class RewriteRequest:
    """Structured retrieval request."""

    violation_type_id: str
    query_text: str
    keywords: list[str]


class QueryRewriter:
    """Rewrite marketing text into compliance retrieval queries.

    Uses rule-based matching as fast path and LLM semantic rewriting as deep
    path. Results are merged and deduplicated.
    """

    _QUERY_TEMPLATES: dict[str, str] = {
        "V01": "承诺保本保息",
        "V02": "承诺确定收益",
        "V03": "使用绝对化用语",
        "V04": "无资质代言",
        "V05": "缺失风险提示",
        "V06": "误导产品比较",
        "V07": "隐瞒淡化费用",
        "V08": "诱导退保转保",
        "V09": "伪造备案信息",
        "V10": "不当使用客户信息",
        "V11": "违规承诺增值服务",
        "V00": "通用合规要求",
    }

    def __init__(self, api_key: str | None = None, use_llm: bool = True) -> None:
        self._api_key = api_key or BAILIAN_API_KEY
        self._use_llm = use_llm and bool(self._api_key)

    def rewrite(self, text: str) -> list[RewriteRequest]:
        """Rewrite raw marketing copy into structured retrieval requests."""
        if not text or not text.strip():
            return []

        normalized = text.strip()

        # Fast path: rule-based keyword matching
        rule_requests = self._rule_rewrite(normalized)

        # Deep path: LLM semantic intent recognition
        llm_requests = self._llm_rewrite(normalized) if self._use_llm else []

        merged = self._merge(rule_requests, llm_requests)

        # Catch-all V00 only when no other intent detected
        if not any(r.violation_type_id == "V00" for r in merged):
            merged.append(
                RewriteRequest(
                    violation_type_id="V00",
                    query_text=normalized,
                    keywords=[],
                )
            )

        return merged

    # ------------------------------------------------------------------
    # Rule-based path
    # ------------------------------------------------------------------

    def _rule_rewrite(self, text: str) -> list[RewriteRequest]:
        requests: list[RewriteRequest] = []
        for vid, vinfo in VIOLATION_TYPES.items():
            if vid == "V00":
                continue
            matched = [kw for kw in vinfo.get("keywords", []) if kw in text]
            if matched:
                requests.append(
                    RewriteRequest(
                        violation_type_id=vid,
                        query_text=self._QUERY_TEMPLATES.get(vid, vinfo["name"]),
                        keywords=matched,
                    )
                )
        return requests

    # ------------------------------------------------------------------
    # LLM semantic path
    # ------------------------------------------------------------------

    def _llm_rewrite(self, text: str) -> list[RewriteRequest]:
        prompt = _LLM_REWRITE_PROMPT.format(marketing_text=text)
        raw = self._call_llm(prompt)
        if not raw:
            return []
        return self._parse_llm_response(raw)

    def _call_llm(self, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": _REWRITE_MODEL,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.0,
        }

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    BAILIAN_CHAT_URL,
                    headers=headers,
                    json=payload,
                    timeout=LLM_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM rewrite attempt %d failed: %s", attempt, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))

        logger.error("LLM rewrite failed after %d attempts: %s", MAX_RETRIES, last_error)
        return ""

    def _parse_llm_response(self, raw: str) -> list[RewriteRequest]:
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                cleaned = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()

            data = json.loads(cleaned)
            results: list[RewriteRequest] = []
            for item in data.get("queries", []):
                vid = item.get("violation_type_id", "")
                query = item.get("query", "").strip()
                confidence = float(item.get("confidence", 0.0))
                if vid and query and confidence >= 0.5:
                    results.append(
                        RewriteRequest(
                            violation_type_id=vid,
                            query_text=query,
                            keywords=[],
                        )
                    )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse LLM rewrite response: %s | raw=%r", exc, raw[:200])
            return []

    # ------------------------------------------------------------------
    # Merge & dedup
    # ------------------------------------------------------------------

    def _merge(
        self,
        rule_requests: list[RewriteRequest],
        llm_requests: list[RewriteRequest],
    ) -> list[RewriteRequest]:
        """Merge rule + LLM results; rule keywords win when vid matches."""
        seen_vids: set[str] = set()
        merged: list[RewriteRequest] = []

        # Rule requests first (preserve keyword evidence)
        for req in rule_requests:
            if req.violation_type_id not in seen_vids:
                seen_vids.add(req.violation_type_id)
                merged.append(req)

        # LLM requests fill in what rules missed
        for req in llm_requests:
            if req.violation_type_id not in seen_vids:
                seen_vids.add(req.violation_type_id)
                merged.append(req)

        return merged
