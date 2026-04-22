"""Query rewriter for compliance review RAG system.

Transforms raw marketing copy into structured retrieval requests using
rule-based matching against violation type keywords.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config.violation_types import VIOLATION_TYPES


@dataclass(frozen=True)
class RewriteRequest:
    """Structured retrieval request."""

    violation_type_id: str
    query_text: str
    keywords: list[str]


class QueryRewriter:
    """Rewrite marketing text into compliance retrieval queries."""

    # Mapping from violation type to concise regulatory-style query text (<=10 chars).
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

    def rewrite(self, text: str) -> list[RewriteRequest]:
        """Rewrite raw marketing copy into structured retrieval requests.

        Args:
            text: Original marketing copy.

        Returns:
            List of rewrite requests (deduplicated by query_text).
        """
        if not text or not text.strip():
            return []

        normalized = text.strip()
        requests: list[RewriteRequest] = []
        seen_query_texts: set[str] = set()

        # 1. Rule-based matching against violation types
        for vid, vinfo in VIOLATION_TYPES.items():
            if vid == "V00":
                continue
            matched_keywords = self._match_keywords(normalized, vinfo.get("keywords", []))
            if matched_keywords:
                query_text = self._QUERY_TEMPLATES.get(vid, vinfo["name"])
                if query_text not in seen_query_texts:
                    seen_query_texts.add(query_text)
                    requests.append(
                        RewriteRequest(
                            violation_type_id=vid,
                            query_text=query_text,
                            keywords=matched_keywords,
                        )
                    )

        # 2. Append the original text as a catch-all query only if no V00
        #    query already exists (avoids duplicate V00 entries that harm
        #    multi-query agreement scores).
        if normalized not in seen_query_texts and not any(
            req.violation_type_id == "V00" for req in requests
        ):
            requests.append(
                RewriteRequest(
                    violation_type_id="V00",
                    query_text=normalized,
                    keywords=[],
                )
            )

        return requests

    @staticmethod
    def _match_keywords(text: str, keywords: list[str]) -> list[str]:
        """Return the list of keywords that appear in text."""
        found = []
        for kw in keywords:
            if kw in text:
                found.append(kw)
        return found
