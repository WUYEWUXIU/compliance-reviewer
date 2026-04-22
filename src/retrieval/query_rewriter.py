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

    # Regulatory-sensitive keywords used as fallback when rule matches are sparse.
    _FALLBACK_KEYWORDS = [
        "保本",
        "保息",
        "零风险",
        "年化",
        "稳稳到手",
        "固定收益",
        "确定回报",
        "保证收益",
        "稳赚",
        "最优",
        "第一",
        "最强",
        "绝无仅有",
        "最好",
        "顶级",
        "唯一",
        "明星推荐",
        "专家推荐",
        "权威认证",
        "风险提示",
        "犹豫期",
        "退保损失",
        "比其他",
        "业内最优",
        "贬低",
        "碾压",
        "免保费",
        "零手续费",
        "无管理费",
        "免费",
        "退保",
        "转保",
        "升级换代",
        "旧保单",
        "换了买",
        "银保监会批准",
        "备案编号",
        "特别批准",
        "监管备案",
        "客户案例",
        "内部客户",
        "专享",
        "仅限老客户",
        "送体检",
        "保单变现",
        "免费旅游",
        "送礼",
        "返佣",
    ]

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

        # 2. Fallback: if fewer than 2 rule matches, extract sensitive keywords
        if len(requests) < 2:
            fallback_keywords = self._extract_fallback_keywords(normalized)
            if fallback_keywords:
                fallback_query = "、".join(fallback_keywords[:3])
                if fallback_query not in seen_query_texts:
                    seen_query_texts.add(fallback_query)
                    requests.append(
                        RewriteRequest(
                            violation_type_id="V00",
                            query_text=fallback_query,
                            keywords=fallback_keywords,
                        )
                    )

        # 3. Append the original text as a catch-all query only if no V00
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

    @staticmethod
    def _extract_fallback_keywords(text: str) -> list[str]:
        """Extract regulatory-sensitive keywords from text for fallback queries."""
        found = []
        for kw in QueryRewriter._FALLBACK_KEYWORDS:
            if kw in text and kw not in found:
                found.append(kw)
        return found
