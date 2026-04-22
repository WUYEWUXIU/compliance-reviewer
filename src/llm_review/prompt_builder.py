"""Prompt builder for LLM compliance review.

Constructs system and user prompts for the Bailian LLM, embedding the
full violation-type catalogue, negation instructions, and structured output
schema.
"""

from __future__ import annotations

from typing import List

from src.config.violation_types import (
    COMPLIANCE_TAGS,
    NEGATION_PATTERNS,
    VIOLATION_TYPES,
)
from src.retrieval.hybrid_search import RerankResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_system_prompt() -> str:
    """Build the system prompt for the compliance-review LLM.

    Returns:
        A detailed system prompt containing role definition, violation
        catalogue, negation rules, and output schema instructions.
    """
    # --- Violation table ---
    violation_lines: List[str] = []
    for vid, vinfo in VIOLATION_TYPES.items():
        if vid == "V00":
            continue
        severity = vinfo.get("severity", "warning")
        violation_lines.append(
            f"  {vid}: {vinfo['name']} (severity: {severity})"
        )

    # --- Compliance tag table ---
    compliance_lines: List[str] = []
    for cid, cinfo in COMPLIANCE_TAGS.items():
        compliance_lines.append(f"  {cid}: {cinfo['name']}")

    # --- Negation patterns ---
    negation_list = ", ".join(NEGATION_PATTERNS)

    prompt = f"""你是一名资深的金融合规审核专家，擅长审查保险及金融产品的营销文案是否符合监管要求。

## 审核原则
1. 仅根据下方提供的【相关条文】和【参考条文】进行判断，禁止引用上下文之外的任何条文或知识。
2. 对文案中的否定表述保持敏感：如果文案明确否定了某种违规行为（如"不保本""非保证收益"），则不应将其判定为违规。
3. 常见否定表述包括：{negation_list}
4.  severity 分级：
   - critical：严重违规，必须立即整改
   - warning：一般违规，建议修改
   - info：轻微问题或提示性建议

## 违规类型枚举表（V01-V11）
{chr(10).join(violation_lines)}

## 合规正向行为标签（C01-C03）
{chr(10).join(compliance_lines)}

## 输出格式要求
你必须以严格的 JSON 格式输出审核结论，包含以下字段：
- compliant: "yes" 或 "no"
- violations: 违规项数组，每项包含：
    - violation_type_id: 违规类型编号（如 V02）
    - violation_type_name: 违规类型名称
    - article_id: 相关条文编号（如"第二十一条"）
    - doc_name: 来源文档名称
    - article_text: 条文原文摘要
    - reason: 具体违规说明
    - severity: critical / warning / info
    - directional_advice: 修改建议
- positive_compliance: 检测到的合规正向行为数组，每项包含：
    - tag_id: 标签编号（如 C01）
    - tag_name: 标签名称
    - evidence: 文案中的具体证据

如果未发现违规，violations 为空数组；如果未发现合规正向行为，positive_compliance 为空数组。
"""
    return prompt.strip()


def build_user_prompt(
    marketing_text: str,
    top_chunks: List[RerankResult],
    reference_chunks: List[RerankResult],
) -> str:
    """Build the user prompt from marketing text and retrieved chunks.

    Args:
        marketing_text: The marketing copy to review.
        top_chunks: Top-5 chunks after reranking, sorted by score descending.
        reference_chunks: Citation-graph expansion results.

    Returns:
        A formatted user prompt ready for the LLM.
    """
    lines: List[str] = []
    lines.append("## 待审核营销文案")
    lines.append("")
    lines.append(marketing_text.strip())
    lines.append("")

    # --- Top-5 relevant articles ---
    lines.append("## 相关条文（按相关性排序，Top-5）")
    lines.append("")
    for idx, chunk in enumerate(top_chunks[:5], start=1):
        doc_name, article_id = _parse_chunk_id(chunk.chunk_id)
        lines.append(f"[{idx}] 来源：{doc_name} | 条文：{article_id}")
        lines.append(f"正文：{chunk.text.strip()}")
        lines.append("")

    # --- Reference chunks (graph expansion) ---
    if reference_chunks:
        lines.append("## 参考条文（图扩展结果，仅作引用上下文）")
        lines.append("")
        for idx, chunk in enumerate(reference_chunks, start=1):
            doc_name, article_id = _parse_chunk_id(chunk.chunk_id)
            lines.append(f"[R{idx}] 来源：{doc_name} | 条文：{article_id}")
            lines.append(f"正文：{chunk.text.strip()}")
            lines.append("")

    lines.append("请根据以上条文对营销文案进行合规审核，并以 JSON 格式输出审核结论。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_chunk_id(chunk_id: str) -> tuple[str, str]:
    """Parse a chunk_id like 'docName_articleId_seq' into (doc_name, article_id).

    Falls back to ('未知文档', '未知条文') if the format is unrecognised.
    """
    parts = chunk_id.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "未知文档", "未知条文"
