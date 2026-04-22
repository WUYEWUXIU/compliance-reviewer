#!/usr/bin/env python3
"""
自动为合规审核 RAG 系统的 chunk 标注 violation_tags 和 compliance_tags。

用法:
    python scripts/auto_tag_chunks.py
"""

import json
import sys
from pathlib import Path

# 将 src 加入路径以导入配置
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config.violation_types import VIOLATION_TYPES, COMPLIANCE_TAGS

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "chunks.json"
MAX_VIOLATION_TAGS = 3

# 关键词上下文排除规则：tag_id -> list of exclusion substrings
# 若关键词出现在包含排除子串的上下文中，则跳过该匹配
EXCLUSION_CONTEXTS = {
    "V03": ["第一款", "第二款", "第三款", "第 款", "本条第", "第 条"],
    "V05": ["发布保险消费风险提示", "做好风险提示", "消费风险提示"],
    "V08": ["退保金", "退保损失", "退保流程", "退保渠道", "退保条件", "退保管理", "退保申请", "退保时"],
    "V09": ["备案手续", "备案编号", "备案名称"],
}


def _is_excluded_context(text: str, keyword: str, start_pos: int, tag_id: str) -> bool:
    """检查关键词是否出现在应排除的上下文中。"""
    exclusions = EXCLUSION_CONTEXTS.get(tag_id, [])
    if not exclusions:
        return False
    # 扩大窗口到关键词前后各 15 个字符
    window_start = max(0, start_pos - 15)
    window_end = min(len(text), start_pos + len(keyword) + 15)
    window = text[window_start:window_end]
    for ex in exclusions:
        if ex in window:
            return True
    return False


def match_tags(text: str, tag_definitions: dict, is_violation: bool = False) -> list:
    """根据 keywords 匹配标签，返回匹配的 tag ID 列表（去重，保持首次匹配顺序）。"""
    matched = []
    seen = set()
    text_lower = text.lower()

    for tag_id, info in tag_definitions.items():
        # V00 keywords 为空，跳过自动匹配
        if tag_id == "V00":
            continue
        keywords = info.get("keywords", [])
        for kw in keywords:
            kw_lower = kw.lower()
            idx = text_lower.find(kw_lower)
            if idx != -1:
                # 上下文排除过滤
                if is_violation and _is_excluded_context(text_lower, kw_lower, idx, tag_id):
                    continue
                if tag_id not in seen:
                    matched.append(tag_id)
                    seen.add(tag_id)
                break  # 该 tag 已匹配，不再检查其他关键词
    return matched


def is_general_article(article_id: str) -> bool:
    """判断是否为总则/定义性条文（第一条、第二条）。"""
    return article_id in ("1", "2")


def tag_chunks(chunks: list) -> list:
    """为每个 chunk 标注标签，返回新的 chunk 列表。"""
    tagged = []
    for chunk in chunks:
        new_chunk = dict(chunk)
        text = new_chunk.get("article_text", "")
        article_id = str(new_chunk.get("article_id", ""))

        # 合规标签：所有 chunk 都检查
        compliance = match_tags(text, COMPLIANCE_TAGS, is_violation=False)

        # 违规标签：总则/定义性条文跳过
        if is_general_article(article_id):
            violations = []
        else:
            violations = match_tags(text, VIOLATION_TYPES, is_violation=True)
            if len(violations) > MAX_VIOLATION_TAGS:
                violations = violations[:MAX_VIOLATION_TAGS]

        new_chunk["violation_tags"] = violations
        new_chunk["compliance_tags"] = compliance
        tagged.append(new_chunk)
    return tagged


def generate_report(tagged: list) -> dict:
    """生成统计报告。"""
    report = {
        "total_chunks": len(tagged),
        "violation_counts": {},
        "compliance_counts": {},
        "untagged": 0,
        "examples": {},
    }

    # 初始化计数器
    for vid in VIOLATION_TYPES:
        if vid == "V00":
            continue
        report["violation_counts"][vid] = 0
        report["examples"][vid] = []

    for cid in COMPLIANCE_TAGS:
        report["compliance_counts"][cid] = 0

    for chunk in tagged:
        vtags = chunk.get("violation_tags", [])
        ctags = chunk.get("compliance_tags", [])

        if not vtags and not ctags:
            report["untagged"] += 1

        for vid in vtags:
            report["violation_counts"][vid] = report["violation_counts"].get(vid, 0) + 1
            if len(report["examples"].get(vid, [])) < 2:
                report["examples"].setdefault(vid, []).append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "article_text": chunk.get("article_text", "")[:120],
                    }
                )

        for cid in ctags:
            report["compliance_counts"][cid] = report["compliance_counts"].get(cid, 0) + 1

    return report


def print_report(report: dict) -> None:
    """打印统计报告到 stdout。"""
    print("=" * 60)
    print("Chunk 自动标注统计报告")
    print("=" * 60)
    print(f"\n总 chunk 数: {report['total_chunks']}")
    print(f"未标注任何标签的 chunk 数: {report['untagged']}")

    print("\n--- 违规标签分布 ---")
    for vid, cnt in sorted(report["violation_counts"].items()):
        name = VIOLATION_TYPES.get(vid, {}).get("name", vid)
        print(f"  {vid} ({name}): {cnt}")

    print("\n--- 合规标签分布 ---")
    for cid, cnt in sorted(report["compliance_counts"].items()):
        name = COMPLIANCE_TAGS.get(cid, {}).get("name", cid)
        print(f"  {cid} ({name}): {cnt}")

    print("\n--- 标注示例（每类违规类型取2个典型 chunk） ---")
    for vid, examples in sorted(report["examples"].items()):
        name = VIOLATION_TYPES.get(vid, {}).get("name", vid)
        print(f"\n{vid} - {name}:")
        for i, ex in enumerate(examples, 1):
            print(f"  示例 {i}: [{ex['chunk_id']}]")
            print(f"    {ex['article_text']}")

    print("\n" + "=" * 60)


def main() -> None:
    if not CHUNKS_PATH.exists():
        print(f"错误: 找不到 chunks.json: {CHUNKS_PATH}", file=sys.stderr)
        sys.exit(1)

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    tagged = tag_chunks(chunks)

    # 覆盖原文件
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(tagged, f, ensure_ascii=False, indent=2)

    report = generate_report(tagged)
    print_report(report)


if __name__ == "__main__":
    main()
