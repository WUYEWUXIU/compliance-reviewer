# 关键设计说明

## 一、技术路径选择

本系统采用 **RAG + Prompt Engineering** 双路增强方案：

- **RAG**：将 3 篇监管文档按条文粒度分块，建立离线索引。审核时实时检索最相关条文作为 LLM 上下文，确保判断依据可追溯，避免幻觉。
- **Prompt Engineering**：System Prompt 明确要求模型输出结构化 JSON，包含合规结论、违规类型 ID、引用条文、置信度、改进建议，约束输出格式，便于下游解析。

> 为何不用 Agent Workflow？本任务是单轮确定性判断，无需多步工具调用。RAG + 强约束 Prompt 比 Agent 链更稳定、延迟更低。

---

## 二、混合检索设计

```
BM25（关键词匹配）+ FAISS（语义向量）→ RRF 融合 → gte-rerank 精排
```

- **BM25** 擅长命中"保本保息"等监管术语的精确匹配
- **向量检索** 覆盖语义相近的隐式违规
- **RRF 融合**（k=60）无需调参，稳定合并两路排序
- **Rerank** 用cross-encoder对 Top-60 再精排，取 Top-10 送入 LLM

---

## 三、Prompt 结构

```
[System]
你是金融合规审核专家。以下是相关监管条文：
{检索到的条文原文 × 10}

审核规则：
1. 严格基于上述条文判断，不得超出范围
2. 输出 JSON：{"compliant": "yes"|"no", "violations": [...], "confidence": 0.0-1.0}
3. 每条违规必须引用具体条文编号和原文

[User]
请审核以下营销文案：
{营销文案}
```

关键约束：**强制 JSON Schema**，输出解析失败时触发 fallback mock 审核，不影响系统可用性。

---

## 四、多模态支持

图文输入时，先用 `qwen-vl-max` 对图片做 OCR + 内容理解，提取文字描述后与文本拼接，进入统一审核流程。

---

## 五、效果评估模块

标注了 **38 条 Golden Set 样本**，覆盖：
- V01–V11 每类违规的典型正例 + 否定表述反例
- 多违规叠加的复合案例
- 隐式违规等边界难判案例

评估指标：
| 指标 | 含义 |
|------|------|
| `compliant_accuracy` | 合规/违规二分类准确率 |
| `exact_match_accuracy` | 违规类型集合完全匹配率 |
| `macro_precision/recall/F1` | 各违规类型宏平均 P/R/F1 |

运行 `python scripts/evaluate.py` 输出完整评估报告。

---
