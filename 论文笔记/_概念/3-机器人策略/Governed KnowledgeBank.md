---
type: concept
aliases: [治理知识库, 有治理的知识银行, KnowledgeBank]
---

# Governed KnowledgeBank

## 定义

Governed KnowledgeBank 是 GeneralVLA-2 中提出的受控长期记忆系统，通过引入质量元数据（置信度、生命周期状态、冲突追踪）和精度导向检索，将仅追加型操作经验库转变为支持准入控制和生命周期管理的主动治理知识库。

## 数学形式

记忆记录结构：

$$
m = (i, q, c, y, s, z, \kappa, R, u, d, L, v)
$$

精度导向检索评分：

$$
S(q_t, X_t, m) = r_{text}(q_t, m) + \kappa_m + b_{success}(m) + b_{recency}(m) + b_{usage}(m) - p_{conflict}(m) - p_{stale}(m)
$$

## 核心要点

1. **准入控制**: 仅通过验证器质量门控的记忆才进入 active 状态，防止低质量经验污染
2. **生命周期状态机**: provisional → active → summary → archive，自动管理记忆时效
3. **精度导向检索**: 综合语义相似度、置信度、成功经验、近期使用等多维度评分，超越纯语义检索

## 代表工作

- [[GeneralVLA2]]: 提出 Governed KnowledgeBank，在 Terminal-Bench 2.0 上提升 SR 4.53%，SWE-Bench 提升 Resolve 3.73%

## 相关概念

- [[In-Context Retrieval]]
- [[MemoryVLA]]
- [[episodic-memory]]
- [[GeoFuse-MV3D]]
