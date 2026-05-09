---
type: concept
aliases: [Open FAST, open fast tokenizer]
---

# OpenFAST

## 定义

OpenFAST 是 MolmoAct2 团队发布的开源动作分词器，将机器人 1 秒连续动作轨迹转化为 2048 词汇量的离散 token，支持 5 种机器人形态。

## 核心要点

1. 流程：1 秒轨迹 → [[频域变换]] → 系数量化 → [[字节对编码]]（BPE）→ 2048-token 词汇
2. 各维度用 1-99 百分位统计归一化，夹爪指令单独处理
3. 所有形态统一 padding 到 32 维，实现跨形态统一表示
4. 在 100 万动作序列（5 种形态）上训练

## 代表工作

- [[MolmoAct2]]：使用 OpenFAST 实现离散动作 token 预训练监督

## 相关概念

- [[频域变换]]
- [[字节对编码]]
- [[Action Chunking]]
