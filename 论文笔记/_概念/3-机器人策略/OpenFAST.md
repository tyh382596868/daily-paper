---
type: concept
aliases: [OpenFAST, Open FAST Tokenizer]
---

# OpenFAST

## 定义

Allen Institute for AI 开源的跨形态机器人动作分词器，将 1 秒连续动作轨迹通过频域变换 + 字节对编码映射为 2048 词汇量的离散 token，支持 5 种机器人形态，是 MolmoAct2 的核心组件之一。

## 核心要点

1. 在 100 万动作序列上训练，覆盖绝对关节角控制和末端执行器增量控制两种模式
2. 所有形态统一 padding 到 32 维，1-99 百分位归一化消除量纲差异
3. 2048 token 词汇表，支持语言模型式自回归动作预测

## 代表工作

- [[MolmoAct2]]: OpenFAST 用于 MolmoAct2 预训练阶段的离散动作表示

## 相关概念

- [[字节对编码]]
- [[频域变换]]
- [[VLA]]
- [[Action Chunking]]
