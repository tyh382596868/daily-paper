---
type: concept
aliases: [动作块预测, action chunk]
---

# Action Chunking（动作块预测）

## 定义

Action Chunking 是机器人学习中的一种动作表示策略：模型在每个时间步预测未来 $H$ 步的动作序列（chunk），而非单步动作，从而减少预测频率、降低因果混淆。

## 核心要点

1. 预测整段轨迹（如 1 秒 = 50 帧），降低累积误差
2. 与时间集成（Temporal Ensembling）配合使用，平滑执行
3. 是 ACT、Diffusion Policy、VLA 等主流方法的标配

## 代表工作

- [[MolmoAct2]]：预测 1 秒动作块，通过 OpenFAST 离散化为 token

## 相关概念

- [[Flow Matching]]
- [[VLA]]
- [[OpenFAST]]
