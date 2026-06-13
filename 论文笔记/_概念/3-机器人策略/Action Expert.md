---
type: concept
aliases: [Action Expert, 动作专家]
---

# Action Expert

## 定义

Action Expert 是 [[VLA]] / [[WLA]] / [[World Action Model|WAM]] 类模型中负责**生成可执行动作**的 head。通常基于共享 backbone 隐藏态和机器人当前状态，用 [[Flow Matching]] / 扩散 / 离散 token 等方式解码出未来 $H$ 步 [[Action Chunking|动作块]]。

## 核心要点

1. **输入**: 共享隐藏态 $h_t$ + 机器人状态 $q_t$
2. **输出**: 动作块 $a_{t:t+H}$（如 $H=10$ 或 $H=20$）
3. **生成方式**:
   - **Flow Matching**: [[π0]]、[[WLA]] 主流做法
   - **扩散**: [[Diffusion Policy]]
   - **离散 token**: [[OpenVLA]] 早期方案
4. **轻量**: [[WLA]] 中仅 390M，远小于 backbone

## 代表工作

- [[WLA]]: flow-matching head，390M
- [[π0]]: flow matching 动作头的代表
- [[FLOWER]]: flow matching 动作专家
- [[Diffusion Policy]]: 扩散式动作头
- [[APT]]: 通过两阶段预训练解决行动专家捷径学习问题，引入[[层级门控融合]]注入语言特征

## 相关概念

- [[Flow Matching]]
- [[Action Chunking]]
- [[Diffusion Policy]]
