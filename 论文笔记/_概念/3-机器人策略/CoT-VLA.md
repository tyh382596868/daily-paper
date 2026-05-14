---
type: concept
aliases: [CoTVLA, Chain-of-Thought VLA]
---

# CoT-VLA

## 定义

Autoregressive Joint WAM，在 VLA 推理链中加入"未来视觉"作为 Chain-of-Thought 的中间步骤，把视觉推理与动作生成串到同一 token 序列。

## 核心要点

1. **视觉 CoT**: 显式预测未来帧作为推理中间产物。
2. **统一离散表示**: 视频 + 动作共享离散 token 空间。
3. 属于 [[Joint WAM|Joint WAM]] 的 Autoregressive / Unified-Discrete 子类。

## 代表工作

- [[WAM-Survey]] Table 2 中作为 Unified-Discrete 代表。

## 相关概念

- [[CoT]]
- [[WorldVLA]]
- [[World Action Model]]
