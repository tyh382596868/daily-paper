---
type: concept
aliases: [Unified World Model]
---

# UWM

## 定义

Unified World Model：单 Transformer 内**解耦** world 与 action 的扩散调度——给两侧分别分配独立可控的噪声级别。通过 test-time 噪声切换，**同一模型**可承担策略推理 / forward dynamics / inverse dynamics / 纯视频生成等多种角色。

## 核心要点

1. **解耦扩散调度**: $\text{video} \perp \text{action}$ noise schedule。
2. **多任务统一**: 一个 checkpoint, 四种使用模式。
3. **优雅处理无动作数据**: 把 action 侧全噪声化即得到纯视频建模目标。
4. 属于 [[Joint WAM|Joint WAM]] 的 Diffusion / Unified-Stream / Explicit-Future 子类。

## 代表工作

- [[WAM-Survey]] Table 3 中作为可灵活切换模式的 Unified-Stream 代表。

## 相关概念

- [[PAD]]
- [[Cosmos-Policy]]
- [[World Action Model]]
