---
type: concept
aliases: []
---

# Motus

## 定义

Multi-Stream Diffusion Joint WAM，使用独立的 Video DiT 与 Action DiT，通过 [[交叉注意力|Cross-Attention]] 在去噪步骤中互相条件化。

## 核心要点

1. **多流耦合**: 不同模态走独立扩散主干，避免互相干扰。
2. **Cross-Attention 桥接**: 每层把视频 token 与动作 token 交互。
3. 属于 [[Joint WAM|Joint WAM]] 的 Diffusion / Multi-Stream / CA-Coupled 子类。

## 代表工作

- [[WAM-Survey]] Figure 6 中作为 CA-Coupled 模式代表。

## 相关概念

- [[扩散变换器]]
- [[交叉注意力]]
- [[World Action Model]]
