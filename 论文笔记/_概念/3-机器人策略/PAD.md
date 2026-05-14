---
type: concept
aliases: [Predict-Act-Diffuse]
---

# PAD

## 定义

Unified-Stream Diffusion Joint WAM 的奠基性工作：把 RGB / pose / depth 编码到统一 latent token 序列，与 action chunk 一起在共享 DiT 中**联合去噪**预测未来帧与动作。

## 核心要点

1. **从零训练**: 利用无动作的互联网视频做预训练，通过 action padding + attention masking 兼容。
2. **联合监督**: 消融显示去掉 future-image prediction 或 video co-training 都会降性能，证明世界监督对训练 / 推理都有意义。
3. 属于 [[Joint WAM|Joint WAM]] 的 Diffusion / Unified-Stream / Explicit-Future 子类。

## 代表工作

- [[WAM-Survey]] Table 3 中作为 Unified-Stream Explicit 的开山之作。
- [[UWM]] 在 PAD 基础上让 video / action 噪声独立可控。

## 相关概念

- [[扩散变换器]]
- [[UWM]]
- [[World Action Model]]
