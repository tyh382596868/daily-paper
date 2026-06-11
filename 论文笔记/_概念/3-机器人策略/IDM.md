---
type: concept
aliases: [Inverse Dynamics Model, 逆动力学模型, IDM]
---

# IDM（Inverse Dynamics Model）

## 定义

给定两个相邻观测帧 $(o_t, o_{t+k})$，推断出导致该视觉变化的**潜在动作** $\hat{z}$ 的模型。是 [[LAM]] 的核心组件之一，用于从无标注视频中自监督学习动作表示。

## 数学形式

$$
\hat{z} = \text{IDM}(o_t, o_{t+k})
$$

在离散化设计（如 Moto / LARA）中进一步量化：

$$
\hat{z} = \arg\min_{k} \| z_e(o_t, o_{t+k}) - e_k \|_2
$$

其中 $e_k$ 为 [[VQ-VAE]] codebook 中的向量。

## 核心要点

1. **无需动作标注**: 仅依赖视频帧对进行自监督训练
2. **通常与 FDM 配合**: IDM 推断潜在动作，[[FDM]] 验证该动作能否重建未来帧
3. **虚假相关性风险**: 若无真实动作接地，IDM 可能将背景变化、光照等非动作信息编入潜在动作
4. **LARA 的改进**: 通过联合训练让真实动作轨迹监督 IDM，减少虚假视觉相关性

## 代表工作

- [[LARA]]: 联合训练 IDM（LAM 组件）与扩散 VLA，通过表示对齐实现接地
- [[Moto]]: 使用 IDM + FDM 的 ViT 架构预训练运动 token
- [[VLA-JEPA]]: 类 JEPA 方式学习潜在动作

## 相关概念

- [[FDM]]
- [[LAM]]
- [[VQ-VAE]]
- [[LARA]]
