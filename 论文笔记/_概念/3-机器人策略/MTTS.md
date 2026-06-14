---
type: concept
aliases: [Morphology-Aware Tactile Token Space, 形态感知触觉令牌空间]
---

# MTTS（形态感知触觉令牌空间）

## 定义
MTTS（Morphology-Aware Tactile Token Space）是 [[FTP-1]] 提出的统一触觉表示框架，将异构触觉传感器（图像型、阵列型、状态型）的输出组织为 24 个功能区域的令牌序列，通过共享功能区域嵌入实现跨传感器对齐。

## 数学形式

触觉令牌 $\mathbf{T}$ 由各传感器编码器生成，加上可学习功能区域嵌入 $\mathbf{E}_{area}$：

$$
\mathbf{T} = \mathrm{Encoder}(\mathcal{X}) + \mathbf{E}_{area} + \mathbf{E}_{hand}
$$

其中 $\mathbf{E}_{hand}$ 区分左右手，$\mathcal{X}$ 为异构触觉观测。

## 24 功能区域划分

- **槽位 0–14**: 手部功能区域（拇指尖、各指节等）
- **槽位 15–20**: 腕部 + 手指力矩传感器
- **槽位 0–1（夹爪）**: 映射至拇指尖 + 食指尖
- **槽位 21–23**: 保留扩展

## 核心要点
1. 传感器特定编码器（图像型→ViT+T3；阵列型→CNN；状态型→Fourier+MLP）将原始信号映射至 MTTS 令牌
2. **功能区域嵌入全传感器共享**，是实现跨传感器对齐的关键
3. 设计遵循手部形态学，使不同传感器的相同功能区域在表示空间中对齐

## 代表工作
- [[FTP-1]]: MTTS 提出论文，核心创新，实现 21 种传感器的统一表示

## 相关概念
- [[触觉传感器]]
- [[GelSight]]
- [[Contactile]]
- [[T3 Transformer]]
- [[触觉专家网络]]
