---
type: concept
aliases: [分组超调制, Grouped Modulation, 分组频谱调制]
---

# Grouped Hyper-Modulation

## 定义

[[NIAF]] 提出的一种轻量 [[Hypernetwork]] 调制方案：把 [[MLLM]] 输出的 query token 序列**按目标 [[SIREN]] 层数 $L$ 分组**，每层用 $G$ 个 token 控制频率（权重）、$1$ 个 token 控制相位（偏置），通过独立小 MLP 映射为乘性 $\gamma^{(\ell)}$ 与加性 $\beta^{(\ell)}$ 调制系数。

## 数学形式

$$
\gamma^{(\ell)} = \mathrm{Concat}(\psi_{\gamma_1}(\mathbf{Z}_{\ell,1}), \ldots, \psi_{\gamma_G}(\mathbf{Z}_{\ell,G})), \quad \beta^{(\ell)} = \psi_\beta(\mathbf{Z}_{\ell, \mathrm{bias}})
$$

$$
\hat{\mathbf{W}}^{(\ell)} = \mathbf{W}^{(\ell)} \odot (1 + \gamma^{(\ell)}), \quad \hat{\mathbf{b}}^{(\ell)} = \mathbf{b}^{(\ell)} + \beta^{(\ell)}
$$

## 核心要点

1. **细粒度频谱控制**：每层独立 $G$ 个频率组，可表达不同 task 所需的频谱分布；
2. **参数量小**：相比全权重生成，调制系数只与隐层维度同阶；
3. **与 [[SIREN]] 天然配合**：乘性 $\gamma$ 改变正弦频率，加性 $\beta$ 改变相位，物理意义清晰；
4. **NIAF 中 $G=64$ 为甜蜜点**：$G=16$ 表现明显下降。

## 代表工作

- [[NIAF]]：原始提出者

## 相关概念

- [[SIREN]]
- [[Hypernetwork]]
- [[Neural Implicit Action Field]]
