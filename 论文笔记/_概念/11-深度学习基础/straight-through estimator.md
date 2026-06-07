---
type: concept
aliases: [straight-through estimator, STE, Straight-Through Estimator, 直通梯度估计器]
---

# straight-through estimator

## 定义

**Straight-Through Estimator (STE)** 是一种针对**不可微算子**（如阶跃函数、argmax、采样）的近似梯度方法：**前向用真实离散值，反向用恒等映射的梯度**。

## 数学形式

设可微近似 $\tilde{f}(x)$（如 sigmoid / softmax），真实离散映射 $f(x)$：

$$
y_{forward} = f(x),\quad \frac{\partial y}{\partial x}\bigg|_{backward} = \frac{\partial \tilde{f}(x)}{\partial x}
$$

PyTorch 标准写法：
```python
y = x + (f(x) - x).detach()
```

## 典型用途

- **二值化神经网络**：sign(x) 前向、tanh' 后向。
- **VQ-VAE** 的 codebook lookup：前向最近邻、反向恒等。
- **Categorical latent 采样**：[[DreamerV2]] / [[DreamerV3]] 用 STE 通过 categorical 采样反向传梯度。
- **Gumbel-Softmax** 的硬采样版本。

## 关联概念

- [[Categorical Latent]]
- [[DreamerV3]]
- [[stop-gradient]]
