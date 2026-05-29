---
type: concept
aliases: [自动微分, Automatic Differentiation, AutoDiff]
---

# Autograd

## 定义

Autograd（自动微分）是深度学习框架（PyTorch / JAX / TF）自动构建计算图、按链式法则计算梯度的能力。除了对参数求导，**还可以对输入坐标求解析高阶导数**——这一性质是 [[Neural Implicit Representation]] / [[SIREN]] / [[NIAF]] 能直接监督速度、加速度的底层基础。

## 数学形式

对一个网络 $f_\theta(x)$，框架自动构建：

$$
\nabla_x f, \quad \nabla^2_x f, \quad \ldots, \quad \nabla^k_x f
$$

均可通过反向传播得到，无需手工推导或有限差分。

## 核心要点

1. **对参数求导**：常规训练（loss 对 $\theta$）；
2. **对输入求导**：用于 INR / PINN / 梯度惩罚 / adversarial；
3. **高阶导数**：调用 `torch.autograd.grad(..., create_graph=True)` 递归求；
4. **稳定性依赖激活**：ReLU 二阶以上几乎处处为 0，因此 [[SIREN]] 选 $\sin$。

## 代表工作

- PyTorch / JAX / TensorFlow：所有现代框架的核心
- [[SIREN]]：依赖 autograd 对 $\tau$ 求高阶导数
- [[NIAF]]：用 autograd 解析求得动作速度、jerk

## 相关概念

- [[SIREN]]
- [[Neural Implicit Action Field]]
- [[Jerk]]
