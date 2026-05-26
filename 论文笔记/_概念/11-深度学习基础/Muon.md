---
type: concept
aliases: [Muon Optimizer, Muon 优化器]
---

# Muon

## 定义

一种针对 2D 矩阵参数的[[矩阵感知优化器]],用 [[Newton-Schulz 迭代]]近似[[矩阵符号函数]] $\mathrm{msign}$,把动量矩阵投影到 $UV^\top$ 后再更新,等价于对所有[[奇异值分解|奇异值]]做统一[[谱白化]]。

## 数学形式

$$
\Theta_t = \Theta_{t-1} - \eta \cdot \mathrm{msign}(M_t),\quad M_t = \mu M_{t-1} + G_t
$$

$$
\mathrm{msign}(M) = UV^\top \approx X_k,\quad X_{i+1} = a X_i + b X_i X_i^\top X_i + c X_i (X_i^\top X_i)^2
$$

默认 5 步 NS 迭代,系数 $(a,b,c) = (3.4445, -4.7750, 2.0315)$。

## 核心要点

1. **谱白化**: 所有奇异方向获得等量更新,等价于在正交矩阵流形上做最近邻投影。
2. **无 SVD**: 用 NS 多项式迭代代替 SVD,单步只含矩阵乘法,GPU 友好。
3. **适用于 2D 参数**: 嵌入层 / LayerNorm / 输出 head 通常仍用 [[AdamW]]。
4. **预训练加速**: 在 LLM 预训练阶段显著快于 AdamW。
5. **后训练局限**: 在低[[有效秩]]、低[[梯度信噪比]]场景(VLA 微调 / RLVR)会放大噪声方向,被 [[Pion]] 修复。

## 代表工作

- [[Pion]]: 用高通 NS 迭代替换 Muon 的全谱白化,修复 VLA/RLVR 失败模式

## 相关概念

- [[Newton-Schulz 迭代]]
- [[矩阵符号函数]]
- [[谱白化]]
- [[AdamW]]
- [[奇异值分解]]
