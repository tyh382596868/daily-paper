---
type: concept
aliases: [Delta Rule, delta 更新]
---

# Delta 规则

## 定义
经典在线学习中的误差驱动更新规则，仅对"目标-当前预测"的残差进行修正，是 [[Gated DeltaNet]] 的核心更新机制。

## 数学形式

$$
S_t = S_{t-1} (I - K_t \beta_t K_t^\top) + V_t \beta_t K_t^\top
$$

可改写为残差形式：

$$
S_t = S_{t-1} + (V_t - S_{t-1} K_t) \beta_t K_t^\top
$$

## 核心要点

1. **残差更新**: 只修正预测与目标之间的差，比直接覆盖更稳定
2. **$\beta_t$ 门控**: 控制每次更新强度，可学习
3. **谱稳定**: 转移矩阵 $I - K_t \beta_t K_t^\top$ 在 $K$ 归一化下满足 $\|M\|_2 \le 1$
4. **应用**: Gated DeltaNet、DeltaNet、Mamba 等线性循环模型的关键单元

## 代表工作
- DeltaNet（原始论文）
- Gated DeltaNet
- [[SANA-WM]]: 将 Delta 规则用于帧级 GDN

## 相关概念
- [[Frame-wise Gated DeltaNet]]
- [[线性注意力]]
- [[空间稳定的键归一化]]
