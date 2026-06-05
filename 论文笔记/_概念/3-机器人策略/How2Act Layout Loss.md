---
type: concept
aliases: [How2Act Layout, 10-DoF 位姿回归损失]
---

# How2Act Layout Loss

## 定义

AffordanceVLA 中将 layout token 回归到 10-DoF 空间参数（位置 3 + 6D 旋转 + 夹爪 1）的 [[Smooth-L1 Loss|Smooth-L1]] 损失。

## 数学形式

$$
\mathcal{L}_{layout} = \frac{1}{10} \sum_{j=1}^{10} \mathrm{SmoothL1}\left( \hat{y}^{(j)}_{layout},\ y^{(j)}_{layout} \right)
$$

## 核心要点

1. **10-DoF 表示**: 位置 (3) + 6D 旋转 (6) + 夹爪开合 (1)；6D 旋转避免欧拉角奇异。
2. **Smooth-L1**: 在 $|x|<1$ 时为 $0.5 x^2$，否则为 $|x|-0.5$，兼顾大误差鲁棒和小误差精度。
3. **与 Shape 配对**: 与 [[How2Act Shape Loss]] 联合给出完整 3D 几何指引。
4. **权重低**: $\lambda_{layout} = 0.04$，因其数值范围较小，需调权防止主导其他损失。

## 代表工作

- [[AffordanceVLA]]

## 相关概念

- [[Smooth-L1 Loss]]
- [[How2Act Shape Loss]]
- [[Affordance Generation Expert]]
- [[Affordance Forecasting]]
