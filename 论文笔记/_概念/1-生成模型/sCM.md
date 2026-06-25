---
type: concept
aliases: [Scaled Consistency Model, Continuous-time CM, 连续时间一致性模型, sCM]
---

# sCM（Scaled Consistency Model）

## 定义

sCM 是 [[Consistency Model]] 的连续时间扩展，将离散步长的一致性约束替换为沿 ODE 轨迹的**切线监督**，通过 [[JVP]] 计算精确切线方向，实现比 dCM（离散 CM）快约 **10×** 的收敛速度。

## 数学形式

将基础预测函数 $\mathbf{f}_\theta$ 提升为连续时间版本 $\mathbf{F}_\theta$，训练损失为：

$$
\mathcal{L}_{\text{sCM}}(\theta) = \mathbb{E}\!\left[\left\| \mathbf{F}_\theta - \mathbf{F}_{\theta^-} - \frac{\mathbf{g}}{\|\mathbf{g}\|_2^2 + c} \right\|_2^2\right]
$$

其中切线 $\mathbf{g} = w(t)\,\mathrm{d}\mathbf{f}_{\theta^-}/\mathrm{d}t$ 由 [[JVP]] 沿 ODE 方向计算。

## 与离散时间 dCM 的对比

| 特性 | dCM | sCM |
|------|-----|-----|
| 一致性目标 | 相邻离散步 $\hat{\mathbf{x}}_{t-\Delta t}$ | ODE 切线（连续） |
| 梯度估计精度 | 有限差分近似 | 精确 JVP |
| 收敛速度 | 基准 | **约 10× 更快** |
| 实现复杂度 | 低 | 需要 JVP 内核 |

## 扩展变体

- **TF-sCM**（[[Causal-rCM]]）：将 sCM 应用于 Teacher-Forcing 因果自回归视频场景，需要自定义掩码的 FlashAttention-2 JVP 核
- [[MeanFlow]]：从 Mean Flow 角度理解连续时间蒸馏，与 sCM 兼容

## 代表工作

- [[Causal-rCM]]: 首个因果自回归视频 sCM 实现，TF-sCM 在 1k 步内达到 TF-dCM 10k 步效果

## 相关概念

- [[Consistency Model]]
- [[JVP]]
- [[MeanFlow]]
- [[rCM]]
- [[DMD]]
