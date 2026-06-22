---
type: concept
aliases: [内容自适应缓存调度, AdaCache Schedule, 自适应缓存]
---

# Adaptive Caching Schedule

## 定义

视频扩散模型推理加速技术：通过实时测量相邻去噪步之间 Transformer 层特征的变化速率，动态决定当前层可以安全地缓存残差多少步，使不同内容复杂度的视频自动获得最优算力分配。

## 核心要点

1. **内容依赖性**：不同视频（甚至同一视频的不同去噪阶段）的特征变化速率差异可达 3–5 倍，固定调度必然次优
2. **Codebook 映射**：将连续的变化速率 $c_t^l$ 离散映射到整数缓存时长 $\tau_t^l$，变化越慢缓存越久
3. **每层独立决策**：每个 Transformer 层各自维护缓存和计数器，互不影响
4. **无需训练**：纯推理端修改，不改变模型权重

## 数学形式

变化速率（Eq. 4）：

$$
c_t^l = \frac{\| p_t^l - p_{t+k}^l \|}{k}
$$

Codebook 查询与缓存决策（Eq. 5-6）：

$$
\tau_t^l = \mathrm{codebook}(c_t^l)
$$

$$
p_{t-k}^l = \begin{cases} p_t^l & k < \tau_t^l \\ \mathrm{STA}(f_{t-k}^l) & k = \tau_t^l \end{cases}
$$

**典型 Codebook（AdaCache-fast，Open-Sora 100步）**：

| $c_t^l$ 范围 | $\tau_t^l$（缓存步数）|
|:---:|:---:|
| < 0.03 | 12 |
| < 0.05 | 10 |
| < 0.07 | 8 |
| < 0.09 | 6 |
| < 0.11 | 4 |
| ≥ 0.11 | 3 |

## 代表工作

- [[AdaCache]]: 首次提出并系统验证内容自适应缓存调度，在 Open-Sora 720p 上实现 4.7× 推理加速
- [[TeaCache]]: 同类工作，使用固定阈值缓存（非内容自适应）

## 相关概念

- [[Cache Update]]: 缓存更新触发机制
- [[Motion Regularization]]: AdaCache 的运动感知扩展
- [[Diffusion Transformer]]: 应用目标架构
- [[TeaCache]]: 固定调度基线
