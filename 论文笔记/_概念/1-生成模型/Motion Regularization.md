---
type: concept
aliases: [MoReg, 运动正则化, Motion-Aware Caching]
---

# Motion Regularization (MoReg)

## 定义

AdaCache 提出的一种运动感知调制机制：通过在视频 DiT 的 Temporal Self-Attention 残差中提取帧间差分，估计当前视频的运动量，将其乘入缓存调度信号，使运动剧烈的区域自动获得更频繁的重计算（更少缓存）。

## 核心要点

1. **零参数运动估计**：直接从已有的 STA 残差 $p_t^l$ 做帧差，不引入任何额外网络
2. **双信号调制**：同时使用运动量 $m_t^l$（当前运动强度）和运动梯度 $mg_t^l$（运动加速/减速趋势）
3. **乘性调制**：把运动量乘入特征变化速率 $c_t^l$，通过 codebook 传导到缓存时长 $\tau_t^l$
4. **步数越多效果越显著**：在 Open-Sora-Plan（150步）上 VBench 提升约 3.5 点，而短步数（30步）提升约 0.1 点

## 数学形式

运动量（Eq. 7）：

$$
m_t^l = \| p_{t,\, i:N}^l - p_{t,\, 0:N-i}^l \|
$$

运动梯度（Eq. 8）：

$$
mg_t^l = \frac{m_t^l - m_{t+k}^l}{k}
$$

运动正则化后的距离（Eq. 9）：

$$
c_t^l \leftarrow c_t^l \cdot (m_t^l + mg_t^l)
$$

其中 $N$ 为视频帧数，$i$ 为帧偏移（取 1），$k$ 为自上次重算起的步数间隔。

## 代表工作

- [[AdaCache]]: 首次提出 MoReg，用于视频 DiT 推理的自适应缓存调度

## 相关概念

- [[Adaptive Caching Schedule]]: MoReg 是其运动感知扩展
- [[Spatio-Temporal Attention]]: MoReg 操作的特征来源
- [[Cache Update]]: 缓存更新触发逻辑
