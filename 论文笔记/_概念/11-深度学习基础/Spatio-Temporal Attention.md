---
type: concept
aliases: [时空注意力, ST-Attention, Spatiotemporal Attention]
---

# Spatio-Temporal Attention

## 定义

**Spatio-Temporal Attention** 指允许 query 跨时间和空间两种维度同时计算注意力的机制。与"先空间后时间"或"先时间后空间"的分解式相比，时空注意力可以让一个 token 直接看到**不同时间步、不同空间邻居**的特征，从而捕捉物体运动、接触传递、相对位移等强时空耦合现象。

## 数学形式

设第 $t$ 帧第 $i$ 个 token 特征为 $h_{t,i}$，时空注意力的 query/key/value 形式：

$$
\mathrm{Attn}(h_{t,i}) = \sum_{(t',j) \in \mathcal{N}(t,i)} \mathrm{softmax}\!\left(\frac{q_{t,i}^\top k_{t',j}}{\sqrt{d}}\right) v_{t',j}
$$

其中邻域 $\mathcal{N}(t,i)$ 可定义为：

- 时空 k-近邻（按 $(x, y, z, t)$ 距离取最近 k 个）
- 跨时间窗口 $|t-t'| \le \Delta T$ 内的所有空间邻居

## 实现变体

1. **Full Spatio-Temporal**: 注意力跨所有 (空间, 时间) 对，$O(N^2 T^2)$ 复杂度
2. **Local ST**: 限定时空窗口 (常见，复杂度 $O(N k T)$ 或类似)
3. **Vector ST (PTv2-style)**: 在 [[Point Transformer]] 的向量注意力基础上扩展邻域到跨时间，用相对位置 + 时间偏移编码

## 与分解式注意力对比

| 类型 | 复杂度 | 捕捉能力 |
|------|--------|----------|
| 时间 + 空间分解 | 较低 | 只能间接通过逐层混合捕捉时空耦合 |
| 联合 Spatio-Temporal | 较高 | 直接建模"某物体在过去某帧的某邻居" |

MRO-GWM 实验表明：在多刚体动力学预测中去掉时空注意力后误差增加约 10%，说明跨时间的空间近邻信息显著有助于运动预测。

## 代表工作

- **TimeSformer** (Bertasius 2021): 分解式视频时空注意力
- **MViT v2**: 多尺度视频时空 ViT
- **MRO-GWM** (2026): 提出新颖的物体中心稀疏时空注意力（k-NN 跨时间）

## 关联概念

- [[Transformer]]
- [[自注意力]]
- [[Cross-Attention]]
- [[Point Transformer]]
