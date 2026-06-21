---
type: concept
aliases: [PearlVLA RefineNet, 未来引导精炼网络, Future-Guided RefineNet]
---

# RefineNet (PearlVLA)

## 定义
PearlVLA 中用于渐进精炼潜在动作计划的轻量网络模块，将冻结潜空间世界模型输出的未来观测潜变量转化为残差修正，逐步将粗粒度语义草稿精炼为细粒度潜在动作计划。

## 数学形式

$$
\Delta z^{(k)} = f_{\mathrm{RefineNet}}\!\left(\hat{o}_{t+1}^{(k)},\, z^{(k-1)}\right)
$$

$$
z^{(k)} = z^{(k-1)} + \alpha_k \cdot \Delta z^{(k)}
$$

其中 $\alpha_k$ 为 scheduled 系数，控制每轮写入步长。

## 核心要点
1. **scheduled residual write-back**: 每轮用 scheduling 系数 $\alpha_k$ 控制更新幅度，防止精炼过程不稳定
2. **反馈驱动**: 每轮世界查询随更新后的计划变化（而非复用静态未来特征），形成真正的迭代反馈环路
3. **辅助对齐**: 配合 anchored world queries 和对齐损失，将精炼过程约束在冻结 WM 的条件空间内

## 代表工作
- [[PearlVLA]]: RefineNet 的提出论文，K=4 轮精炼，27.5 Hz 实时吞吐

## 相关概念
- [[Latent Plan Refinement]]
- [[Latent World Model]]
- [[VLM Latent Space Deliberation]]
