---
type: concept
aliases: [Continual Learning, Lifelong Learning, 持续学习, 终身学习]
---

# Continual Learning

## 定义

Continual Learning（持续学习）研究模型在**数据流非 i.i.d.、任务/分布随时间变化**的设定下如何持续积累知识而不灾难性遗忘（catastrophic forgetting）。在[[世界模型]]背景下，CL 通常体现为：环境的 [[Factors of Variation]] 在训练过程中逐步引入或漂移，模型需要在保留旧任务表现的同时适应新分布。

## 数学形式

任务序列 $\{\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_K\}$ 下，目标是：

$$
\min_\theta \; \sum_{k=1}^K \mathcal{L}_k(\theta) \quad \text{s.t. 数据按顺序到达，仅当前任务数据可见}
$$

平均准确率 $\mathrm{ACC}$ 与后向迁移 $\mathrm{BWT}$ 是经典指标：

$$
\mathrm{BWT} = \frac{1}{K-1} \sum_{i=1}^{K-1} \big( a_{K,i} - a_{i,i} \big)
$$

其中 $a_{j,i}$ 表示训练完任务 $j$ 后在任务 $i$ 上的准确率。

## 核心要点

1. **灾难性遗忘**：神经网络在新任务上微调会迅速损失旧任务能力
2. **三大类方法**：
   - 正则化（EWC、SI）：约束重要参数变化
   - 回放（Experience Replay、Generative Replay）：保留/生成旧数据
   - 结构性（PNN、PackNet）：为新任务分配新参数
3. **在 WM 中的意义**：智能体面对动态环境（光照/物体/物理参数变化），WM 需要在线适应而不忘旧策略
4. **评测要素**：FoV 维度的渐进引入提供了天然的 CL benchmark（[[StableWM]] 已支持）

## 代表工作

- Kirkpatrick et al., 2017: EWC（Elastic Weight Consolidation）
- Rebuffi et al., 2017: iCaRL
- [[StableWM]]: 为 WM 持续学习研究提供 FoV 可控环境

## 相关概念

- [[Factors of Variation]]
- [[世界模型]]
- [[Self-Supervised Learning]]
- [[Zero-shot Robustness]]
