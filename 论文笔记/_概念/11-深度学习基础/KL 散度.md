---
type: concept
aliases: [KL 散度, KL Divergence, Kullback-Leibler Divergence, 相对熵]
---

# KL 散度

## 定义
Kullback-Leibler 散度衡量两个概率分布之间的差异，常用作正则项把学习到的分布约束在某个参考分布附近。

## 数学形式
$$D_{\text{KL}}(P\,\|\,Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}$$

连续形式为积分。KL 非负、不对称，$D_{\text{KL}}(P\|Q)=0$ 当且仅当 $P=Q$。

## 核心要点
1. 在 RL 与对齐中作为信任域约束，防止策略偏离行为分布过远导致 OOD。
2. 在 VAE / 扩散模型中作为变分下界的一部分约束潜变量分布。
3. 不对称性使「前向 KL」（覆盖型）与「反向 KL」（寻峰型）行为不同。

## 代表工作
- [[PROWL]]: 用 KL 约束把对抗课程的策略拽在行为分布附近，避免世界模型在 OOD 轨迹上训练
- [[PPO]]: 用 KL 惩罚/裁剪限制策略更新幅度

## 相关概念
- [[强化学习]]
- [[DPO]]
- [[PPO]]
