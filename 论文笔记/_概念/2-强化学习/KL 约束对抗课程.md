---
type: concept
aliases: [KL 约束对抗课程, KL-constrained Adversarial Curriculum, 约束对抗课程]
---

# KL 约束对抗课程

## 定义
KL 约束对抗课程是一种自动课程方法：训练一个对抗策略去主动暴露学习器（如世界模型）的高误差案例，同时用 [[KL 散度|KL 散度]]惩罚把该对抗策略锚定到一个冻结的行为参考策略上，防止它漂移到不真实的分布外行为。

## 数学形式
对抗策略的优化目标：
$$
\max_{\phi}\;\mathbb{E}_{\tau\sim\pi_{\phi}}[\mathrm{score}(\tau)]-c_{\mathrm{kl}}\,\mathbb{E}_{s\sim d^{\pi_{\phi}}}\left[\mathrm{KL}\left(\pi_{\phi}(\cdot\mid s)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid s)\right)\right]
$$

## 核心要点
1. 把"发现失败"与"行为有效性"解耦：score 项鼓励暴露高误差轨迹，KL 项确保失败是行为上可信、可学习的。
2. 约束强度 $c_{\mathrm{kl}}$ 是关键超参——太弱会导致 reward hacking（如运动饱和捷径），太强则探索不足。
3. 与无监督环境设计（UED）、对抗课程学习同源，但对抗者作用于动作分布而非环境参数。

## 代表工作
- [[PROWL]]: 用 KL 约束对抗策略暴露视频世界模型的高误差轨迹，证明有效训练取决于"失败发现"与"行为正则化"的平衡

## 相关概念
- [[KL 散度]]
- [[PPO]]
- [[PLR]]
- [[Reward Hacking]]
- [[强化学习]]
