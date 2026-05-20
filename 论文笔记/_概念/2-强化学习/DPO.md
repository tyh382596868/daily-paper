---
type: concept
aliases: [DPO, Direct Preference Optimization, 直接偏好优化]
---

# DPO

## 定义
Direct Preference Optimization（直接偏好优化）是一种无需显式训练奖励模型、直接用偏好数据（preferred/rejected 成对样本）优化策略的对齐方法。

## 数学形式
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

其中 $y_w$ 为偏好样本、$y_l$ 为拒绝样本，$\pi_{\text{ref}}$ 为冻结参考策略，$\beta$ 控制偏离参考策略的强度。

## 核心要点
1. 把 RLHF 的「奖励建模 + RL 优化」两步合并为一个分类式损失，训练更稳定、实现更简单。
2. 隐式奖励由策略与参考策略的对数似然比给出，无需单独的 reward model。
3. 在机器人/VLA 场景中被用于把无标签信号（如延迟错位、物理一致性）转成成对偏好。

## 代表工作
- [[DEFLECT]]: 把异步推理的「新鲜/陈旧」反事实动作对构造成 DPO margin，做延迟鲁棒后训练
- [[PhyWorld]]: 用偏好优化对齐视频世界模型的物理一致性

## 相关概念
- [[强化学习]]
- [[GRPO]]
- [[PPO]]
