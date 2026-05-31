---
type: concept
aliases: [RLHF, Reinforcement Learning from Human Feedback, 基于人类反馈的强化学习]
---

# RLHF

## 定义

Reinforcement Learning from Human Feedback（基于人类反馈的强化学习）：用人类对模型输出的偏好数据训练一个奖励模型（reward model），再用 RL 算法（典型如 [[PPO]] 或 [[GRPO]]）以该奖励模型为反馈信号微调策略，使输出对齐人类偏好。

## 数学形式

经典三阶段流水线：

1. **监督微调（SFT）**：在示范数据上做 MLE 训练。
2. **奖励模型训练**：偏好对 $(x, y_w, y_l)$ 上拟合 Bradley-Terry 模型：
$$
\mathcal{L}_R = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\bigl(r_\phi(x,y_w) - r_\phi(x,y_l)\bigr)\right]
$$

3. **RL 微调**：以 reward $r_\phi$ 为信号，带 KL 约束防漂移：
$$
\max_{\pi_\theta}\;\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta}\!\left[r_\phi(x,y)\right] - \beta\, D_\mathrm{KL}\!\left[\pi_\theta(\cdot\mid x)\,\|\,\pi_\mathrm{ref}(\cdot\mid x)\right]
$$

## 核心要点

1. **三阶段**：SFT → Reward Model → PPO/GRPO 微调。
2. **KL 约束防止 reward hacking** 和 mode collapse。
3. **视频/图像生成里的扩展**：偏好数据可换成美学评分、物理一致性评分；reward model 可直接用 VLM 当 judge。
4. **替代方案**：DPO 直接用偏好对绕过显式 reward model；GRPO 用 group-relative advantage 省 critic；KTO 用单边二元反馈。

## 代表工作

- InstructGPT (2022): 第一个把 RLHF 用到 LLM 对齐的系统工作。
- DPO (2023): 不训 reward model 的等价目标。
- 视频生成里的 [[RLHF]] 应用如 LongCat-Video 的 Multi-Reward GRPO。

## 相关概念

- [[PPO]]
- [[GRPO]]
- [[KL 散度]]
- [[Reward Model]]
