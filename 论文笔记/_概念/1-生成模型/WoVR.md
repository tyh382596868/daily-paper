---
type: concept
aliases: [World-Policy Co-evolution, World of Visual Robotics]
---

# WoVR

## 定义

显式的**世界模型-策略协同进化**方法 — 不把 WM 视为固定模拟器，而是用策略 rollout 的失败案例反过来更新 WM，形成闭环。

## 数学形式

$$
\phi^{k+1} \leftarrow \text{UpdateWM}(\phi^k,\ \mathcal{D}_{\text{real}} \cup \mathcal{D}_{\text{policy}}(\pi_{\theta^k}))
$$

$$
\theta^{k+1} \leftarrow \text{UpdatePolicy}(\theta^k,\ \hat{\mathcal{D}}(\phi^{k+1}))
$$

## 核心要点

1. [[RobotWM-Survey]] Section 4.1 中"第二层"co-evolution 路线的代表
2. 解决了"WM 误差会污染策略"的核心问题 — 通过持续用真实策略数据修正 WM
3. 与 World-VLA-Loop、VLAW 同期，但 WoVR 提供了最清晰的形式化
4. 在长 horizon、稀有事件任务上优于固定 WM

## 代表工作

- Jiang et al., 2026: WoVR 原始论文

## 相关概念

- [[世界模型]]
- [[WMPO]]
- [[强化学习]]
- [[RobotWM-Survey]]
