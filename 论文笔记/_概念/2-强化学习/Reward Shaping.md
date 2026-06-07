---
type: concept
aliases: [Reward Shaping, 奖励塑形, 奖励工程]
---

# Reward Shaping

## 定义

**Reward Shaping**: 在原始稀疏 reward 之外，**手工设计**或**自动生成**额外的密集子 reward（如速度、姿态、jerk、距离、安全余裕等），把它们加权求和组成最终 reward，让策略在训练早期能从更稠密的信号中学习。

## 数学形式

$$
r(s, a) = r_{task}(s, a) + \sum_i \alpha_i \cdot r_{shape, i}(s, a)
$$

理论上，若 shape term 形如 $\gamma \Phi(s') - \Phi(s)$（**potential-based shaping**），不会改变最优策略；其他形式则可能引入偏差。

## 优劣

**优**:
- 显著加速稀疏 reward 任务收敛
- 把人类先验（"贴墙危险"、"jerk 大不安全"）注入策略

**劣**:
- 权重调参代价大（[[MAD]] 用 8 项 + 8 个权重 $\alpha$）
- 可能引发 [[Reward Hacking|reward hacking]]——策略钻 shape term 漏洞，不做真任务
- 跨任务/跨环境不易迁移

## 典型组件（机器人飞行/操作）

| 类别 | 例子 |
|------|------|
| 存活 | 每步常数 reward |
| 任务 | 距目标距离、目标速度跟踪 |
| 安全 | 碰撞惩罚、最近障碍距离 |
| 平滑 | jerk、力变化、姿态变化 |
| 进度 | 沿期望方向投影速度 |

## 代表工作

- [[MAD]]：8 项加权 reward（存活/碰撞/速度/位置/高度/接近/规避/jerk）
- 大量 sim2real RL 工作

## 关联概念

- [[强化学习]]
- [[Reward Hacking]]
- [[PPO]]
- [[SHAC]]
