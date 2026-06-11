---
type: concept
aliases: [部分可观测马尔可夫决策过程, Partially Observable Markov Decision Process]
---

# POMDP

## 定义
部分可观测马尔可夫决策过程：智能体无法直接观测真实状态 $s$，只能通过观测 $o$ 间接感知，因此需要维护对状态的概率分布——**信念状态(belief state)**——并基于信念做决策。

## 数学形式

信念状态及其序贯更新（贝叶斯滤波）：

$$
b(s^t) = P(s^t \mid o^{1:t}), \qquad b(s^{t+1}) \propto \sum_{s^t} P(s^{t+1}\mid o^{t+1}, s^t)\, b(s^t)
$$

## 核心要点
1. 由七元组 $(\mathcal{S}, \mathcal{A}, T, R, \Omega, O, \gamma)$ 定义：状态、动作、转移、奖励、观测、观测模型、折扣。
2. 关键思想：在**信念空间**而非状态空间上求最优策略 $\pi(b)$；信念空间是连续高维的。
3. 新观测到来时按贝叶斯规则更新信念（filtering），是序贯感知的核心。
4. 在具身 AI 里，"世界模型"可被看作维护并更新对未观测 3D 环境的信念。

## 代表工作
- [[3D-Belief]]: 用显式 3D 高斯场景实例化 POMDP 信念，对未观测区域采样多假设并序贯更新
- 经典：PBVI / SARSOP（point-based value iteration）、QMDP

## 相关概念
- [[强化学习]]
- [[世界模型]]
- [[场景级 3D 扩散]]
