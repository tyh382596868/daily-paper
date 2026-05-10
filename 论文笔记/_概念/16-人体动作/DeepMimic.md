---
type: concept
aliases: [DeepMimic]
---

# DeepMimic

## 定义
通过强化学习模仿人体参考动作的物理角色控制方法，让仿真角色在保持物理可行性的同时复现人体运动风格。

## 核心要点
1. 参考状态初始化（RSI）：从参考动作的随机时刻开始训练，加速收敛
2. 用 PPO 最大化参考状态相似度 reward
3. 是后续 PHC、ProtoMotions 等工作的基础

## 代表工作
- Peng et al., 2018: DeepMimic 原始论文（SIGGRAPH）

## 相关概念
- [[AMASS]]
- [[PPO]]
- [[ReActor]]
