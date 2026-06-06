---
type: concept
aliases: [Residual Mimic, Humanoid Motion Distillation]
---

# ResMimic

## 定义
ResMimic 是一类把高自由度参考人体轨迹（通常是 [[SMPL]]）蒸馏到 humanoid 机器人控制策略的方法，核心思路是先训一个 base imitation policy 跟踪参考关节，再叠加一个 residual policy 修正 sim-to-real 差距与接触不一致。常用 [[PPO]] 作为后端 RL 算法。

## 核心要点
1. **两段式策略**：base policy 输出主要 torque/joint command，residual policy 输出修正量
2. **训练数据**：SMPL 参考轨迹（由 [[CHOIS]] / [[GENMO]] 等生成或来自 MoCap）
3. **奖励设计**：参考跟踪奖励（关节位置/速度）+ 接触奖励 + 平衡惩罚
4. **优势**：相比纯 imitation，residual head 提供修正空间，可缓解 morphology gap 与物理 mismatch

## 代表工作
- ResMimic 系列工作（humanoid imitation 文献中近 1-2 年的常见 baseline）
- [[GRAIL]] 把生成的 HOI 轨迹蒸馏成 humanoid 控制策略时使用 ResMimic 类方法

## 相关概念
- [[SMPL]]
- [[HOI]]
- [[PPO]]
- [[MPJPE]]
