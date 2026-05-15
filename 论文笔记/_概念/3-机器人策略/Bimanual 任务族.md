---
type: concept
aliases: [Bimanual 任务族, Bimanual tasks, 双臂任务族]
---

# Bimanual 任务族

## 定义

[[AliasBench]] 的第三类任务：双臂协同 / 交接物体，两臂在对称构型下角色（"给"vs"接"）可互换，仅凭当前帧无法判断当前是哪条手在主动。

## 核心要点

1. **典型例子**: Hand Over Roller（双臂交接擀面杖）
2. **混叠源**: 双臂对称姿态下视觉几乎一样
3. **解法**: 历史帧揭示哪条手最近在动
4. **任务数量**: 2 个，是 [[AliasBench]] 最难的一族（绝对值最低）

## 代表工作

- [[IntentVLA]]: 在 Bimanual 上从 5.5% 提升到 17.0%（+11.5 pp），仍是绝对值最难

## 相关概念

- [[AliasBench]]
- [[观测混叠]]
