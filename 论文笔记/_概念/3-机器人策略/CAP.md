---
type: concept
aliases: [CAP robot, Closed-loop Action Planning]
---

# CAP

## 定义

CAP（Closed-loop Action Planning）是一种机器人操作基线方法，采用闭环动作规划策略，在 GeneralVLA-2 评测中作为零样本对比系统，在部分标准操作任务上表现良好，但在精细操作任务上存在局限。

## 核心要点

1. **零样本能力**: 支持无训练数据的零样本操作规划
2. **局限性**: 在 Play_jenga、Sort_mustard、Open_wine 等复杂操作任务上成功率为 0
3. **真实世界**: Move_spray_bottle 任务上仅 6.67% 成功率，Open_drawer 为 0%

## 代表工作

- [[GeneralVLA2]] 中作为对比基线

## 相关概念

- [[VLA]]
- [[VoxPoser]]
