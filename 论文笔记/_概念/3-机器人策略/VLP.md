---
type: concept
aliases: [Video Language Planning]
---

# VLP

## 定义

Video Language Planning（VLP）：在 [[UniPi]] 基础上引入 VLM 做分层子动作生成和树搜索价值评分，缓解长程视频生成的误差累积。

## 核心要点

1. **分层规划**: VLM 先把任务拆成子目标，再逐段生成视频。
2. **Tree search**: 每段用价值函数评分，约束单段误差。
3. 属于 [[Cascaded WAM|Cascaded WAM]] 的 Explicit / Learned-Action 子类。

## 代表工作

- [[WAM-Survey]] 综述中作为 Cascaded WAM 改进 UniPi 的代表。

## 相关概念

- [[UniPi]]
- [[World Action Model]]
- [[VLA]]
