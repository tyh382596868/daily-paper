---
type: concept
aliases: [RoboBrain 2, Robotics VLM]
---

# RoboBrain2

## 定义
RoboBrain2 是面向 robotics / embodied 场景的多模态大模型（VLM），针对真实机器人任务做了大规模指令微调，具备空间感知、任务规划、affordance 推理等能力，常作为 embodied benchmark 上的强 baseline。

## 核心要点
1. **目标领域**：机器人操作、导航、任务规划等真实世界任务
2. **训练数据**：robotics-flavored instruction tuning（操作演示、任务分解、空间问答等）
3. **能力侧重**：affordance、空间关系、长程任务分解
4. **角色**：在 embodied benchmark（如 [[OVO-S-Bench]]）里常被列为开源 baseline

## 代表工作
- RoboBrain 系列论文（北智院 BAAI 等团队）
- 应用：被 [[OVO-S-Bench]] / [[VeBrain]] 等 benchmark / 模型做对比

## 相关概念
- [[InternVL]]
- [[VeBrain]]
- [[VLA]]
