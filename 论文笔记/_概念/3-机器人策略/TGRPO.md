---
type: concept
aliases: [TGRPO, Trajectory-wise GRPO]
---

# TGRPO（Trajectory-wise GRPO）

## 定义
一种把 [[GRPO]] 用于 [[VLA]] 模型后训练的方法：对同一指令采样一组轨迹，用轨迹级 reward 计算组相对优势来更新策略。

## 核心要点
1. 把 LLM 推理训练中的 [[GRPO]] 范式迁移到机器人操作的 VLA 后训练。
2. 局限：对一条轨迹内的所有动作一视同仁，用同一个轨迹级优势更新每一步，无法区分关键决策与稠密执行动作。
3. 是 [[PAPO-VLA]] 的主要改进对象与对比基线。

## 代表工作
- 作为 [[PAPO-VLA]] 在 [[LIBERO]] 上的对比基线（平均成功率 0.81）。

## 相关概念
- [[GRPO]]
- [[VLA]]
- [[规划动作]]
