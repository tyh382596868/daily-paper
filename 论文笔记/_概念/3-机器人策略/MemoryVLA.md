---
type: concept
aliases: [MemoryVLA]
---

# MemoryVLA

## 定义

一种把 **token-level working memory** 集成进 [[VLA]] 的方法：通过额外的 memory token 把历史信息编码在 transformer 序列中，无需显式的关键帧库。

## 核心要点

1. token 级别工作记忆，容量受 context window 限制
2. 端到端训练，无需启发式关键帧抽取
3. 在真正长程任务（>1000 步）上表现受限，因为 token 容量不够
4. 推理时直接通过自注意力查询历史

## 代表工作

- [[RoboMemArena]] 中作为对比基线，平均 TSR 15.0%
- [[EvoScene-VLA]] 相关工作对比：MemoryVLA 是 token-level 工作记忆，本方法是几何信念递归

## 相关概念

- [[VLA]]
- [[关键帧记忆库]]
- [[MemER]]
- [[SOMA]]: 走"显式空间 3D 记忆"路线，与 MemoryVLA 的 token 时间记忆形成正交补充
