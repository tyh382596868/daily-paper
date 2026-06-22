---
type: concept
aliases: [BoN, Best-of-N, Best-of-N Sampling]
---

# BoN（Best-of-N Sampling）

## 定义
推理时 scaling 策略：生成 N 个候选输出，用 reward model 打分后选择最优者，以计算换质量。

## 核心要点
1. 无需重训，推理时即插即用
2. reward model 质量决定 BoN 效果上限
3. PRISM 将 BoN 用于视频生成：在扩散中间态打分，避免完整解码 N 次

## 代表工作
- [[PRISM]]：preference representation in intermediate states of video diffusion（2606.20310）

## 相关概念
- [[RLHF]]
- [[Video Diffusion Model]]
- [[UnifiedReward]]
