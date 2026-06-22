---
type: concept
aliases: [UnifiedReward, Unified Video Reward]
---

# UnifiedReward

## 定义
统一的视频质量奖励模型，对视频生成质量（视觉质量、时序一致性、文本对齐等）进行多维度综合打分。

## 核心要点
1. 多维度评分：视觉质量、运动合理性、文本-视频对齐
2. 可用于 RLHF 微调视频生成模型，也可用于 BoN 推理时搜索
3. PRISM 论文中以此作为 baseline，在扩散中间态打分比完整解码后打分更高效

## 代表工作
- [[PRISM]]：2606.20310

## 相关概念
- [[RLHF]]
- [[Video Diffusion Model]]
- [[BoN]]
- [[VideoScore2]]
