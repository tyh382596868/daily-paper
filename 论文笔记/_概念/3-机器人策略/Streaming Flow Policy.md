---
type: concept
aliases: [SFP, Streaming Flow Policy, 流式流策略]
---

# Streaming Flow Policy

## 定义

Streaming Flow Policy（SFP）是一种将整条动作轨迹视为连续流轨迹、以流匹配方式建模的机器人策略生成方法，改善了相邻 chunk 之间的连续性。

## 核心要点

1. **流式轨迹建模**: 不以固定 chunk 为单位，而是将动作序列视为连续时间流
2. **平滑性优于 DP**: 相较于标准 [[Diffusion Policy]] 有一定平滑性改善（LDLJ 较高）
3. **多模态弱**: 在 FAFM 对比实验中，SFP 在多模态覆盖方面不足（仅 3 条不同路径）
4. **无频率异构处理**: 未解决异构控制频率引起的可识别性失败问题

## 代表工作

- [[FAFM]]（Guo et al., 2026）: 以 SFP 为基线对比，FAFM 在成功率和多模态上均优于 SFP
- Jiang et al., 2025: "Streaming Flow Policy: Action Trajectories as Flow Trajectories"

## 相关概念

- [[Flow Matching]]
- [[Action Chunking]]
- [[Diffusion Policy]]
- [[FAFM]]
