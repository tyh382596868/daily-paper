---
type: concept
aliases: [Real2Sim2Real Tactile Policy]
---

# TwinTac (Real2Sim2Real Tactile Policy Learning)

## 定义
上科大/BIGAI 提出的盲触觉抓取框架：Real2Sim2Real 流水线学习纯触觉（无视觉）灵巧手抓取策略。

## 核心要点
1. Real2Sim：真实 FSR 触觉数据 → Kinematic-Tactile 仿真建模
2. Sim2Real：ACT + PPO 混合训练 → LEAP 灵巧手部署
3. "盲抓取"：无视觉，纯触觉驱动
4. v2 版本迭代，真实机器人验证

## 代表工作
- [[TwinTac]]: arXiv 2606.11767

## 相关概念
- [[ACT (Action Chunking Transformer)]]
- [[PPO]]
- [[LEAP]]
