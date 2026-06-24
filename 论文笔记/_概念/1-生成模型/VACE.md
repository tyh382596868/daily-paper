---
type: concept
aliases: [VACE, Video-guided Action-Conditioned Editing]
---

# VACE

## 定义
动作条件化视频编辑/生成方法，输入视频片段和动作/控制信号，输出符合动作语义的视频编辑结果，广泛用作交互式世界模型的 baseline。

## 核心要点
1. 动作条件化：视频生成以机器人动作或控制信号为条件，保证动作一致性
2. 局部编辑：只修改与动作相关的区域，保留背景一致性
3. 在 RoboTwin 等机器人仿真场景中验证

## 代表工作
- [[VACE]]: 原始论文（Alibaba DAMO Academy 相关工作）
- [[IOI]]: 将 VACE 作为交互式世界模型的对比 baseline

## 相关概念
- [[交互式世界模型]] — 应用场景
- [[视频扩散模型]] — 技术基础
- [[ControlNet]] — 条件控制机制
