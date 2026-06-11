---
type: concept
aliases: [Vid2Robot, Video-conditioned Policy]
---

# Vid2Robot

## 定义

Vid2Robot 是一种端对端视频条件化机器人策略学习方法，利用跨注意力 Transformer 将演示视频直接映射为机器人动作，无需显式技能提取或轨迹监督。

## 核心思路

- 输入：任务演示视频 + 当前相机观测
- 处理：跨注意力机制融合演示视频特征与当前观测特征
- 输出：机器人动作序列

## 核心要点

1. 纯端到端学习，不依赖显式轨迹标注
2. 跨注意力 Transformer 架构桥接演示与执行
3. 缺乏显式空间定位监督，在小目标定位任务上精度有限

## 代表工作

- [[SeeTraceAct]]: 作为 demo-conditioned VLA 对比基线，SeeTraceAct 通过可见性感知轨迹监督超越 Vid2Robot 精度

## 相关概念

- [[Demo-Conditioned VLA]]
- [[Cross-Attention]]
- [[Cross-Embodiment Learning]]
