---
type: concept
aliases: [ViVLA, Video-conditioned VLA]
---

# ViVLA

## 定义

ViVLA 是一种视频条件化视觉-语言-动作模型（VLA），将演示视频作为条件输入，通过视频理解增强 VLA 对 demo-conditioned 任务的理解能力。

## 核心思路

- 在预训练 VLA 骨干基础上引入视频条件化机制
- 演示视频特征通过交叉注意力或 token 拼接方式融入 VLA 的视觉-语言表示
- 兼顾语言指令和视频演示的双重条件

## 核心要点

1. 基于现有 VLA 骨干的轻量化视频条件化扩展
2. 继承 VLA 的语言理解能力，同时新增演示视频理解
3. 缺乏显式空间轨迹监督，精度受限

## 代表工作

- [[SeeTraceAct]]: 作为视频条件化 VLA 基线，SeeTraceAct 在 RoboCasa-DC 各评测设置上超越 ViVLA

## 相关概念

- [[Demo-Conditioned VLA]]
- [[VLA]]
- [[Cross-Embodiment Learning]]
