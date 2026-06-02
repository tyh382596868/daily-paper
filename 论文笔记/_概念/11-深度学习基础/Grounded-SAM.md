---
type: concept
aliases: [Grounded SAM, Grounded-Segment-Anything]
---

# Grounded-SAM

## 定义

将 Grounding-DINO（开放词汇目标检测）与 SAM（Segment Anything Model）级联起来的开放词汇分割流水线：给定一句文本（如"the red marker"），检测出 bbox 后让 SAM 输出像素级 mask。

## 核心要点

1. 开放词汇 — 不需要预先定义类别，任意文本即可分割
2. 输出像素级 mask，常用于自动化数据处理（抠物体、抠人、抠机器人）
3. 在机器人数据合成里常作为「物体先验提取」的第一步
4. 计算成本较高但准确率高

## 代表工作

- [[RoboDream]]: 用 Grounded-SAM 抠出任务相关物体作为物体先验

## 相关概念

- [[SAM]]
- [[OmniPaint]]
- [[Compositional Conditioning]]
