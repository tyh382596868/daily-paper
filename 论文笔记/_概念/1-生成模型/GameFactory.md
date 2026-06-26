---
type: concept
aliases: [Game Factory, 游戏世界生成模型]
---

# GameFactory

## 定义
一类基于视频扩散/生成模型构建的游戏世界生成框架，能够合成 action-conditioned 的可交互游戏场景视频。

## 核心要点
1. 以游戏截图序列为训练数据，学习动态一致的视觉世界
2. 支持用户输入动作条件（键盘/手柄操作），生成对应的未来帧
3. 物理动力学为隐式学习，非显式建模
4. 代表了 "neural game engine" 这一研究范式

## 代表工作
- [[PhysEditWorld]]: 批评 GameFactory 只学隐式物理相关性，提出物理可编辑 WM 数据集
- [[GameGen-X]]: 同类工作，支持开放世界游戏生成

## 相关概念
- [[Action-Conditioned World Model]]
- [[Diffusion Model]]
- [[Cosmos]]
