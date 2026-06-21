---
type: concept
aliases: [动画驱动灵巧操作]
---

# Animation-Based Dexterous Manipulation

## 定义
将灵巧手操作问题重新表述为动画生成问题，先生成关键帧 pose，再通过运动规划和 RL 生成完整轨迹。

## 数学形式
$$\pi^* = \arg\min_\pi \sum_t \|q_t - q_t^{\text{ref}}\|$$

## 核心要点
1. 用程序化方法生成多样化的抓取关键帧而无需真实数据
2. 粗到细：关键帧 → 完整轨迹
3. 支持零样本 sim-to-real 迁移

## 代表工作
- [[Mana]]: 提出 animation-based 灵巧操作框架

## 相关概念
