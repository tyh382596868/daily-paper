---
type: concept
aliases: [多智能体视频世界模型]
---

# Multi-Agent Video World Model

## 定义
从单视角视频数据训练能够模拟多个智能体交互动态的视频世界模型。

## 数学形式
$$p(v_{1:T} | a^1_{1:T}, \ldots, a^N_{1:T}) = \prod_t p(v_t | v_{<t}, a^{1:N}_{\leq t})$$

## 核心要点
1. 克服单智能体世界模型的视角局限
2. 从单视角视频学习多智能体的相对运动
3. 应用于多机器人协作场景的仿真

## 代表工作
- [[MetaWorld]]: 提出多智能体视频世界模型（注意与 MuJoCo MetaWorld benchmark 区分）

## 相关概念
