---
type: concept
aliases: [全形态运动跟踪]
---

# Omni-bodied Motion Tracking

## 定义
跨越不同机器人形态（人形、四足、机械臂）的统一运动跟踪网络，无需为每种形态单独训练。

## 数学形式
$$a^{emb}_t = \pi_{\theta}(o_t, e_{emb}, \tau^{ref})$$
其中 $e_{emb}$ 为具身编码，$\tau^{ref}$ 为参考轨迹。

## 核心要点
1. 统一编码不同形态的关节状态和参考运动
2. 形态 embedding 处理具身差异
3. 同一网络权重适用于多种机器人

## 代表工作
- [[VENOM]]: 提出 Omni-bodied Motion Tracking 网络

## 相关概念
