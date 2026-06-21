---
type: concept
aliases: [世界模型目标定位]
---

# World Model Object Localization

## 定义
利用世界模型预测到达目标附近后的视觉观测，以预测未来观测与目标描述的一致性驱动导航。

## 数学形式
$$a_t^* = \arg\max_{a_t} \text{sim}(\hat{o}_{t+H|a_t}, l_{target})$$
其中 $\hat{o}$ 为世界模型预测的未来观测。

## 核心要点
1. 无需预建地图，支持开放词汇目标
2. 世界模型预测目标场景的视觉外观
3. 最大化预测观测与语言目标的相似度驱动探索

## 代表工作
- [[WoMAP]]: 提出 World Model Object Localization

## 相关概念
