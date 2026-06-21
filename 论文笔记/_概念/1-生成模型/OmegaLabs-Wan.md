---
type: concept
aliases: [Wan 视频模型]
---

# OmegaLabs-Wan

## 定义
OmegaLabs 开发的开源视频生成基础模型，支持高质量文本/图像到视频生成，是多个交互世界模型的底层生成器。

## 数学形式
$$p(v | c) = \text{FlowMatching}_{\theta}(v, c)$$

## 核心要点
1. 基于 Flow Matching 的高效视频生成
2. 开源权重支持社区构建世界模型
3. 与 [[BiWM]] 等交互世界模型工作密切相关

## 代表工作
- [[BiWM]]: 基于 Wan 视频模型构建交互世界模型底座

## 相关概念
