---
type: concept
aliases: [CFG, classifier-free guidance, 无分类器引导]
---

# Classifier-Free Guidance

## 定义

**Classifier-Free Guidance (CFG)** 是扩散 / 流模型中常用的推理时引导技术：训练时随机丢弃条件 $c$ 学习一个 unconditional 分支，推理时把 conditional 和 unconditional 预测线性外推，提升生成与条件的对齐度：

$$
\hat{v}_\theta(z_\tau, \tau, c) = (1 + w) \, v_\theta(z_\tau, \tau, c) - w \, v_\theta(z_\tau, \tau, \varnothing)
$$

## 核心要点

1. **训练**: 随机概率（典型 10%）把条件 $c$ 替换为空 $\varnothing$
2. **推理**: $w$（guidance scale）越大，与 prompt 对齐越强但多样性下降
3. **典型值**: 文生图 $w \approx 7.5$，文生视频 $w \approx 5$
4. **vs Classifier Guidance**: 不需要额外训练分类器

## 代表工作

- [[Cosmos3]]: 扩散塔采样使用 CFG
- 几乎所有 [[Diffusion Model]] / [[Flow Matching]] 模型

## 关联

- [[Diffusion Model]]
- [[Flow Matching]]
- [[Rectified Flow]]
