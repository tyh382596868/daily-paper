---
type: concept
aliases: [MLLM Judge, MLLM-based Scoring, S_mllm]
---

# MLLM评分

## 定义

MLLM 评分是利用多模态大语言模型（[[MLLM]]）作为自动评判员，对视频生成质量进行细粒度语义评估的方法，在 [[WorldOlympiad]] 中同时用于物理合规性判断和交互保真度的三层评分。

## 数学形式

$$
S_{\mathrm{chunk}} = \frac{1}{5T}\sum_{i=1}^{T}a_i, \quad S_{\mathrm{trans}} = \frac{1}{5(T-1)}\sum_{i=1}^{T-1}b_i, \quad S_{\mathrm{global}} = \frac{g}{5}
$$

$$
S_{\mathrm{mllm}} = \frac{1}{3}(S_{\mathrm{chunk}} + S_{\mathrm{trans}} + S_{\mathrm{global}})
$$

## 核心要点

1. **三层粒度**：Chunk 级（局部对齐）、Transition 级（边界平滑）、Global 级（长时一致性）。
2. **5 分制**：MLLM 对每个评测单元打 0-5 分，归一化到 [0,1]。
3. **物理赛道应用**：相关性分（Relevance）+ 合规分（Compliance）判断物理规律遵守情况。
4. **人类对齐**：与人类偏好排名的 Spearman 相关系数 ρ=0.95。

## 代表工作

- [[WorldOlympiad]]: 大规模使用 MLLM 评分替代人工标注实现可扩展评测

## 相关概念

- [[MLLM]]
- [[CLIP语义对齐]]
- [[交互保真度评分]]
