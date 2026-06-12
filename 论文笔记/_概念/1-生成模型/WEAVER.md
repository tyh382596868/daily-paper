---
type: concept
aliases: [WEAVER robot world model]
---

# WEAVER

## 定义
"Better, Faster, Longer" — CMU/Mila 提出的高效 robot manipulation 世界模型，同时优化生成质量、推理效率和长程能力。

## 数学形式
$$\hat{o}_{t+1:t+H} = f_\text{WEAVER}(o_{\leq t}, a_{\leq t}; \theta)$$

## 核心要点
1. 基于 Ctrl-World 控制注入方式改进 NFE（function evaluation 次数）
2. 在 DROID 大规模数据集上训练，具备长程预测能力
3. 提供 PnP（plug-and-play）规划接口，支持 policy evaluation/improvement/test-time planning
4. 用 FVD 评估生成质量

## 代表工作
- [[WEAVER]]: arXiv 2606.13672，CMU + Mila

## 相关概念
- [[WAM]]
- [[Ctrl-World]]
- [[FVD]]
- [[DROID]]
