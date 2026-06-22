---
type: concept
aliases: [Diffusion NFT, 扩散模型软更新, DiffusionNFT 软更新]
---

# DiffusionNFT

## 定义

扩散模型强化学习后训练中的软更新策略，通过 KL 正则化约束策略更新幅度，在提升奖励的同时防止模型偏离原始预训练分布过远（类比 RLHF 中的 KL penalty）。

## 核心要点

1. **软更新机制**: 每次参数更新为原始权重与新权重的加权混合，避免过激更新导致的模型崩溃
2. **KL 正则化**: 显式约束策略分布与参考分布之间的 KL 散度，维持生成质量稳定性
3. **渐进式调度**: 配合渐进式更新调度，在训练早期保守更新，后期逐步放开
4. **解耦优化窗口**: 将强化学习的优化窗口与 rollout 时域解耦，允许长时程 rollout 与短片段奖励评估分离

## 代表工作

- [[DreamXWorld]]: 在 RL 后训练阶段使用 DiffusionNFT 软更新 + KL 正则化，提升相机控制精度和视频质量

## 相关概念

- [[DMD]]: DreamX-World 蒸馏阶段使用的 Distribution Matching Distillation
- [[扩散模型]]: 基础生成框架
- [[自回归生成]]: RL 后训练中的 rollout 生成范式
