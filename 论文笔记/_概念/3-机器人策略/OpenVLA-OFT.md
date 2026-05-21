---
type: concept
aliases: [OpenVLA-OFT, OpenVLA Optimized Fine-Tuning]
---

# OpenVLA-OFT

## 定义
[[OpenVLA]] 的优化微调（Optimized Fine-Tuning）版本，通过改进的微调配方（如并行解码、连续动作、L1 回归目标等）大幅提升 OpenVLA 在操作 benchmark 上的成功率与推理效率。

## 核心要点
1. 在 [[LIBERO]] 上平均成功率从原始 [[OpenVLA]] 的 0.76 提升到 0.91，是很强的微调基线。
2. 强调"微调配方"本身对 VLA 性能的巨大影响，而非更换 backbone。
3. 在 [[PAPO-VLA]] 实验中作为 [[LIBERO]] 与 [[RoboTwin]]2.0 上的对比基线。

## 代表工作
- 作为 [[PAPO-VLA]] 的对比基线（LIBERO 平均 0.91）。

## 相关概念
- [[OpenVLA]]
- [[VLA]]
- [[LIBERO]]
