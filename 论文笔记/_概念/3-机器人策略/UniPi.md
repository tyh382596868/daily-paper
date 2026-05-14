---
type: concept
aliases: [Universal Policy via Video]
---

# UniPi

## 定义

UniPi（Du et al., 2023）是 [[Cascaded WAM|Cascaded WAM]] 路线的开山之作：用文本条件视频扩散模型生成任务执行视频，再用学习式 [[逆动力学|IDM]] 从相邻帧回归动作。

## 核心要点

1. **两阶段**: Stage-1 text-conditioned spatiotemporal U-Net 扩散视频生成；Stage-2 卷积 IDM 回归动作。
2. **局限**: 单 pass 长程生成存在语义漂移和误差累积。
3. **影响**: 启发了 VLP、RoboEnvision、Gen2Act 等改进。

## 代表工作

- [[WAM-Survey]] 将其归类为 Cascaded-Explicit / Learned-Action 的标杆。

## 相关概念

- [[逆动力学]]
- [[视频扩散模型]]
- [[World Action Model]]
