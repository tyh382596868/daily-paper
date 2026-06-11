---
type: concept
aliases: [FlowPRO framework, 无奖励 VLA 后训练]
---

# FlowPRO

## 定义

Reward-Free Reinforced Fine-Tuning of Flow-Matching VLAs via Proximalized Preference Optimization，一个无需显式奖励函数的离线偏好优化框架，专门用于 flow-matching VLA 模型的后训练。

## 核心要点

1. **算法侧**: [[RPRO]] 损失（对比项 + 近端正则化）稳定优化，消除 [[Flow-DPO]] 的奖励黑客
2. **数据侧**: 干预-回滚范式 + [[平滑插值]] 产生密集偏好监督，无需单独录制失败轨迹
3. **验证**: Tencent Robotics X 在 Dobot XTrainer 双臂平台 4 任务上超越 DAgger/TPO/π₀.6* 等基线

## 代表工作

- arXiv 2606.05468（[[FlowPRO]]论文本体）

## 相关概念

- [[RPRO]]（核心损失函数）
- [[平滑插值]]（数据增强模块）
- [[Flow-DPO]]（直接前驱方法）
- [[π₀]]（使用的 base policy）
- [[VLA 后训练]]（所属研究方向）
