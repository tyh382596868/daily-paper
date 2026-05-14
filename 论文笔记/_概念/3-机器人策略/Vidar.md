---
type: concept
aliases: []
---

# Vidar

## 定义

Cascaded WAM，使用视频生成模型为机器人合成长程任务视频，再由动作解码器输出控制信号。强调 sample-efficient 真实数据扩展。

## 核心要点

1. **视频先验**: 利用预训练大视频模型的运动 / 物理先验。
2. **少样本机器人微调**: 在少量真实机器人数据上即可适配。

## 代表工作

- [[WAM-Survey]] 综述列为 Cascaded Explicit 代表。

## 相关概念

- [[UniPi]]
- [[World Action Model]]
- [[视频扩散模型]]
