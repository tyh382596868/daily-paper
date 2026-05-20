---
type: concept
aliases: [VPT, Video PreTraining, 视频预训练]
---

# VPT

## 定义
Video PreTraining（视频预训练）是 OpenAI 提出的范式：先用少量带动作标注的数据训一个逆动力学模型（IDM）给海量无标注网络视频打伪动作标签，再用行为克隆在这些伪标注视频上预训练一个通用 agent。

## 核心要点
1. 解决「网络视频海量但无动作标签」的问题——IDM 把视频补上动作标签。
2. 在 Minecraft 上验证：预训练 agent 可零样本完成多种任务，并作为微调/RL 的强初始化。
3. 是「从被动视频学习可执行策略」这条路线的奠基工作之一。

## 代表工作
- [[PROWL]]: 在 Minecraft（MineRL / BASALT）系基准上以 VPT 为基础世界模型/策略评估对抗课程

## 相关概念
- [[强化学习]]
- [[逆动力学]]
- [[世界模型]]
