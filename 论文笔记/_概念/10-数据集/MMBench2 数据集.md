---
type: concept
aliases: [MMBench2, mmbench2, Multi-Modal Benchmark 2]
---

# MMBench2 数据集

## 定义

用于评估视觉[[世界模型]]质量的大规模 benchmark，提供真实动作标注、奖励信号和可运行仿真器，覆盖 10 个连续控制领域，支持幻觉检测与缓解研究。

## 核心要点

- **规模**: 427 小时，65,600 条轨迹，23M 帧（224×224），210 个任务
- **领域**: DMControl、DMControl-Extended、Meta-World、ManiSkill3、MuJoCo、MiniArcade（Pygame）、Box2D、RoboDesk、OGBench、Atari（共 10 个）
- **数据混合**: 专家策略 + 人类游玩 + 随机策略
- **关键特性**: 提供真实动作、奖励、在线仿真器——现有视频数据集通常缺失

## 代表工作

- [[MMBench2]]：提出该数据集的论文（Nicklas Hansen, Xiaolong Wang, 2026）

## 相关概念

- [[世界模型]]
- [[视频分词器]]
- [[动态模型]]
- [[Model-Based RL]]

## 链接

- Dataset: https://huggingface.co/datasets/nicklashansen/mmbench2
- Models: https://huggingface.co/nicklashansen/mmbench2-models
