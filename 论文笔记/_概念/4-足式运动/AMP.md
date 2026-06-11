---
type: concept
aliases: [AMP, Adversarial Motion Priors, 对抗运动先验]
---

# AMP: Adversarial Motion Priors

## 定义
利用对抗训练将动作捕捉数据中的运动风格迁移到 RL 策略中，使机器人在完成任务的同时保持自然的运动风格。

## 数学形式
$$r_{style} = \log D(s, s') - \log(1 - D(s_{ref}, s'_{ref}))$$

其中 $D$ 是判别器，区分策略生成的状态转换和参考动作数据中的转换。

## 核心要点
1. 不需要人工设计风格奖励，判别器自动学习风格约束
2. 利用现有动捕数据库（如 AMASS）作为运动参考
3. 与 RL 任务奖励结合，同时优化任务完成度和运动自然度
4. 是生成式运动先验方法的重要基线

## 代表工作
- Peng et al. (2021, SIGGRAPH): AMP 原作，DeepMimic 的后续
- [[T-GMP]]: 与 AMP、CALM 等对比，提出地形条件化的生成先验

## 相关概念
- [[CALM]]
- [[T-GMP]]
- [[locomotion]]
