---
type: concept
aliases: [CALM, Conditional Adversarial Latent Models]
---

# CALM: Conditional Adversarial Latent Models

## 定义
在 AMP 基础上加入潜在条件编码，使运动先验不仅能产生自然运动，还能根据潜在代码控制不同的运动技能和风格。

## 数学形式
$$r_{style}(s, s', z) = \log D(s, s', z) - \log(1 - D(s_{ref}, s'_{ref}, z))$$

其中 $z$ 是从先验 $p(z)$ 采样的潜在风格代码。

## 核心要点
1. 在 AMP 的基础上引入潜在空间 $z$ 实现多样化运动风格
2. 可以在不同运动风格之间进行插值和切换
3. 通过 conditional discriminator 将风格代码与动态对应
4. 常用作更复杂运动生成方法的基线

## 代表工作
- Tessler et al. (2023): CALM 原作，NeurIPS 2023
- [[T-GMP]]: 与 CALM 等对比，提出显式地形条件化

## 相关概念
- [[AMP]]
- [[T-GMP]]
- [[locomotion]]
