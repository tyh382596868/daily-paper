---
type: concept
aliases: [General Navigation Model, 通用导航模型]
---

# GNM（General Navigation Model）

## 定义
GNM 是 Shah et al. (2023) 提出的通用视觉导航模型，在多种机器人平台和多种室内/室外环境数据上联合训练，学习可跨 embodiment 迁移的导航策略。

## 数学形式
目标条件策略：$\pi_\theta(a_t | o_t, o_g)$

其中 $o_t$ 为当前视觉观测，$o_g$ 为目标图像。距离预测：
$$\hat{d} = f_\phi(o_t, o_g) \in \mathbb{R}$$

## 核心要点
1. 跨平台训练（轮式/四足/无人机）提升泛化能力
2. 目标图像作为条件，不依赖显式地图
3. NoMaD 在 GNM 基础上加入扩散策略，支持多模态动作分布
4. ViNT 进一步扩展到拓扑图导航

## 代表工作
- Shah et al. (2023): GNM 原论文
- [[NoMaD]]: 扩散策略版本的 GNM
- [[ViNT]]: 拓扑图增强的 GNM

## 相关概念
- [[SLAM]]
- [[Navigation Score]]
- [[SPOC]]
