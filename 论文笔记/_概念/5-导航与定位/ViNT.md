---
type: concept
aliases: [ViNT, Visual Navigation Transformer]
---

# ViNT

## 定义
基于 Transformer 的通用视觉导航策略，在多机器人平台（差速轮、四足、无人机）上联合训练，实现零样本跨平台导航迁移。输入当前图像和目标图像，输出导航动作。

## 数学形式
$$a_t = \pi_\theta(o_t, g) \quad \text{where} \quad o_t, g \in \mathbb{R}^{H \times W \times 3}$$

通过对比学习训练 goal image encoder，测量当前-目标语义距离指导规划。

## 核心要点
1. 跨体态联合训练：多平台数据统一为归一化动作空间，实现跨体态泛化
2. 目标图像条件化：以目标图像（而非语言）作为导航目标，适用于 topological navigation
3. 与 NoMaD 等方法的比较：ViNT 关注跨平台，NoMaD 关注多模态探索

## 代表工作
- [[ViNT]]: Shah et al., 2023 (Berkeley + CMU)，原始论文
- [[NavWM]]: 将 ViNT 作为导航世界模型的对比 baseline

## 相关概念
- [[NoMaD]] — 同方向竞品
- [[GNM]] — 前作 General Navigation Model
- [[导航]] — 方向概念
