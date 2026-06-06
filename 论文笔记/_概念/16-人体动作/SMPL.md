---
type: concept
aliases: [Skinned Multi-Person Linear, SMPL 人体模型]
---

# SMPL (Skinned Multi-Person Linear Model)

## 定义
SMPL 是 MPI 提出的参数化人体网格模型，把人体形态拆成低维的 shape 参数 $\beta \in \mathbb{R}^{10}$ 和 pose 参数 $\theta \in \mathbb{R}^{24 \times 3}$（24 个关节的轴角），通过线性 blend skinning 渲染出 6890 个顶点的三角网格。是 humanoid 动作生成 / 人体重建 / motion capture 的事实标准。

## 数学形式
给定 shape 参数 $\beta$ 和 pose 参数 $\theta$：
$$M(\beta, \theta) = W\!\Big(T_P(\beta, \theta),\ J(\beta),\ \theta,\ \mathcal{W}\Big)$$

其中：
- $T_P(\beta, \theta) = \bar{T} + B_S(\beta) + B_P(\theta)$ 是 rest-pose 模板 + shape blend shape + pose blend shape
- $J(\beta)$ 是从 shape 回归出来的关节位置
- $\mathcal{W}$ 是 blend skinning 权重，$W(\cdot)$ 是 linear blend skinning 函数

## 核心要点
1. **可微**：所有过程都是可微的，能直接嵌入神经网络做 end-to-end 训练
2. **变体**：SMPL-X 加了脸/手（FLAME + MANO），SMPL-H 加了手，STAR 修了网格 artifact
3. **真机映射**：humanoid 机器人控制里常先合成 SMPL 轨迹，再用 retargeting 映射到机器人关节
4. **数据集生态**：AMASS、3DPW、Human3.6M、MotionX 全都以 SMPL 为底层表示

## 代表工作
- 原论文：Loper et al., "SMPL: A Skinned Multi-Person Linear Model", SIGGRAPH Asia 2015
- [[GRAIL]]：用 SMPL 合成 human-object interaction 轨迹再蒸馏到 humanoid
- [[CHOIS]] / [[GENMO]]：基于 SMPL 的 human-object motion generation
- [[ResMimic]]：把 SMPL 轨迹蒸馏成 humanoid 控制策略

## 相关概念
- [[HOI]]
- [[MPJPE]]
- [[FoundationPose]]
