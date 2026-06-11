---
type: concept
aliases: [SMPL-X, Expressive Body Model, 全身参数化人体模型]
---

# SMPL-X (Expressive Body Capture Model)

## 定义
SMPL-X 是 SMPL 的扩展版本，将身体（SMPL）、手部（MANO）和面部（FLAME）统一为一个可微的全身参数化人体模型，支持精细的手指和面部表情建模，是人形机器人全身动作生成与 4D HOI 重建的主流表示。

## 数学形式

$$
\mathbf{M}(\boldsymbol{\theta}, \boldsymbol{\beta}, \boldsymbol{\psi}) = W(T(\boldsymbol{\beta}, \boldsymbol{\theta}, \boldsymbol{\psi}), J(\boldsymbol{\beta}), \boldsymbol{\theta}, \mathcal{W})
$$

其中：
- $\boldsymbol{\theta}$：姿态参数（身体 + 手部 + 面部关节旋转）
- $\boldsymbol{\beta}$：形状参数（体型，10 维）
- $\boldsymbol{\psi}$：面部表情参数（FLAME 表情空间）
- $T(\cdot)$：T-pose 模板 + blend shapes 变形函数
- $J(\boldsymbol{\beta})$：从形状参数回归的关节位置
- $W(\cdot)$：Linear Blend Skinning 函数
- $\mathcal{W}$：蒙皮权重矩阵

## 核心要点
1. **全身统一**: 同时建模躯体、双手（45 个手部关节）和面部，比 SMPL 更适合交互场景
2. **可微参数化**: 支持端到端优化，可直接作为 HOI 重建的优化变量
3. **与机器人对接**: 常通过 [[GMR]] 等重定向方法将 SMPL-X 轨迹映射到机器人骨骼
4. **估计工具链**: HMR2 + ViTPose + VIMO + HMR4D（身体），WiLoR / MANO 估计器（手部）

## 代表工作
- 原论文：Pavlakos et al., "Expressive Body Capture: 3D Hands, Face, and Body from a Single Image", CVPR 2019
- [[GRAIL]]：从 Kling 生成的视频中重建 SMPL-X 轨迹，再重定向到 Unitree G1
- [[CHOIS]]、[[GENMO]]：基于 SMPL-X 的人-物交互动作生成
- [[DeepMimic]]：早期基于参考动作追踪的前置工作

## 相关概念
- [[SMPL]]: 基础人体模型，SMPL-X 的前身
- [[HOI]]: SMPL-X 是 4D HOI 重建的标准人体表示
- [[GMR]]: 将 SMPL-X 轨迹重定向到人形机器人骨骼
- [[FoundationPose]]: 配合 SMPL-X 用于物体 6-DoF 估计
