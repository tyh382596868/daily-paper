---
type: concept
aliases: [HaMeR, Hand Mesh Recovery]
---

# HaMeR

## 定义
基于 Transformer 的单目手部网格恢复方法，从 RGB 图像估计 MANO 参数手部形状和姿态（3D 网格 + 关节角度），是当前手部重建的 SoTA 之一。

## 数学形式
$$(\theta_{hand}, \beta_{hand}) = f_{HaMeR}(I_{crop})$$

其中 $\theta_{hand}$ 为 MANO 姿态参数，$\beta_{hand}$ 为形状参数，$I_{crop}$ 为手部检测框裁剪图像。

## 核心要点
1. ViT 骨干：大规模预训练 ViT 提取手部特征，比 CNN 更鲁棒
2. 弱监督训练：利用 pseudo-GT 和野外数据的 2D 关键点监督
3. 与 MANO 模型结合：输出可解释的参数化手部模型，支持动力学仿真

## 代表工作
- [[HaMeR]]: Pavlakos et al., 2024，原始论文
- [[Supervise-What-Survives]]: 用 HaMeR 从合成视频重建手部几何，过滤伪动作噪声

## 相关概念
- [[MANO]] — 参数化手部模型
- [[SMPL]] — 同系列身体模型
- [[ViTPose]] — 同类姿态估计框架
