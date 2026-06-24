---
type: concept
aliases: [RVT, Robotic View Transformer, RVT-2]
---

# RVT

## 定义
多视角 Transformer 机器人操作策略，通过虚拟相机生成多角度图像，在 3D 特征空间中预测关键点位置和动作。在 RLBench benchmark 上取得 SoTA 水平。

## 数学形式
输入 $N$ 视角图像 $\{I_k\}$，通过 Transformer 预测 3D heatmap $H \in \mathbb{R}^{H \times W \times N}$，再解码为末端执行器姿态。

## 核心要点
1. 虚拟视角渲染：从固定相机集合生成一致的多角度视图，减少视角歧义
2. 重投影注意力：特征跨视角交叉注意力聚合 3D 结构信息
3. RVT-2 改进：分辨率更高，引入 recurrent 机制处理多步操作

## 代表工作
- [[RVT]]: Goyal et al., 2023，原始方法
- [[G³VLA]]: 将 RVT 作为几何先验 VLA 的对比 baseline

## 相关概念
- [[OpenVLA]] — 同类操作策略
- [[3D视觉]] — 技术基础
- [[机器人操作]] — 应用方向
