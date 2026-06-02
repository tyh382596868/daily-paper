---
type: concept
aliases: [RAFT, Recurrent All-Pairs Field Transforms]
---

# RAFT

## 定义

ECCV 2020 提出的[[光流]]估计网络，全称 **Recurrent All-Pairs Field Transforms for Optical Flow**，是当前最常用的 dense optical flow estimator。核心思路是构建全对相关体积（all-pairs correlation volume），然后用 GRU 迭代精化流场。

## 核心设计

1. **All-Pairs Correlation**: 对两帧特征图计算所有像素对的相似度，得到 4D 相关体积
2. **GRU 迭代精化**: 12 个 [[GRU]] step 迭代更新光流估计
3. **多尺度查找**: 在不同尺度上从相关体积中采样
4. **端到端可微**: 整个 pipeline 可端到端训练

## 工程实践

- 推理时 GRU step 数可减少（4-6 次）以省 latency，精度略降
- 论文如 [[AHEAD]] 中将 RAFT 蒸馏到 6 次更新以满足 200 ms 实时预算
- 与 CoTracker3 等 sparse tracker 的对比：dense flow 信号更丰富但参数大

## 在机器人中的应用

- [[Im2Flow2Act]]: 用光流作为动作中间表示
- [[3DFlowAction]]: 3D 流场驱动动作
- [[AHEAD]]: 用 RAFT 估计 patch 速度/加速度作为世界模型条件

## 已知局限

- 反光、低纹理、快速旋转表面易失效
- 大位移（>32 px）需要金字塔结构
- 对相机抖动敏感

## 相关概念

- [[光流]]
- [[Im2Flow2Act]]
- [[3DFlowAction]]
- CoTracker3: 替代品，sparse point tracking
