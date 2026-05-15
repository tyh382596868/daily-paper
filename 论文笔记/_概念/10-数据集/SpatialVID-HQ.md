---
type: concept
aliases: [SpatialVID-HQ, SpatialVID]
---

# SpatialVID-HQ

## 定义

一个针对相机控制视频生成的大规模真实视频数据集，HQ 版本提供高质量的相机轨迹与位姿标注，常用于训练带 6-DoF 相机条件的视频扩散模型。

## 核心要点

1. 真实拍摄视频，覆盖室内、户外、城市等多种场景
2. 在 SANA-WM 流水线中用改造版 [[VIPE]]（+ [[Pi3X]] / MoGe-2）二次标注度量尺度位姿
3. 主要为 10 秒短片段，是相机控制学习的核心来源
4. 与 OmniWorld、Sekai、MiraData 共同构成训练混合

## 代表工作

- [[SANA-WM]]: 158,369 段 10s 片段，是训练数据中规模最大的来源

## 相关概念

- [[OmniWorld]]
- [[VIPE]]
- [[10-数据集]]
