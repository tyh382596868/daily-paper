---
type: concept
aliases: [OpenVid-1M, OpenVid Dataset]
---

# OpenVid

## 定义

大规模开源视频生成训练数据集（约 1M 视频片段），带文本描述，是开源 T2V / TI2V 模型预训练的常用数据来源之一。

## 数学形式

不适用。

## 核心要点

1. 量大、覆盖面广、带 caption；
2. **不含精确相机参数**，单独使用难以训练相机可控模型；
3. 常与 [[WorldPlay]] 等"按轨迹合成视频"工具配合：把 OpenVid 当作图像源，由 WorldPlay 生成相机轨迹明确的视频对。

## 代表工作

- [[minWM]]: 把 OpenVid 当做图像源 + 用 [[WorldPlay]] 合成带轨迹的训练视频。

## 相关概念

- [[DL3DV]]
- [[SpatialVID-HQ]]
- [[文本驱动视频生成]]
