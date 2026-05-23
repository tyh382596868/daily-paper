---
type: concept
aliases: [Depth Anything V2, DAv2]
---

# Depth Anything V2

## 定义

由 HKU & TikTok 发布的通用单目深度估计基础模型，在数千万合成 + 真实数据上预训练，支持 zero-shot 高质量相对深度估计。是 V1 的升级版，对几何细节与边界更精确。

## 核心要点

1. **架构**: [[DINOv2]] 编码器 + [[DPT]] 风格深度解码 head
2. **训练数据**: 大规模 synthetic + relative 标签蒸馏
3. **输出**: 相对深度，可通过相机内参或额外尺度对齐转为度量深度
4. **常见用途**: 作为**伪 ground-truth 深度**为下游任务（世界模型、3D 重建、机器人策略）提供密集监督
5. **局限**: 反光 / 透明 / 非朗伯表面失效，长距离精度退化

## 代表工作

- [[GaussianDream]]: 用 Depth Anything V2 生成训练时的伪 GT 深度，结合相机内参转度量深度，用于深度损失与 3D 场景流构造

## 相关概念

- [[DINOv2]]
- [[DPT]]
- [[MoGe-2]]
- [[相机投影]]
