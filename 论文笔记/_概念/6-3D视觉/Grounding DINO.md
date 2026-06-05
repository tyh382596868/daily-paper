---
type: concept
aliases: [GroundingDINO, G-DINO]
---

# Grounding DINO

## 定义

开放词汇目标检测器，结合 DINO 视觉骨干与文本编码器，能根据自然语言提示在图像中检出对应物体框。

## 核心要点

1. 文本-图像跨模态融合，无需固定类别集
2. 常与 [[SAM2]] 组合形成"文本→mask"流水线
3. 在机器人感知、视频跟踪、数据自动标注中广泛使用

## 代表工作
- Liu et al., Grounding DINO (2023)
- [[Dream-exe]]：用于自动定位末端与目标物体

## 相关概念
- [[SAM2]]
- [[CLIP]]
- [[Open-Vocabulary Detection]]
