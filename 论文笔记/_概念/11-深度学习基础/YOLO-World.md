---
type: concept
aliases: [YOLO-World, YOLO World, Open-Vocabulary YOLO]
---

# YOLO-World

## 定义

YOLO 系列中的**开放词汇目标检测器**——接受任意文本类别作为提示，输出对应物体的 2D 检测框，不需要为新类重新训练。

## 核心要点

1. 视觉编码器（YOLO 主干）+ 文本编码器（CLIP-like）双塔，文本嵌入指导检测头
2. 速度接近闭集 YOLO，远快于纯 [[VLM]] 描述
3. 在机器人系统中常做"任意物体定位"的前端，配合 [[DINOv3]] 提取语义特征
4. 输出 2D 框可以与深度/位姿结合，lift 到 3D 用于空间表征

## 代表工作

- [[SOMA]]: 作为开放词汇检测前端，每帧检测物体框送入空间记忆构建

## 相关概念

- [[DINOv3]]
- [[CLIP]]
- [[VGGT]]
- [[SOMA]]
