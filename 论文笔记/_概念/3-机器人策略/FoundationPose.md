---
type: concept
aliases: [FoundationPose]
---

# FoundationPose

## 定义
NVIDIA 提出的统一物体姿态估计和追踪框架，支持已知 CAD 模型和仅有参考图像两种输入模式，在多个 benchmark 上达到 SOTA。

## 核心要点
1. 模型无关：无需针对每个物体重训练
2. 支持 CAD-free 模式（只需几张参考图像）
3. 在 [[DexSynRefine]] 等操作任务中用于物体姿态感知

## 代表工作
- Wen et al., 2024: FoundationPose（CVPR 2024）

## 相关概念
- [[SAM]]
- [[VLA]]
