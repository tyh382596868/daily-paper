---
type: concept
aliases: [VGGT, VGGT-1B, Visual Geometry Grounded Transformer]
---

# VGGT

## 定义

VGGT（Visual Geometry Grounded Transformer）是一个 ~1B 参数、为**多视图 3D 重建**预训练的 transformer，每帧输出一个 camera token 和若干 register token，天然编码视角、深度、几何关系等 3D 信息，常被下游任务用作冻结的几何先验编码器。

## 核心要点

1. **camera token**: 每帧 1 个，编码相机外参 / 视角
2. **register token**: 每帧 4 个左右，捕获跨帧几何一致性
3. **冻结使用**: 在小规模机器人数据上微调容易破坏几何先验，常被冻结
4. 与 [[DINOv2]] / [[DINOv3]] 不同，VGGT 显式针对多视图 3D，对"我刚从哪儿来"这类视角变化更敏感

## 代表工作

- [[IntentVLA]]: 用 VGGT 编码最近 $K=16$ 帧历史，每帧取 camera + 4 register token，作为短期意图证据注入 [[Qwen3-VL]] 主干
- [[GaussianDream]]: 冻结 VGGT 提取 3 帧多尺度特征，通过 [[TGE]] 时序融合得到 GaussianDream prefix
- [[SOMA]]: 用 VGGT 一次前向得到扫描序列的相机位姿和点图，把 [[YOLO-World]] 检框 lift 到统一 3D 坐标，作为空间记忆的几何骨架

## 相关概念

- [[DINOv2]]
- [[DINOv3]]
- [[ViT]]
- [[短期意图表征]]
