---
type: concept
aliases: [3D Multimodal RoPE, 3D 多模态旋转位置编码]
---

# 3D mRoPE

## 定义

**3D Multimodal Rotary Position Embedding** 是 [[RoPE|Rotary Position Embedding]] 在多模态时空数据上的扩展：把 hidden 维度切成三段，分别为 token 的 $(t, h, w)$ 三维坐标各自做旋转，从而让文本 / 图像 / 视频 / 音频 / 动作 token 都能落到统一的 attention 索引空间。

## 核心要点

1. **三轴分离**: 时间 $t$、空间高度 $h$、空间宽度 $w$ 分别使用独立的频率基底
2. **跨模态对齐**: 动作 token 沿时间轴对齐到对应视频帧 $(t, 0, 0)$；文本 token 复用 1D 时间维度
3. **解决长视频外推**: 比单纯 1D RoPE 在长序列 / 高分辨率上泛化更好
4. **来源**: Qwen2-VL 引入 mRoPE，[[Cosmos3]] 将其升级为统一 5 模态版本

## 代表工作

- [[Qwen3-VL]]: 视频理解侧使用 mRoPE
- [[Cosmos3]]: 把 mRoPE 推广到包含动作在内的 5 种模态对齐

## 关联

- [[RoPE]]: 基础旋转位置编码
- [[PRoPE]]: 另一种位置编码扩展
- [[Spatio-Temporal Attention]]: 时空注意力机制
