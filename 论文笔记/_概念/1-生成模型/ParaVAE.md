---
type: concept
aliases: [Parallel VAE, 并行 VAE 解码, ParaVAE 解码]
---

# ParaVAE

## 定义

[[VAE]] 解码的并行加速方案，通过将视频帧沿高度维度切分并分发到多块 GPU 上并行解码，显著减少视频生成流水线中 VAE 解码的延迟。

## 核心要点

1. **高度维度切分**: 将视频帧沿 H 维度（高度）切分为多个 patch，分配至不同 GPU
2. **跨 GPU 聚合**: 解码完成后通过 patch gathering 操作拼接还原完整帧
3. **与 DiT 流水线解耦**: VAE 解码与 DiT 生成异步并行，不阻塞下一块视频的扩散去噪
4. **结构剪枝结合**: 通常配合 VAE 结构剪枝（如 75% 剪枝率）共同使用

## 代表工作

- [[DreamXWorld]]: 使用 ParaVAE + 75% 剪枝率 + torch.compile，配合异步 DiT-VAE 流水线实现 16 FPS 实时推理

## 相关概念

- [[VAE]]: 基础变分自编码器
- [[自回归流式推理]]: ParaVAE 所服务的推理框架
- [[Diffusion Transformer]]: 配套生成骨干
