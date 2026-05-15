---
type: concept
aliases: [LTX2 VAE, LTX-2 Video Autoencoder]
---

# LTX-2 VAE

## 定义
Lightricks 发布的视频自编码器，将 RGB 视频压缩到 128 通道、高时空压缩比的潜空间，是 [[SANA-WM]] 等长视频生成模型的关键 tokenizer。

## 核心要点

1. **128 潜通道**: 比 SD-VAE 的 4 通道大得多，承载更多时空信息
2. **高压缩率**: 比 Wan2.1-VAE 小 ~8 倍、比 ST-DC-AE 小 ~2 倍，显著缩短序列长度
3. **训练速度**: 在 [[SANA-WM]] 中将端到端延迟从 1266ms 降到 372ms（3.4×）
4. **稍弱细节**: VBench-I2V 的 CM/AQ 指标略低于 Wan-VAE，但 IB/SC 更高
5. **配套精炼器**: LTX-2.3 自带短视频精炼器，可被改造为长视频精炼器

## 代表工作
- [[SANA-WM]]: Stage 1 用 LTX-2 VAE 替换 Wan-VAE
- LTX-2 / LTX-2.3 系列（Lightricks）

## 相关概念
- [[视频扩散模型]]
- [[扩散变换器]]
- [[截断 σ 流匹配]]
