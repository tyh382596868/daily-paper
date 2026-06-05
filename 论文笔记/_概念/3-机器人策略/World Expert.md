---
type: concept
aliases: [World Expert, 世界专家]
---

# World Expert

## 定义

World Expert 是 [[WLA]] / [[World Action Model|WAM]] 类模型中负责**预测未来视觉观测**的 head。基于 backbone 共享隐藏态 $h_t$，在 [[VAE]] 潜空间生成 $n$ 步之后的子目标帧 $o_{t+n}$，作为动力学先验。

## 核心要点

1. **扩散 head**: 通常用 [[SANA]]、[[Diffusion Transformer|DiT]] 等扩散模型
2. **单帧 vs 视频**: [[WLA]] 选择只预测单帧 $o_{t+n}$ 而非完整视频，降低开销
3. **VAE 潜空间**: 在 [[VAE]] latent 上做扩散，比像素监督更紧凑
4. **训练-推理解耦**: 训练时提供动力学梯度；推理可关闭以节省延迟
5. **TTS 用途**: 推理时也可保留，用于 [[Test-Time Scaling]] 想象未来

## 代表工作

- [[WLA]]: 600M SANA 扩散 head，预测单帧未来
- [[DreamVLA]]: 完整视频生成的 World Expert
- [[Cosmos-Policy]]: 用 Cosmos World Foundation Model 作为 World Expert

## 相关概念

- [[Action-Conditioned World Model]]
- [[SANA]]
- [[VAE]]
- [[Test-Time Scaling]]
