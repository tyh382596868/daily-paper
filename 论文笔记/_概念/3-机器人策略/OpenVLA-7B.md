---
type: concept
aliases: [OpenVLA 7B, openvla-7b]
---

# OpenVLA-7B

## 定义
OpenVLA 的 7B 参数版本，融合 SigLIP + DINOv2 视觉特征与 LLaMA-2 7B 语言主干，通过 3 层 GELU MLP 投影器将视觉特征映射到语言嵌入空间，是目前多个 VLA 工作（如 OpenVLA-OFT、PearlVLA）的标准初始化基础。

## 架构组成
- **视觉编码器**: [[SigLIP]] + [[DINOv2]] 融合特征
- **投影器**: 3 层 MLP（GELU 激活）
- **语言主干**: [[LLaMA-2]] 7B

## 核心要点
1. 开源社区中用于机器人操作的高频基础 VLA 模型
2. 支持 [[LoRA]] 高效微调，与套件特定或跨任务策略均兼容
3. 原始 OpenVLA-7B 在 LIBERO 上约 76% 成功率；[[OpenVLA-OFT]] 微调后约 97.1%

## 代表工作
- [[OpenVLA]]: 原始论文
- [[OpenVLA-OFT]]: 优化微调版本
- [[PearlVLA]]: 在此基础上插入潜空间精炼模块

## 相关概念
- [[OpenVLA]]
- [[OpenVLA-OFT]]
- [[SigLIP]]
- [[DINOv2]]
- [[LLaMA-2]]
