---
type: concept
aliases: [Wan2.1, Wan2.1-T2V, Wan2.1-T2V-1.3B]
---

# Wan2.1

## 定义
Wan2.1 是阿里 Wan 系列的视频生成基础模型，基于扩散 Transformer（[[DiT]]）架构，支持文生视频（T2V）等任务；其 1.3B 版本（Wan2.1-T2V-1.3B）常作为轻量级视频生成 backbone。

## 核心要点
1. 在冻结 [[VAE]] 的隐空间操作，时间压缩 + 空间压缩（如 $(4,8,8)$），用扩散 Transformer 做去噪。
2. 文本条件通过冻结的 [[UMT5]]-XXL 编码器经 cross-attention 注入。
3. 1.3B 版本：$\dim=1536$、30 层、12 头，可在中等算力下微调。
4. cross-attention 通路可被复用于其他条件（如动作条件化）。

## 代表工作
- [[PROWL]]: 用 Wan2.1-T2V-1.3B 作为世界模型 backbone，把 cross-attention 通路从文本条件复用为动作条件
- [[Wan2.2]]: 后续版本

## 相关概念
- [[DiT]]
- [[VAE]]
- [[UMT5]]
- [[Diffusion Forcing]]
