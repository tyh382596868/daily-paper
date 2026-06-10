---
type: concept
aliases: [π₀-FAST, pi0 FAST, pi-zero FAST]
---

# pi0-FAST

## 定义

Physical Intelligence (π.ai) 提出的 π₀ 模型使用 FAST 动作分词器的版本，通过将连续动作替换为 FAST 离散 token，实现更快速的自回归动作生成，是当前 LIBERO 等 benchmark 的强基线。

## 核心要点

1. **FAST 集成**: 将 π₀ 的连续扩散动作头替换为 [[FAST Action Tokenizer]] 离散 token 生成
2. **自回归推理**: 使用 LLM 风格的自回归生成替代扩散多步去噪，推理更快
3. **LIBERO 性能**: 平均 85.5%（UniVLA 以 95.5% 超越）
4. **代码/模型**: Physical Intelligence 开源

## 代表工作

- [[UniVLA]]: 将 pi0-FAST 作为主要对比基线并全面超越（+10pp on LIBERO）

## 相关概念

- [[FAST Action Tokenizer]]: pi0-FAST 的核心动作编码方法
- [[π₀]]: 基础模型
- [[LIBERO]]: 主要评估 benchmark
- [[Action Chunking]]: 预测粒度
