---
type: concept
aliases: [FastWAM, Fast World Action Model, 快速世界动作模型]
---

# FastWAM

## 定义

FastWAM 是一种高效 World Action Model，通过缓存图像编辑/视频生成过程的中间 KV 特征（而非生成完整视频帧）来条件化动作预测，在保持 WAM 语义世界建模能力的同时大幅降低推理延迟。

## 核心要点

1. **KV Cache 条件化**: 不解码完整视频 token，直接复用生成模型中间层的 Key-Value 缓存作为动作专家的条件信号
2. **延迟改善**: 相比视频生成 WAM（FastWAM-IDM）从 1081ms 降至 302ms（~1/4 延迟），FLOPs 从 63.65T 降至 13.21T
3. **单步视频去噪**: 推理时仅执行 1 步去噪，提取缓存即停，无需完整扩散链

## 效率指标

| 变体 | 延迟 (ms) | TFLOPs |
|------|----------:|-------:|
| FastWAM-IDM（完整视频）| 1081 | 63.65 |
| FastWAM（1 步去噪）| 302 | 13.21 |
| FastWAM（Prefix Only）| 194 | — |

## 代表工作

- [[ImageWAM]]: 继承 FastWAM 的 KV Cache 条件化范式，进一步将视频生成替换为更轻量的图像编辑骨干，延迟降至 263ms，FLOPs 降至 9.72T

## 相关概念

- [[World Action Model]]
- [[KV Cache]]
- [[Flow Matching]]
- [[Action Expert]]
