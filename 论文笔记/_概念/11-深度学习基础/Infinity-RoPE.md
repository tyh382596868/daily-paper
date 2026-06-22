---
type: concept
aliases: [Infinity RoPE, 无限 RoPE, 长序列 RoPE 外推]
---

# Infinity-RoPE

## 定义

针对超长序列的 [[RoPE]] 外推方法，通过扩展旋转位置编码的频率范围，使模型能够处理远超训练长度的序列，常用于视频生成中的长时程自回归生成。

## 核心要点

1. **频率范围扩展**: 将 RoPE 的频率基数（base）扩展，使位置编码覆盖更长序列
2. **与 YaRN 结合**: 通常配合 [[YaRN]] 或 [[NTK-aware RoPE]] 使用，提升长序列推理质量
3. **视频生成应用**: 在视频世界模型中用于处理分钟级长视频的时序位置信息

## 代表工作

- [[DreamXWorld]]: 在长序列适应阶段使用 Infinity-RoPE，支持生成最长约 1 分钟的稳定视频

## 相关概念

- [[RoPE]]: 基础旋转位置编码
- [[NTK-aware RoPE]]: 基于 NTK 理论的 RoPE 外推
- [[YaRN]]: Yet Another RoPE extensioN
- [[自回归生成]]: 应用场景
