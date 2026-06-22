---
type: concept
aliases: [Yet Another RoPE extensioN, YaRN RoPE, YaRN 外推]
---

# YaRN

## 定义

"Yet Another RoPE extensioN"，一种高效的 [[RoPE]] 上下文窗口扩展方法，在 [[NTK-aware RoPE]] 的基础上引入注意力缩放和分段插值，以少量微调代价将 LLM/视频模型的有效上下文窗口扩展数十倍。

## 核心要点

1. **分段频率处理**: 将 RoPE 频率按高、中、低三段分别处理——高频保持不变（局部相对位置），低频线性插值（全局位置），中频 NTK 缩放
2. **注意力温度缩放**: 引入 $\sqrt{s}$ 因子补偿长序列下注意力分布的方差变化（$s$ 为序列长度缩放比）
3. **少量微调**: 仅需约 400 步微调即可获得良好的长程泛化性能
4. **即插即用**: 也支持零微调推理，性能略低于微调版

## 代表工作

- [[DreamXWorld]]: 在场景记忆模块的时序位置编码中使用 YaRN 处理长时程视频序列

## 相关概念

- [[RoPE]]: 基础旋转位置编码
- [[NTK-aware RoPE]]: YaRN 的前驱方法
- [[Infinity-RoPE]]: 视频生成专用长序列外推
