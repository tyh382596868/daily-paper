---
type: concept
aliases: [NTK RoPE, NTK-aware Scaled RoPE, 神经切核感知旋转位置编码]
---

# NTK-aware RoPE

## 定义

基于神经切线核（Neural Tangent Kernel，NTK）理论的 [[RoPE]] 长度外推方法，通过非线性缩放各频率分量的 base，在不微调的情况下将位置编码外推至训练长度数倍的序列。

## 数学形式

将 RoPE base $b$ 替换为按外推比例缩放后的 $b'$：

$$
b' = b \cdot \alpha^{d/(d-2)}
$$

其中 $\alpha$ 为外推倍率，$d$ 为 embedding 维度。

## 核心要点

1. **无需微调**: 在推理时直接替换 base，适用于已训练模型的即插即用外推
2. **高频低频分离**: NTK 缩放对高频分量（局部相对位置）影响小，对低频分量（全局绝对位置）影响大
3. **与 YaRN 互补**: [[YaRN]] 在 NTK-aware 基础上进一步加入插值和 attention scaling

## 代表工作

- [[DreamXWorld]]: 在 Memory-Conditioned Scene Persistence 中用于长序列时序位置编码

## 相关概念

- [[RoPE]]: 基础方法
- [[YaRN]]: 更完整的长度外推框架
- [[Infinity-RoPE]]: 视频生成场景下的极长序列外推
