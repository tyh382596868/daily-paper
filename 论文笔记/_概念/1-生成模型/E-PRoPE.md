---
type: concept
aliases: [Efficient PRoPE, 高效投影位置编码, Efficient Projective Rotary Position Embedding]
---

# E-PRoPE

## 定义

[[PRoPE]] 的高效变体，通过对空间降采样后的 token 应用投影注意力，在保持相机控制精度的同时将推理延迟降低约 30%。

## 数学形式

$$
D_s^{\text{E-PRoPE}} = D_s^{\text{Proj}}
$$

E-PRoPE 省略 RoPE 分量，仅保留投影子矩阵，作用于空间下采样至 4,096 的 token（原始 18,480 token 的 4.5× 压缩）。

## 核心要点

1. **空间 token 降采样**: 将输入 token 从 18,480 压缩至 4,096，大幅减少投影注意力计算量
2. **省略 RoPE 分量**: 只保留 $D_s^{\text{Proj}}$ 投影子矩阵，去除 $D_s^{\text{RoPE}}$ 旋转编码部分
3. **冻结 backbone**: DiT 主干参数冻结，仅对 PRoPE 模块反向传播
4. **即插即用**: 推理时可直接接入，不改变 DiT 接口

## 代表工作

- [[DreamXWorld]]: 提出 E-PRoPE，在 1280×720 分辨率下将延迟从 80s 降至 59s，相机控制分仅降 0.14

## 相关概念

- [[PRoPE]]: 原始投影旋转位置编码
- [[Diffusion Transformer]]: E-PRoPE 所附接的 backbone
- [[RoPE]]: 旋转位置编码
- [[Camera-Controlled 视频生成]]: E-PRoPE 的应用场景
