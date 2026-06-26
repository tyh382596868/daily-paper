---
type: concept
aliases: [往返残差, 分词器往返残差, tokenizer round-trip residual]
---

# Round-Trip Residual

## 定义

视觉[[世界模型]]幻觉检测的无监督信号之一：将动力学模型预测的潜码 $\hat{z}$ 经过分词器解码再重编码，如果 $\hat{z}$ 落在分词器重建流形之外，则往返残差 $u_r$ 较大，预示感知幻觉。

## 数学形式

$$
u_r = \|\hat{z} - \mathrm{Encode}(\mathrm{Decode}(\hat{z}))\|
$$

运动归一化版本：

$$
u_r^{\text{norm}} = \frac{u_r}{m}
$$

其中 $m$ 为场景运动幅度（逐步 RMS 潜表示变化量）。

## 核心要点

1. **无需标注**：仅用分词器的编解码操作计算，不需要真实下一帧或仿真器
2. **靶向感知幻觉**：检测 Tokenizer 对分布外场景的重建失真
3. **与展开误差的相关性**：Spearman ρ ≈ 0.80（MMBench2 验证）

## 代表工作

- [[MMBench2]]：提出该指标，与 $u_f$（流不稳定性）和 $u_s$（跨种子方差）构成三元幻觉检测体系

## 相关概念

- [[Flow Instability]]
- [[Inter-Seed Variance]]
- [[Motion-Normalized Predictor]]
- [[世界模型]]
- [[视频分词器]]
