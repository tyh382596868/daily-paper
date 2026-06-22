---
type: concept
aliases: [DCT Action Tokenization, 频域动作量化, FAST动作Token化]
---

# DCT 动作 Token 化

## 定义

将机器人动作 chunk 通过离散余弦变换（DCT）投影到频域，再量化并用 BPE 编码为紧凑离散 Token 序列的动作表示方法，使动作可与视觉/语言 Token 在同一自回归词表中建模。

## 数学形式

$$
A_{\text{freq}} = \text{DCT}(a_{t:t+k}), \quad z_a = \text{BPE}(\text{Quantize}(A_{\text{freq}}))
$$

## 核心要点

1. **频域能量集中**: 低频系数包含大部分运动能量，DCT 后量化精度更高、Token 更少。
2. **时序结构保留**: DCT 将整个 chunk 的时序相关性编码进少量系数，避免逐帧量化的误差累积。
3. **与 FAST 方法关联**: FAST（Frequency-space Action Sequence Tokenization）是同类方法；UniVLA 在此基础上加入 BPE 进一步压缩。
4. **词表共享**: 量化后的动作 Token 与视觉/语言 Token 同属一个离散词表，使自回归统一建模成为可能。

## 代表工作

- [[UniVLA-ICLR2026]]（Wang et al., 2025）: 将 DCT+BPE 动作 Token 化集成到统一 VLA 框架
- [[FAST]]（pi 团队, 2025）: 提出频域动作序列 Token 化方法 FAST
- [[FAFM]]（Guo et al., 2026）: 在 DCT 系数空间执行 Flow Matching，解决异构频率与时序抖动

## 相关概念

- [[离散余弦变换 (DCT)]]
- [[字节对编码 (BPE)]]
- [[Action Chunking]]
- [[动作 Token]]
- [[VQ-VAE]]
