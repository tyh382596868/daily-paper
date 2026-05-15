---
type: concept
aliases: [GDN, Gated Delta Network, 门控 DeltaNet]
---

# Gated DeltaNet

## 定义

一种循环线性注意力变体，把每一步的状态更新写成"门控衰减 + 秩-1 擦写 + 加性更新"的形式，状态规模始终保持为 $D \times D$，因此可以在分钟级长上下文中保持显存恒定。

## 数学形式

$$
\mathbf{S}_t = \mathbf{S}_{t-1} \mathbf{M}_t + \mathbf{U}_t
$$

其中：

$$
\mathbf{M}_t = \gamma_t (\mathbf{I} - \hat{\mathbf{K}}_t \bm{\beta}_t \hat{\mathbf{K}}_t^\top), \quad
\mathbf{U}_t = \mathbf{V}_t \bm{\beta}_t \hat{\mathbf{K}}_t^\top, \quad
\mathbf{O}_t = \mathbf{S}_t \hat{\mathbf{Q}}_t
$$

## 核心要点

1. 转移矩阵 $\mathbf{M}_t$ 是 Householder 形式秩-1 修正，可证明 $\|\mathbf{M}_t\|_2 \le \gamma_t$
2. 门控 $\gamma_t \in (0, 1]$ 提供"软遗忘"，决定旧状态保留比例
3. Key 必须 RMSNorm + ReLU 后再做 $1/\sqrt{D \cdot S}$ 缩放，否则会数值爆炸
4. 支持 chunkwise parallel scan，训练可与 [[自注意力]] 同速
5. 在长视频/长序列生成中替代纯 [[自注意力]] 的二次代价

## 代表工作

- [[SANA-WM]]: 15 层 GDN + 5 层 softmax 混合，实现 720p 60 秒视频生成

## 相关概念

- [[线性注意力]]
- [[自注意力]]
- [[RMSNorm]]
- [[Context-Parallel Training]]
