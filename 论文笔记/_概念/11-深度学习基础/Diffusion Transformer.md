---
type: concept
aliases: [DiT, Video DiT, 扩散 Transformer, 视频扩散 Transformer]
---

# Diffusion Transformer

## 定义

将扩散模型（DDPM/Flow Matching）的去噪网络从 U-Net 替换为 Transformer 架构，以 token 序列（图像 patch 或视频帧 patch）为基本操作对象，通过 self-attention 建模空间/时序依赖关系。

## 核心要点

1. **Token 化输入**：图像或视频经 VAE 编码到潜空间后，切分成固定大小的 patch，拍平为 token 序列
2. **条件注入**：时间步 $t$ 和文本条件通过 adaLN（自适应层归一化）或 Cross-Attention 注入各层
3. **时序注意力（视频 DiT）**：在 patch 的时间维度额外加一层 Temporal Self-Attention（STA），专门建模跨帧一致性
4. **可扩展性**：相比 U-Net，DiT 更易于按参数量线性 scale，催生了 Sora、CogVideoX、Open-Sora 等大规模视频生成模型

## 数学形式

标准 Video DiT Block 的三步残差计算：

$$
\begin{aligned}
p_t^l &= \mathrm{STA}(f_t^l), \quad \tilde{f}_t^l = f_t^l + p_t^l \\
q_t^l &= \mathrm{CA}(\tilde{f}_t^l), \quad \bar{f}_t^l = \tilde{f}_t^l + q_t^l \\
r_t^l &= \mathrm{MLP}(\bar{f}_t^l), \quad f_t^{l+1} = \bar{f}_t^l + r_t^l
\end{aligned}
$$

其中 $t$ 为去噪时间步，$l$ 为层索引，STA = Temporal Self-Attention，CA = Cross-Attention。

## 代表工作

- [[Sora]]: OpenAI 的旗舰视频生成模型，奠定了 Video DiT 的地位
- [[CogVideoX]]: 开源 Video DiT，采用 3D 全注意力
- [[AdaCache]]: 针对 Video DiT 推理加速的缓存方法，利用 STA 残差的去噪步间平滑性
- [[TeaCache]]: 另一种 DiT 残差缓存加速方法

## 相关概念

- [[DiT]]: 原始图像 DiT（Peebles & Xie 2023）
- [[Spatio-Temporal Attention]]: Video DiT 的时序建模模块
- [[DDPM]]: 扩散模型基础框架
- [[LDM]]: 潜空间扩散模型，DiT 通常在 latent space 操作
