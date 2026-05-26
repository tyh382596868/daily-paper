---
type: concept
aliases: [FVD, Fréchet Video Distance]
---

# FVD (Fréchet Video Distance)

## 定义

FVD 是 [[FID]] 在视频域的扩展：把 Inception-v3 换成 **I3D**（Inflated 3D ConvNet，Kinetics 预训练），衡量生成视频与真实视频在 I3D 特征空间下的 Fréchet 距离。值越小越好。

## 数学形式

与 [[FID]] 公式一致，仅特征提取器换为 I3D：

$$
\text{FVD} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left( \Sigma_r + \Sigma_g - 2 (\Sigma_r \Sigma_g)^{1/2} \right)
$$

其中 $\mu, \Sigma$ 来自 I3D pool 层特征。

## 核心要点

1. **专为视频设计**: I3D 同时建模时间和空间，能捕捉运动一致性
2. **比 FID 更全面**: 一帧帧好不代表视频好——FVD 会暴露闪烁 / 物体 ID 跳变等问题
3. **输入要求**: 通常输入 16 帧或更长 clip
4. **越小越好**: 与 FID 同方向

## 代表工作

- 视频生成 / 视频预测 paper 的标配
- [[X-Foresight]] Vision Renderer 在 1 s 视野 FVD 11.28（vs latent decoder 135.56，~12× 改进）

## 相关概念

- [[FID]]
- [[视频生成]]
- [[视频扩散模型]]
- [[Vision Renderer]]
