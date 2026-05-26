---
type: concept
aliases: [FID, Fréchet Inception Distance]
---

# FID (Fréchet Inception Distance)

## 定义

FID 是评估生成图像质量的标准指标，衡量"生成分布"与"真实分布"在 Inception 网络特征空间下的 **Fréchet 距离**（两个高斯分布间的 Wasserstein-2 距离）。值越小越好。

## 数学形式

把生成图像和真实图像分别通过 Inception-v3 取倒数第二层特征，统计两个高斯 $\mathcal{N}(\mu_r, \Sigma_r)$ 与 $\mathcal{N}(\mu_g, \Sigma_g)$：

$$
\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left( \Sigma_r + \Sigma_g - 2 (\Sigma_r \Sigma_g)^{1/2} \right)
$$

## 核心要点

1. **越小越好**: FID = 0 表示分布完全一致
2. **Inception-v3 特征**: 用 ImageNet 预训练的 pool3 层（2048-d）
3. **样本量**: 一般建议 ≥ 10K 张才稳定
4. **对模式坍塌敏感**: 缺乏多样性会被 $\Sigma_g$ 项检测出来
5. **限制**: 特征来自分类网络，对 ImageNet 类别外的内容不敏感

## 代表工作

- 几乎所有视觉生成 paper 的标配指标
- [[X-Foresight]] Vision Renderer 在 1 s 视野下 FID 1.51（vs latent decoder 10.97）

## 相关概念

- [[FVD]]
- [[扩散模型]]
- [[Vision Renderer]]
