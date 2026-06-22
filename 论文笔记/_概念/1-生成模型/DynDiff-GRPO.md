---
type: concept
aliases: [DynDiff-GRPO, Dynamic-Aware Rollout Diversification, 动态感知 Rollout 多样化]
---

# DynDiff-GRPO

## 定义
DynDiff-GRPO 是一种用于视频扩散/流匹配世界模型 RL 后训练的动态感知 rollout 多样化策略，通过构建动态感知掩码 $M_t$，将 SDE 采样中的随机性优先分配到图像/视频的动态活跃区域，在保持静态背景一致性的同时最大化轨迹探索空间。

## 数学形式

动态感知掩码：

$$
M_{t}=r_{\text{base}}+(1-r_{\text{base}})\left(0.5D+0.5B\right)
$$

其中 $D$ 为连续动态强度图，$B$ 为稀疏二值动态先验：

$$
D=\mathrm{Normalize}\left(\|R\|_{2}\right),\quad B=\mathbb{I}\left[D>\mathrm{Quantile}(D,\tau)\right]
$$

动态感知噪声分配：

$$
\sigma_{t^{-}}^{\text{noise}}=\sigma_{t^{-}}\odot M_{t}
$$

完整训练目标：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{GRPO}}+\beta\mathcal{L}_{\mu\text{-KL}}
$$

## 核心要点
1. **动态感知**: 利用帧间 clean sample 预测残差 $R^{(k)}$ 定位动态活跃区域，无需额外光流或运动估计网络。
2. **各向异性探索**: 噪声以特征维度为单位（feature-wise）各向异性分配，而非全局标量，支持更精细的几何探索。
3. **物理约束保留**: 静态背景区域噪声接近 $r_{\text{base}}$（接近零），防止场景漂移，维持物理一致性。
4. **均值空间 KL**: 使用均值空间 KL 正则（$\mathcal{L}_{\mu\text{-KL}}$）比 log-prob 空间 KL 更有效（实验验证）。
5. 优势截断阈值 $A_{\max}$ 设为 2.5（低于标准 GRPO 的 5.0），以应对课程式投票带来的更高方差。

## 代表工作
- [[RewardAgent]]: DynDiff-GRPO 的提出论文，在 Cosmos-Predict2.5 和 Kairos-3.0-Robot 上验证有效性。

## 相关概念
- [[GRPO]]
- [[Flow Matching]]
- [[Video Diffusion Model]]
- [[Reward Hacking]]
