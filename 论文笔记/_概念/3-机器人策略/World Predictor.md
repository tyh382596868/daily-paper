---
type: concept
aliases: [World Predictor, 世界预测器]
---

# World Predictor

## 定义

World Predictor 指 [[WAIM]] / [[World Action Model|WAM]] 架构中负责**根据当前观测、条件与候选动作输出未来世界状态**的子模块。在 [[DAWN]] 中具体为一个**因果 [[Transformer]]**，在紧凑[[潜空间世界模型|潜空间]]做 short rollout。

## 数学形式

$$
z_{future}^{(r)} = P_\theta(z, c, a_{1:H}^{(r)})
$$

其中 $z$ 是当前观测的潜表示（由 [[Resampler]] 给出），$c$ 是条件 token（自车状态、路由），$a_{1:H}^{(r)}$ 是第 $r$ 轮的候选动作序列。

## 核心要点

1. **Action-conditioned**: 与单纯的"未来视频生成"区别——输入显式包含动作假设
2. **Latent rollout, 不在像素空间**: 极大降低计算量并使预测语义更与规划对齐
3. **Short horizon 即可**: [[DAWN]] 实验 $T_v \in \{0,1,2,3,4\}$s，2–3s 已抓住主要收益
4. **可与 [[World-Conditioned Action Denoiser]] 递归交互**: 形成 [[WAIM]] 的核心迭代

## 代表工作

- [[DAWN]]: 因果 Transformer 实现，短 latent rollout
- [[Drive-JEPA]]: 并行式 world prediction（不接受动作条件）

## 相关概念

- [[WAIM]]
- [[World Action Model]]
- [[潜空间世界模型]]
- [[Transformer]]
