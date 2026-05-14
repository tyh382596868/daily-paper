---
type: concept
aliases: [WAIM, World-Action Interactive Model, 世界-动作交互模型]
---

# WAIM (World-Action Interactive Model)

## 定义

WAIM 是 [[DAWN]] 论文提出的新范式，把"未来世界状态"与"未来动作序列"视作一对**相互依赖、相互函数**的耦合变量，通过推理时的迭代算子求解二者的自洽不动点。它是 [[World Action Model|WAM]] 的超集——并行/顺序/zero-rollout 等已有 WAM 都是 WAIM 在迭代次数或耦合方向上的退化情形。

## 数学形式

自洽对：
$$
\hat{v}_{1:T} = F_\theta(o, l, \hat{a}_{1:H}), \quad \hat{a}_{1:H} = G_\phi(o, l, \hat{v}_{1:T})
$$

迭代算子：
$$
(v_{1:T}^{(k+1)}, a_{1:H}^{(k+1)}) = \mathcal{I}_\Theta(v_{1:T}^{(k)}, a_{1:H}^{(k)}; o, l)
$$

## 核心要点

1. **Reciprocity 原则**: 好的未来依赖所选动作，好的动作又依赖预测的未来——单方向条件化都丢失信息
2. **推理时迭代**: 不靠一次前向，而靠 $K$ 轮 world↔action 互修逼近自洽点
3. **不强求像素级 / 全 horizon rollout**: 在紧凑[[潜空间世界模型|潜空间]]做 short rollout 同样有效（[[DAWN]] 用 $T_v=4$s, $K=4$）
4. **退化关系**:
   - $K=0$ + 无 world → 直接策略
   - $K=1$ + frozen world → 传统 predict-then-plan
   - $K=1$ + 并行 world/action → 现有 [[World Action Model|WAM]]

## 代表工作

- [[DAWN]]: 提出 WAIM 范式，在自动驾驶 NAVSIM v1 拿下 perception-free SOTA

## 相关概念

- [[World Action Model]]
- [[World Predictor]]
- [[World-Conditioned Action Denoiser]]
- [[递归交互推理]]
- [[潜空间世界模型]]
