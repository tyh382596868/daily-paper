---
type: concept
aliases: [villa-X, villaX]
---

# villa-X

## 定义

在 [[LaPA]] 基础上推进的潜在动作 WAM：(1) 引入 proprioceptive Forward Dynamics Model 把 latent action 锚定到物理动力学；(2) latent action expert + robot action expert 的联合扩散框架，让 latent action 显式条件化低层动作生成。

## 核心要点

1. **proprio-FDM**: 视觉重建 + proprioception 预测联合优化。
2. **双 expert 扩散**: 知识从 latent 层结构化传递到 robot 层。
3. 属于 [[Cascaded WAM|Cascaded WAM]] 的 Implicit / Latent 子类（综述定位）。

## 代表工作

- [[WAM-Survey]] 综述中作为潜在动作 WAM 进阶版。

## 相关概念

- [[LaPA]]
- [[残差潜在动作]]
- [[World Action Model]]
