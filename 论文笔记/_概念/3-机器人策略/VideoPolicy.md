---
type: concept
aliases: [Video Policy]
---

# VideoPolicy

## 定义

更宽泛的术语：使用视频生成 backbone（如 Video Diffusion / DiT）作为骨干，把视频先验注入策略学习。

> 注：根据 [[WAM-Survey]] 的术语切分，Video Policy ⊆ WAM 仅当存在 explicit world-modeling supervision；否则属于纯 backbone 借用。

## 核心要点

1. **结构上**: 用视频生成模型架构。
2. **是否 WAM**: 取决于是否对未来状态 $o'$ 有显式预测目标。
3. 在综述 Figure 3 的 Venn 图中部分重叠 WAM。

## 代表工作

- [[VPP]]
- [[WAM-Survey]] 中详细对比 Video Policy 与 WAM 边界。

## 相关概念

- [[World Action Model]]
- [[视频扩散模型]]
