---
type: concept
aliases: [具身幻觉, embodiment hallucination, embodiment inconsistency]
---

# embodiment hallucination

## 定义

视频生成 / 世界模型在合成机器人交互视频时，**机器人形态、关节、动力学相对真实硬件出现不一致或漂移**的现象。常见表现：手指数变化、关节角错乱、夹爪开合相位错、机器人形态在不同帧间漂移。

## 核心要点

1. 出现在 text-driven / 端到端视频扩散生成中（如 [[DreamZero]] / DreamGen），因为模型必须"想象"机器人怎么动
2. 直接导致下游策略训练失效 — 学到的动作在真实硬件上无法执行
3. 解决路线：[[Embodiment Anchoring|具身锚定]] — 显式渲染机器人本体作为像素级条件
4. 是 [[RoboDream]] 等 anchoring 类方法相对 free-form 视频生成的核心动机

## 代表工作

- [[RoboDream]]: 用 IsaacLab 渲染 $v_{rob}$ 锚定，几乎消除 embodiment hallucination
- [[AnchorDream]]: 同思路前作
- [[DreamZero]]: 典型出现 embodiment hallucination 的工作

## 相关概念

- [[Embodiment Anchoring]]
- [[Video Diffusion Model]]
- [[Compositional World Models]]
