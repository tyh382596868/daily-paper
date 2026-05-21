---
type: concept
aliases: [Instructional Consistency, 指令一致性, IC]
---

# Instructional Consistency

## 定义

要求 [[VLA]] 策略在面对"语义等价但措辞不同"的指令时产生一致动作的约束，属于[[多一致性约束]]中针对指令语义变换的一致性。

## 核心要点

1. 缓解"伪语言遵循"问题：现有 VLA 对指令改写敏感，换个说法就失败。
2. 实现方式通常是数据层面的隐式正则——用外部 LLM 为每条指令生成语义等价改写集合 $\mathcal{D}_T$，训练时均匀采样。
3. [[RoVLA]] 用 Qwen3-8B（关闭 thinking）+ 7 个 prompt 模板（用户意图式、功能目标式、礼貌请求式、简洁命令式、教学式、抽象目标式、功能指代式）生成改写，并做去重与拒答过滤。
4. 不引入额外损失项，仅靠"同一任务多样表达"驱动策略学到一致的任务语义。

## 数学形式

$$
\mathcal{D}_T = \{\,T^{(1)}, \ldots, T^{(N_{\text{lang}})}\,\}
$$

## 代表工作

- [[RoVLA]]: 用 Qwen3-8B 生成指令改写集合，消融显示语言鲁棒性维度获得 +25~27 点增益。

## 相关概念

- [[多一致性约束]]
- [[Evolutionary Consistency]]
- [[Observational Consistency]]
- [[VLA]]
