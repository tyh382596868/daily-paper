---
type: concept
aliases: [Controllable World Model, 可控世界模型]
---

# Controllable 世界模型

## 定义

以动作序列为条件预测未来观测的[[世界模型]]，与 [[passive 世界模型]] 的区别在于显式接收 $a_{t:t+H}$ 作为输入。

## 数学形式

$$
p(o_{t+1:t+k} \mid o_t,\ a_{t+1:t+k})
$$

## 核心要点

1. 是机器人世界模型的最常见形式，因为机器人系统必须能响应控制输入
2. 与 [[passive 世界模型]] 的区分对应"被动观察 vs 主动控制"两种数据来源
3. action-conditioning 的质量直接决定 WM 作为模拟器/评估器的可用性
4. 在 [[强化学习]] 中通常还需要扩展输出奖励与终止信号

## 代表工作

- [[IRASim]]: 动作条件视频模拟器
- [[CtrlWorld|Ctrl-World]]: action-faithful rollout
- [[EA-WM]]: KVAF 几何对齐
- [[Cosmos-Policy]]: 动作即潜帧

## 相关概念

- [[世界模型]]
- [[passive 世界模型]]
- [[Inverse Dynamics Model]]
- [[RobotWM-Survey]]
