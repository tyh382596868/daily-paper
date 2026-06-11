---
type: concept
aliases: [Trace VLA, Visual Trace Prompting VLA]
---

# TraceVLA

## 定义

将历史末端执行器运动轨迹以视觉提示方式叠加到输入图像上，增强 VLA 的空间-时序感知能力的方法（arXiv:2412.10345）；通过在 OpenVLA 上微调 15 万条轨迹数据实现。

## 数学形式

$$
\pi(a_t | o_t \oplus \text{Trace}(h_{t-T:t}), l)
$$

其中 $\text{Trace}(\cdot)$ 将历史 $T$ 帧的末端执行器轨迹以彩色点序列叠加到当前图像 $o_t$ 上。

## 核心要点

1. **输入端轨迹**: 轨迹作为视觉提示（prompt）输入，在推理时需要轨迹数据
2. **微调策略**: 基于 OpenVLA 微调，数据集包含 150K 带轨迹标注的操作轨迹
3. **评测**: 在 SimplerEnv 137 个配置上超越 OpenVLA 10%，真实 WidowX 上提升 3.5x

## SeeTraceAct 的区别

SeeTraceAct 将轨迹从**输入端提示**转变为**训练时辅助监督目标**，推理时无需轨迹信息，且显式处理了多视角可见性问题。

## 代表工作

- [[TraceVLA]] (arXiv:2412.10345): 原始论文
- [[SeeTraceAct]]: 后续延伸，训练目标 vs 输入提示的对比

## 相关概念

- [[VLA]]
- [[OpenVLA]]
- [[Visibility-Aware Trace Loss]]
