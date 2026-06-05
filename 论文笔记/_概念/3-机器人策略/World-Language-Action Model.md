---
type: concept
aliases: [WLA, World-Language-Action Model, 世界-语言-动作模型]
---

# World-Language-Action Model

## 定义

World-Language-Action Model（WLA）是 [[VLA]] 与 [[World Action Model|WAM]] 的统一推广：在单一自回归 backbone 上同时输出 (language subtask, future image, action chunk) 三种模态，把高层语义子任务推理、低层物理动力学预测与可执行动作生成耦合到一起。

## 核心要点

1. **三模态最小充分集**:
   - **Language**: 子任务文本 $\mathcal{S}_t$，提供长程语义规划
   - **Image**: 单帧未来子目标 $o_{t+n}$，提供物理动力学先验
   - **Action**: 动作块 $a_{t:t+H}$，落地控制
2. **共享 backbone + 多专家头**: 三个 expert 共享 backbone 表征但各自有解码头，训练联合优化
3. **训练-推理解耦**: 训练时 World/Language Expert 提供监督信号；推理可只启用 Action Expert，保留实时性
4. **隐式动力学注入**: 即使推理时丢弃 World Expert，其训练梯度已经把动力学先验压进 backbone

## 与相关范式的关系

- **[[VLA]]**: WLA 的子集，缺少 Image 模态
- **[[World Action Model|WAM]]**: WLA 的子集，缺少显式 Language 模态
- **[[Embodied Reasoning]]**: WLA 通过 Language Expert 显式承载具身推理

## 代表工作

- [[WLA]]: 首个明确提出 WLA 范式的工作，WLA-0 prototype 2B 活跃参数 / 40 ms 推理
- [[Motus]]: 同三模态思路但推理慢（>1600 ms）
- [[CoT-VLA]]: WLA 的前身，主要做 Language + Action

## 相关概念

- [[VLA]]
- [[World Action Model]]
- [[Test-Time Scaling]]
- [[Action Chunking]]
- [[Flow Matching]]
