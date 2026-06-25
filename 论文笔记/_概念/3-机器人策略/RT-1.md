---
type: concept
aliases: [Robotics Transformer 1, RT1]
---

# RT-1

## 定义
Google DeepMind 提出的首个大规模机器人 Transformer 基础模型，用 130k+ 真实机器人操作轨迹训练，输入视频帧 + 语言指令，输出离散化动作 token。

## 核心要点
1. 将机器人操作表述为 seq2seq 问题：视觉历史帧 + 语言 → 动作 token 序列
2. 用 TokenLearner 压缩视觉 token 数量，使 Transformer 推理在 3Hz 实时运行
3. 首次证明大规模多任务、多机器人 imitation learning 的泛化能力
4. 基础模型后续演进为 RT-2（加入 VLM）、RT-X（多机器人）

## 代表工作
- [[RT-2]]: 在 RT-1 基础上接入 VLM，实现语义推理能力
- [[OpenVLA]]: 开源复现路线

## 相关概念
- [[VLA]]
- [[Action Chunking]]
- [[TokenLearner]]
