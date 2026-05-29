---
type: concept
aliases: [RT2, Robotics Transformer 2]
---

# RT-2

## 定义

RT-2（Robotics Transformer 2）是 Google DeepMind 发布的开创性 [[VLA]] 模型：将 PaLI-X / PaLM-E 等大规模视觉-语言模型直接迁移到机器人控制，把动作离散成 token 与文本 token 共享 vocab，端到端做 next-token prediction。

## 核心要点

1. **VLA 概念奠基者**：首次大规模证明 VLM 可以直接生成机器人动作；
2. **动作 token 化**：把每维动作离散成 256 个 bin；
3. **跨任务泛化**：从互联网 VL 数据中"借" semantic generalization；
4. **是 NIAF 等连续方法要超越的旧范式**：离散动作 token 是 NIAF 反对的对象。

## 代表工作

- **RT-2** (Brohan et al., 2023)
- [[OpenVLA]]：RT-2 的开源版本
- [[NIAF]]：彻底放弃离散 token，改用连续函数

## 相关概念

- [[VLA]]
- [[OpenVLA]]
- [[Action Chunking]]
- [[Neural Implicit Action Field]]
