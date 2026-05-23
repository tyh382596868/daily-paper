---
type: concept
aliases: [PyTorch, pytorch]
---

# PyTorch

## 定义

Meta AI 开源的深度学习框架，提供动态计算图、GPU 加速、自动微分与丰富的神经网络模块，是当前 CV / NLP / RL / 机器人学习领域使用最广泛的训练框架。

## 核心要点

1. **动态图 (Eager Mode)**: 写起来像 NumPy，调试方便，灵活控制流
2. **`torch.compile` / TorchScript**: 可静态化加速
3. **生态**: torchvision / torchaudio / lightning / DDP / FSDP
4. **机器人 / VLA 标配**: 绝大多数 VLA 工作（[[Pi05|π₀.₅]]、[[GaussianDream]] 等）基于 PyTorch
5. **分布式训练**: 支持 DDP、FSDP、Tensor Parallel 等大规模训练范式

## 代表工作

- 几乎所有现代深度学习工作都以 PyTorch 为实现框架

## 相关概念

- [[CUDAGraph]]
- [[Context-Parallel 训练]]
