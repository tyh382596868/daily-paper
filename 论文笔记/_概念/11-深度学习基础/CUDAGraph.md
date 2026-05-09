---
type: concept
aliases: [CUDA Graph, CUDA静态图, 静态计算图优化]
---

# CUDA Graph

## 定义
CUDA Graph 是 NVIDIA CUDA 提供的一种计算图捕获与重放机制，将一系列 GPU 操作（kernel launches、内存拷贝等）预先录制为静态图，推理时一次 launch 触发整个图执行，消除重复的 CPU-GPU 调度开销。

## 数学形式

推理延迟分解：

$$
T_\text{total} = T_\text{compute} + T_\text{launch overhead}
$$

CUDA Graph 将 $N$ 次独立 launch 的 overhead 压缩为 1 次：

$$
T_\text{static} \approx T_\text{compute} + T_\text{single launch}
$$

## 核心要点

1. **图捕获**: 首次推理时录制完整前向传播为 CUDA Graph（warm-up 阶段）
2. **静态执行**: 后续推理直接重放图，所有 kernel 调度由 GPU 驱动层统一管理
3. **适用场景**: 固定拓扑（无 Python-level 控制流分支）的推理，即批量大小、序列长度固定
4. **与 PyTorch Eager 对比**: Eager 模式每次 forward 都触发独立 kernel launch；Static Graph 消除碎片化调度
5. **RLDX-1 加速**: 配合 kernel fusion，将延迟从 71.2ms 降至 43.7ms（1.63×）

## 代表工作

- [[RLDX-1]]: 将 CUDA Graph + 算子融合用于机器人策略实时推理优化

## 相关概念

- [[MSAT]]
- [[算子融合|Operator Fusion]]
