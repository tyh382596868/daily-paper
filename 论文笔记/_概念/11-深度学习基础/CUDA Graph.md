---
type: concept
aliases: [CUDA Graph, CUDA Graphs]
---

# CUDA Graph

## 定义

CUDA Graph 是 NVIDIA CUDA 提供的一种**图执行**机制：把一系列 CUDA kernel 调用预先录制为一个静态图，之后整图一次性提交给 GPU，省去每次 launch kernel 的 CPU-GPU 同步与 Python 调度开销。

## 核心要点

1. **录制 + 重放**: `torch.cuda.graph(...)` 捕获 → `graph.replay()`
2. **消除 dispatch**: 对于固定形状、固定调用序列的推理特别有效
3. **延迟收益**: 可省 10-50 ms 的 CPU 开销
4. **限制**: 输入形状必须固定，控制流不能动态变化

## 代表工作

- [[WLA]]: 用 CUDA Graph + 算子融合把推理延迟从 116 ms 压到 ~40 ms
- [[TensorRT-LLM]]: 推理框架内置 CUDA Graph 支持
- [[vLLM]]: 部分场景使用 CUDA Graph

## 相关概念

- [[Triton Kernel]]
- [[KV Cache]]
- [[Operator Fusion]]
