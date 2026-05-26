---
type: concept
aliases: [LDM, Large Drive Model]
---

# Large Drive Model (LDM)

## 定义

LDM 是 [[X-Foresight]] 提出的统一 token 自回归模型：把语言指令、多相机观测、自车动作 / 状态、查询 token 编码到**同一序列**，用 [[Transformer]] 做 [[Chunk-Wise Autoregression|chunk-wise 自回归]] 预测，同时输出未来相机 token 与未来动作 token。

## 多模态 Prompt 结构

$$
[\text{SYSTEM}] \;|\; [l_0, O_0, A_0, Q_0] \;|\; [l_1, O_1, A_1, Q_1] \;|\; \cdots \;|\; [l_i, O_i, A_i, Q_i]
$$

- **System prompt**: 全局导航目标 + 自车元信息（对所有 chunk 可见的 sink token）
- **$l_i$**: 文本指令（指定预测视野）
- **$O_i$**: [[ViT]] 编码的多相机观测 token
- **$A_i$**: 自车动作 / 状态 token
- **$Q_i$**: 查询 token，触发下一 chunk 预测

## 核心要点

1. **统一 token 空间**: vision + language + action 同一序列，因果耦合而非并行解耦
2. **Chunk-wise**: 每个 chunk = 1 秒，避免帧级 trivial 外推
3. **半因果 Block Sparse Attention**: chunk 内双向，chunk 间块稀疏 + 位置对应邻域
4. **联合训练**: $\mathcal{L}_{total} = \mathcal{L}_{act} + \alpha \mathcal{L}_{cam} + \beta \mathcal{L}_{bev}$
5. **训练用 [[Teacher Forcing]]**: 推理时 chunk-by-chunk 滚动

## 代表工作

- [[X-Foresight]]: 首次提出，配 1024 GPUs / 280K 小时驾驶数据训练

## 相关概念

- [[VLA]]
- [[Chunk-Wise Autoregression]]
- [[Block Sparse Attention]]
- [[Vision Renderer]]: LDM 输出的 camera token 由其解码为像素
- [[World Action Model]]
