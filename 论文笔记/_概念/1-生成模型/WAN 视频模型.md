---
type: concept
aliases: [WAN, Wan Video Model, 万象视频模型]
---

# WAN 视频模型

## 定义

WAN（万象）是阿里巴巴开源的大规模视频生成基础模型，支持文本/图像到视频生成，在多项基准上达到开源最优水平，是多个机器人操作世界模型（如 [[CtrlWorld]]、[[MemWorld]]）的视频扩散骨干。

## 核心要点

1. **开源优势**: 完整权重公开，可作为机器人领域动作条件视频生成的预训练骨干
2. **架构**: 基于 [[视频扩散模型]] 的 Transformer（DiT）架构，支持多帧一致生成
3. **机器人操作应用**: Ctrl-World 在 WAN 基础上添加动作条件，MemWorld 进一步添加 W-VMem 几何记忆
4. **多视角扩展**: CtrlWorld/MemWorld 将 WAN 扩展为同时生成腕部视角 + 两个第三视角的多视角输出

## 代表工作

- [[CtrlWorld]]: 以 WAN 为骨干的动作条件机器人世界模型
- [[MemWorld]]: 在 CtrlWorld（WAN 骨干）基础上添加 W-VMem 记忆模块

## 相关概念

- [[视频扩散模型]]
- [[Action-Conditioned World Model]]
- [[CtrlWorld]]
- [[1-生成模型]]
