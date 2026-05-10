---
type: concept
aliases: [Cosmos Policy]
---

# Cosmos-Policy

## 定义

Cosmos-Policy 是基于 NVIDIA Cosmos 世界基础模型的机器人操作策略，通过像素空间的视频预测（World Action Model 范式）来辅助策略学习，提升操作鲁棒性。

## 核心要点

1. **WAM 范式**: 将世界模型与动作策略结合，通过预测未来视频帧提供额外监督信号
2. **全局 token**: 使用整体视频帧 token，不做对象级分解
3. **LIBERO-Plus 对比**: 在几何扰动轴（Geo-Avg 73.8%）弱于 OA-WAM（84.3%），表明全局 token 的对象可寻址性不足

## 代表工作

- [[OA-WAM]]: 将 Cosmos-Policy 作为 WAM 基线进行对比

## 相关概念

- [[World Action Model]]
- [[VLA]]
- [[OA-WAM]]
