---
type: concept
aliases: [WAM, World Action Model, 世界动作模型]
---

# World Action Model

## 定义

World Action Model（WAM）是一类将场景演化预测（世界模型）与机器人动作预测（动作策略）联合建模的框架，通过同时预测未来场景状态和当前动作来增强 VLA 策略的泛化能力。

## 核心要点

1. **联合建模**: 同时预测 $(\mathbf{A}_t, \hat{\mathcal{S}}_{t+1})$，动作预测从世界预测中获益
2. **辅助监督**: 世界预测提供额外的自监督信号，减少对大规模动作标注数据的依赖
3. **分类**:
   - **整体式 WAM**（Holistic WAM）：预测像素级未来或全局潜变量（如 WorldVLA、Cosmos-Policy）
   - **对象中心 WAM**（Object-Centric WAM）：对每个对象槽分别预测状态（如 OA-WAM）

## 局限性

整体式 WAM 将对象身份与外观纠缠在全局 token 中，场景扰动（视角、布局、光照变化）导致身份漂移，策略鲁棒性差。

## 代表工作

- [[OA-WAM]]: 引入对象可寻址性的 WAM，冻结地址向量实现稳定对象绑定
- [[WorldVLA]]: 整体式 WAM 的典型代表
- [[Cosmos-Policy]]: 基于 Cosmos 世界模型的整体式 WAM
- [[VLA-JEPA]]: 采用 JEPA 式预测目标的 WAM 变体
- [[RLA-WM]]: 极简 WAM——给 BC-ResNet 加线性头预测残差潜在动作（RLA），从无动作视频学策略，不耦合 DINO/视频生成 backbone
- [[DAWN]]: 把 WAM 推广为 [[WAIM]]（World-Action Interactive Model），通过 [[World Predictor]] 与 [[World-Conditioned Action Denoiser]] 的递归交互在自动驾驶 NAVSIM v1 perception-free 区取得 SOTA

## 相关概念

- [[VLA]]
- [[Flow Matching]]
- [[World Model]]
