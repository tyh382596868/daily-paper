---
type: concept
aliases: [3D Diffusion Policy, DP3 Encoder]
---

# DP3

## 定义
3D Diffusion Policy，把 [[Diffusion Policy]] 的输入从 2D 图像扩展为 **3D 点云**的策略学习方法。提出可学习的 DP3 Encoder（轻量 MLP-based 点云编码器）将点云映射为紧凑表征，再喂给 diffusion-based action head。

## 核心要点
1. **输入**: 单视角 RGBD → 点云（数千点的子采样）
2. **DP3 Encoder**: 比 PointNet++ / PTv3 更轻量，但对点云噪声敏感
3. **动作生成**: 基于 [[Diffusion Policy]] 的 DDPM/DDIM 去噪
4. **优势**: 在精细操作（如插针、装配）上数据效率高于 2D 输入策略
5. **劣势**:
   - 依赖深度传感器
   - [[3DThinkVLA]] Table 8 显示在 LIBERO 上 DP3 Encoder 不如隐式 [[Geometry Adapter]]（96.5 vs 97.6）

## 代表工作
- DP3 原论文（NeurIPS 2024）
- [[3DThinkVLA]]: 将 DP3 Encoder 作为 explicit 3D 注入的对照

## 相关概念
- [[Diffusion Policy]]
- [[Point Transformer]]
- [[Geometry Adapter]]
