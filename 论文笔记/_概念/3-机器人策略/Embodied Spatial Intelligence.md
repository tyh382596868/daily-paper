---
type: concept
aliases: [具身空间智能, Spatial Intelligence in VLA, 3D Spatial Reasoning]
---

# Embodied Spatial Intelligence

## 定义
具身智能体（robot、VLA、agent）对**3D 空间**进行感知、推理、规划的综合能力。包含三个层次：
1. **几何感知**: 深度、表面、相对距离、物体尺寸
2. **空间推理**: 相对位置关系、遮挡关系、可达性
3. **动作映射**: 把空间理解转换为正确的动作执行（最容易出现 [[Action Shortcut|动作捷径]]）

## 核心要点
1. **三层能力可解耦**: [[3DThinkVLA]] 主张几何感知（低层）与空间推理（高层）应分别注入 VLM 不同位置
2. **评估方式**:
   - LIBERO-Spatial / LIBERO-Plus（视角扰动）
   - 真机 height-variation、transparent-container、precise-positioning 任务
3. **常见注入方式**:
   - 显式 3D 输入: point cloud、depth map（[[SpatialVLA]]、[[DP3]]）
   - 隐式潜空间蒸馏: 对齐 3D 基础模型（[[3DThinkVLA]]、[[VGGT]]）
   - 显式 CoT 推理: 生成空间推理文本再出动作（[[CoT-VLA]]）

## 代表工作
- [[3DThinkVLA]]: 隐式注入，部署轻量
- [[SpatialVLA]]: 显式 3D 输入
- [[CoT-VLA]]: 显式 CoT
- [[Embodied Reasoning]]: 更广义的具身推理框架

## 相关概念
- [[Embodied Reasoning]]
- [[VLA]]
- [[VGGT]]
