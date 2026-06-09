---
type: concept
aliases: [Sim-to-Real, sim-to-real, sim2real, 仿真到真实, 仿真迁移]
---

# Sim-to-Real

## 定义
Sim-to-Real 是指在仿真环境中训练机器人策略，然后将其直接部署到真实机器人的范式，核心挑战是弥合"仿真-现实差距"（sim-to-real gap）——仿真与现实在物理动力学、感知噪声、接触模型和视觉外观上的差异。

> 注：概念库中已有 [[sim-to-real]] 文件（小写），本条目为其规范化标题版本，内容一致。

## 核心要点
1. **主要手段**:
   - **域随机化（Domain Randomization）**: 在仿真中随机化物理参数（摩擦、质量）和视觉参数（纹理、光照），使策略对变化鲁棒
   - **系统辨识（System Identification）**: 从真实数据中辨识准确的物理参数，填回仿真
   - **域自适应（Domain Adaptation）**: 用少量真实数据做特征对齐
2. **视觉 Sim-to-Real**: 渲染风格差距是主要瓶颈；视觉域随机化 + 相机对齐是常用方案
3. **物理 Sim-to-Real**: 接触和动力学建模不准是核心挑战；高保真仿真器（Isaac Lab）有所缓解

## 代表工作
- [[GRAIL]]: 使用视觉域随机化 + 相机对齐，将 SONIC 追踪策略蒸馏为自中心视觉策略，实现仿真生成数据到真实 G1 的迁移
- [[sim-to-real]] (概念库已有): 更详细的 real-to-sim 分支描述

## 相关概念
- [[IsaacLab]]: 高保真 GPU 加速物理仿真平台，GRAIL Sim-to-Real 的训练环境
- [[behavior cloning]]: Sim-to-Real 中常用的策略蒸馏方法
- [[Loco-Manipulation]]: Sim-to-Real 的主要应用场景之一
- [[视频基础模型]]: 合成数据生成（如 GRAIL）与 Sim-to-Real 结合的新路径
