---
type: concept
aliases: [mimic hand, mimic 机器手, mimic robotics hand]
---

# mimic Hand

## 定义

mimic Hand 是由 mimic robotics 公司开发的一款 16 自由度腱驱动仿人灵巧机器手，设计目标是以高保真度模仿人手的灵巧操作能力，同时保持与工业机械臂的兼容性。

## 核心要点

1. **自由度**: 16 DoF，动作表示为 16 维关节角度向量
2. **驱动方式**: 腱驱动（tendon-driven），轻量化设计
3. **特别配置**: 在 [[LAD]] 实验中，mimic 手配备了腕部 RGB 相机，以提供更丰富的观测信息
4. **安装平台**: 通常安装在 Franka Emika 机械臂末端

## 代表工作

- [[LAD]]: mimic 手作为跨本体学习的一个本体，在积木拾放任务中通过 LAD 跨本体策略提升 +13% 成功率
- [[mimic-one]]: mimic robotics 针对灵巧操作的通用模型配方，专门为 mimic 手设计

## 相关概念

- [[Cross-Embodiment Learning]]
- [[Dex-Retargeting]]
- [[Faive Hand]]
- [[Franka 研究臂]]
