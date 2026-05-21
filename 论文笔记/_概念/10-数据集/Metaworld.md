---
type: concept
aliases: [MetaWorld, MW]
---

# Metaworld

## 定义
基于 [[MuJoCo]] 的多任务机器人操作 benchmark，包含 50 个 Sawyer 机械臂任务（按钮、抽屉、积木等），常用于元学习与世界模型评测。

## 核心要点
1. **任务集合**: ML1（单任务多目标）、ML10（10 任务训练 + 5 新任务测试）、ML45 等
2. **常用子任务**:
   - **MW-Reach (MW-R)**: 末端到达目标位置
   - **MW-Reach-Wall (MW-RW)**: 带障碍墙的 Reach
   - Button-Press、Drawer-Open、Pick-Place 等
3. **特点**:
   - 任务结构简单、奖励/代价面相对平滑
   - 适合做"小规模可控对比"，但与真实任务 gap 较大
4. **评测指标**: success rate（基于物体到目标距离阈值）

## 代表工作
- [[JEPA-WM]]: 在 MW-R / MW-RW 上做世界模型 ablation
- [[DINO-WM]]: 同样使用 Metaworld 评测

## 相关概念
- [[MuJoCo]]
- [[Push-T]]
- [[RoboCasa]]
- [[LIBERO]]
