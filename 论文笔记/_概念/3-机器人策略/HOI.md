---
type: concept
aliases: [Human-Object Interaction, 人-物交互]
---

# HOI (Human-Object Interaction)

## 定义
Human-Object Interaction（HOI）泛指人和物体之间的接触、抓取、操作等交互行为；在 3D / robotics 语境下，HOI 任务通常是合成或重建出一段同时包含人体姿态、物体 6D 位姿和接触约束的运动序列。是 humanoid manipulation / VLA 数据扩增的关键中间表征。

## 核心要点
1. **三要素**：人体姿态（常用 [[SMPL]]）+ 物体 6D 位姿（位置 + 朝向）+ 接触约束（手/脚与物体的接触点和法向）
2. **数据来源**：
   - 传统 motion capture：GRAB、BEHAVE、CHAIRS、InterCap、OMOMO 等
   - 视频先验生成：从单目视频 + VLM 标注合成训练数据（[[GRAIL]] 路线）
   - 合成：基于物理仿真器（IsaacLab / MuJoCo）合成大规模 HOI 轨迹
3. **常见任务**：
   - HOI generation：给文本/物体生成动作（[[CHOIS]], [[GENMO]] 等）
   - HOI reconstruction：从视频恢复 SMPL + 物体位姿（FoundationPose 类）
   - HOI imitation：把 SMPL 轨迹蒸馏到 humanoid 控制策略（[[ResMimic]]）
4. **关键挑战**：接触约束的物理一致性（穿模、滑动、力反馈）

## 代表工作
- 数据集：GRAB（Taheri 2020）、BEHAVE（Bhatnagar 2022）、CHAIRS（Jiang 2022）、InterCap（Huang 2022）
- 生成：[[CHOIS]], [[GENMO]]
- 重建：[[FoundationPose]]
- 应用：[[GRAIL]] 把 HOI 合成数据用于 humanoid loco-manipulation 训练

## 相关概念
- [[SMPL]]
- [[FoundationPose]]
- [[ResMimic]]
- [[MPJPE]]
