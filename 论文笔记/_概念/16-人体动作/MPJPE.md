---
type: concept
aliases: [Mean Per Joint Position Error, 平均关节位置误差]
---

# MPJPE (Mean Per Joint Position Error)

## 定义
MPJPE 是 3D 人体姿态估计 / 动作生成 / humanoid 控制领域最常用的几何评估指标，衡量预测关节位置与真值关节位置之间的平均欧氏距离，单位通常是毫米（mm）。

## 数学形式
对一帧画面，设真值关节坐标 $\hat J = \{\hat J_i\}_{i=1}^{N}$ 和预测关节坐标 $J = \{J_i\}_{i=1}^{N}$（$N$ 为关节数，常用 17 或 24）：
$$\mathrm{MPJPE} = \frac{1}{N}\sum_{i=1}^{N} \big\| J_i - \hat J_i \big\|_2$$

对整段视频取所有帧的平均。

## 核心要点
1. **要先对齐根节点**：通常在 pelvis（骨盆）坐标上做平移对齐，避免全局平移误差掩盖局部姿态错误
2. **PA-MPJPE 变体**：在 MPJPE 基础上做 Procrustes 对齐（旋转+缩放），更宽松，反映纯姿态质量
3. **MPVE / MPJVE**：把 J 换成 mesh 顶点 V 或速度 $\dot J$，分别评价网格几何和动作平滑度
4. **典型阈值**：Human3.6M 上 sota 模型大约 30~50 mm，humanoid 控制场景下 100 mm 算可接受

## 代表工作
- Human3.6M / 3DPW / AMASS 等数据集的标准评估指标
- [[GRAIL]] 用 MPJPE / SR 评估生成的 humanoid loco-manipulation 质量
- [[SMPL]] 体系下的工作几乎全用 MPJPE

## 相关概念
- [[SMPL]]
- [[HOI]]
- [[ResMimic]]
