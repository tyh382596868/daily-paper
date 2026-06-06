---
type: concept
aliases: [Generative Motion Model]
---

# GENMO

## 定义
GENMO 是一类生成式人体动作建模方法，基于扩散 / autoregressive 框架学习从文本、音乐、场景等条件到长序列 [[SMPL]] 动作的映射。通常作为 humanoid 数据生成的上游模块，为下游策略蒸馏（[[ResMimic]] 等）提供高质量参考轨迹。

## 核心要点
1. **任务范畴**：text-to-motion、music-to-motion、scene-aware motion synthesis
2. **架构**：transformer / DiT 主干 + diffusion 训练 + 可选的 ControlNet-like 条件注入
3. **训练数据**：AMASS、HumanML3D、MotionX 系列 + 自标注扩增
4. **与 [[CHOIS]] 区别**：GENMO 更偏纯人体动作生成，CHOIS 强调 human-object 交互（带物体位姿）

## 代表工作
- GENMO 系列工作（具体引用见 [[GRAIL]] 文中 reference）
- 应用：[[GRAIL]] 在 pipeline 中借鉴 GENMO 类方法生成参考轨迹

## 相关概念
- [[SMPL]]
- [[CHOIS]]
- [[HOI]]
- [[MPJPE]]
