---
type: concept
aliases: [Action-Conditioned World Model, 动作条件世界模型, 游戏世界模型]
---

# Action-Conditioned World Model

## 定义

接收**离散键盘/手柄动作**（如 WASD + 鼠标）作为条件输入的视频生成模型。常用于游戏类世界模型，强调对动作的即时响应。

## 核心要点

1. **输入**: 初始观测 + 离散动作序列 $\{a_t\}$（如 W/A/S/D + 鼠标增量）
2. **优势**: 实时响应能力强，适合游戏类闭环环境
3. **劣势**: 物理一致性与画质往往不及大型视频扩散模型
4. **常见模型**: Genie 3, Matrix-Game (2.0 / 3.0), HY-GameCraft, Happy Oyster, Infinite-World
5. **应用域**: 主要面向游戏世界（Minecraft、各类 FPS / TPS 游戏环境）

## 代表工作

- [[WBench]] 评测中: Matrix-Game 3.0 在 Navigation 83.5；Happy Oyster 综合最强
- Genie 3 (Google DeepMind): 标杆动作条件世界模型
- [[RoboScape]]: 物理感知机器人操作世界模型，在动作条件视频生成中引入深度和关键点物理先验

## 相关概念

- [[交互式世界模型]]
- [[文本驱动视频生成]]
- [[Camera-Controlled 视频生成]]
- [[1-生成模型]]
