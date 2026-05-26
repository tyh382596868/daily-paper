---
type: concept
aliases: [Camera-Controlled Video Generation, 相机控制视频生成, 6-DoF 视频生成]
---

# Camera-Controlled 视频生成

## 定义

接收显式 **6-DoF 相机位姿轨迹** 作为条件输入的视频生成模型。在 [[交互式世界模型]] 评测中作为一类范式，主打几何控制精度。

## 核心要点

1. **输入**: 文本/初始图像 + 6-DoF 相机轨迹序列
2. **优势**: 导航精度高，[[Navigation Score]] 通常领先
3. **劣势**: 语义理解相对弱，事件编辑/主体动作能力受限
4. **常见模型**: LingBot-World, HY-World, Fantasy-World, InSpatio-World, Astra
5. **关键工程教训**: 相机控制能力 **不等于** 视角一致性维持能力（[[WBench]] 测得相关性近零）

## 代表工作

- [[WBench]] 评测中: LingBot-World 在 Consistency 89.9 / Video Quality 78.9 综合最强
- HY-World 1.5 在 Navigation 87.5 领跑

## 相关概念

- [[交互式世界模型]]
- [[文本驱动视频生成]]
- [[Action-Conditioned World Model]]
- [[1-生成模型]]
