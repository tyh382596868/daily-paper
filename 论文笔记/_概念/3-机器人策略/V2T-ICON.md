---
type: concept
aliases: [V2T-ICON, Video-to-Trajectory In-Context Operator Network]
---

# V2T-ICON: Video-to-Trajectory In-Context Operator Network

## 定义
将视觉规划视频转化为机器人可执行关节轨迹的 in-context 算子网络，推理时通过检索的 image-state pairs 作为上下文 prompt，无需参数更新即可适应新任务。

## 数学形式
$$\tau = \text{V2T-ICON}(v_{plan}, \{(I_1, s_1), ..., (I_k, s_k)\})$$

其中 $v_{plan}$ 是视觉规划视频，$\{(I_i, s_i)\}$ 是检索到的 in-context 示例。

## 核心要点
1. 输入：分割提取的仅手臂帧（去掉场景背景噪声）+ in-context image-state pairs
2. 输出：机器人关节状态轨迹
3. 通过检索的配对示例实现 visual-to-state mapping，无需针对新任务 fine-tune
4. 是 [[VICX]] 系统中负责执行落地的核心模块

## 代表工作
- [[VICX]]: V2T-ICON 的提出论文

## 相关概念
- [[In-Context Learning]]
- [[VICX]]
- [[Diffusion Policy]]
