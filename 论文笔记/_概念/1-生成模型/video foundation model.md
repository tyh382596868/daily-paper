---
type: concept
aliases: [VFM, Video Foundation Model, 视频基础模型]
---

# Video Foundation Model

## 定义
在大规模视频数据上预训练的大型生成或表征模型，具备对时序动态、物理规律和视觉外观的通用理解能力，可作为下游视频生成、世界模型构建等任务的先验。

## 核心要点
1. 典型代表：Sora、HunyuanVideo、Wan2.1 等大规模视频扩散模型
2. 在机器人数据生成中作为「视频先验」（video prior）：提供物体运动、光照变化、场景过渡的合理性约束
3. 关键能力：时序一致性、物理合理性（soft physics prior）、语言-视觉对齐
4. [[GRAIL]] 将 VFM 的先验用于条件化仿人机器人数据合成，避免具身幻觉

## 代表工作
- [[GRAIL]]：以 VFM 作为视频先验，合成多样化的仿人 loco-manipulation 演示数据
- [[RoboDream]]：三路条件（轨迹 + 物体 + 场景）驱动视频扩散，利用 VFM 先验

## 相关概念
- [[视频扩散模型]]（常见 VFM 架构）
- [[World Model]]（VFM 可用于构建世界模型）
- [[embodiment hallucination]]（VFM 在机器人场景中的典型失败模式）
- [[sim-to-real]]（VFM 生成数据后的部署路径）
