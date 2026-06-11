---
type: concept
aliases: [GMR, General Motion Retargeting, 通用动作重定向]
---

# GMR (General Motion Retargeting)

## 定义
GMR 是 ICRA 2026 提出的通用动作重定向方法，能将人类运动（SMPL-X 格式）实时重定向到多种不同骨骼和形态的人形机器人，在 CPU 上即可实时运行，是 TWIST 等系统的核心重定向引擎。

## 核心要点
1. **通用性**: 不依赖特定机器人形态，可重定向到多种人形机器人（Unitree G1、H1 等）
2. **实时性**: 在 CPU 上实时运行，适合在线部署场景
3. **IK + 时序平滑**: 结合逆运动学求解器和时序平滑引擎，产出运动学可行且时序连贯的机器人轨迹
4. **与 GRAIL 集成**: GRAIL Pipeline 使用 GMR 将 SMPL-X 重建的人体动作重定向到 Unitree G1 骨骼

## 代表工作
- 原论文：YanjieZe et al., "GMR: General Motion Retargeting", ICRA 2026
- [[GRAIL]]：使用 GMR 作为 Stage 4 的动作重定向引擎，连接 4D HOI 重建与 SONIC 追踪策略训练
- TWIST：GMR 是其标配重定向器

## 相关概念
- [[SMPL-X]]: GMR 的输入人体动作格式
- [[Unitree G1]]: GRAIL 中 GMR 的目标机器人平台
- [[HOI]]: GMR 处理的动作来源于 4D HOI 重建
- [[GRAIL]]: GMR 的典型下游应用
