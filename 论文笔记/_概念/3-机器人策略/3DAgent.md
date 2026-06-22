---
type: concept
aliases: [3D规划智能体, 三维规划代理]
---

# 3DAgent

## 定义

3DAgent 是 GeneralVLA-2 系统中的层级规划主干，接收 3D 物体证据（来自 GeoFuse-MV3D）和记忆检索结果（来自 Governed KnowledgeBank），生成多阶段末端执行器轨迹以完成机器人操作任务。

## 数学形式

输出的末端执行器轨迹：

$$
\tau_t = \{(x_\ell, y_\ell, z_\ell, g_\ell)\}_{\ell=1}^{L_t}
$$

其中 $(x_\ell, y_\ell, z_\ell)$ 为 3D 坐标，$g_\ell \in \{0, 1\}$ 为夹爪状态，$L_t$ 为轨迹步数。

## 核心要点

1. **语言+3D 感知**: 联合处理语言指令和 3D 物体中心表示
2. **记忆增强**: 接收 KnowledgeBank 检索的操作经验作为上下文
3. **层级规划**: 先规划高层末端轨迹，再由低层控制器执行

## 代表工作

- [[GeneralVLA2]]: 3DAgent 的集成使用场景，training-free 操作在 RLBench 14 任务上全面领先

## 相关概念

- [[VLA]]
- [[GeoFuse-MV3D]]
- [[Governed KnowledgeBank]]
