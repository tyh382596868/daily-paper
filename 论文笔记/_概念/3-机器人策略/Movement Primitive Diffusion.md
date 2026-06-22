---
type: concept
aliases: [MPD, Movement Primitive Diffusion, 运动基元扩散]
---

# Movement Primitive Diffusion

## 定义

Movement Primitive Diffusion（MPD）是一种将扩散模型与运动基元（Movement Primitive）相结合的机器人操作策略方法，通过参数化运动基元生成平滑轨迹。

## 核心要点

1. **运动基元参数化**: 用 ProDMP 等运动基元表示轨迹，固有平滑性来自基函数本身
2. **平滑性有保证**: 比标准 Diffusion Policy 平滑（LDLJ 较好），但成功率在部分任务上不稳定
3. **外科任务适用**: 在 LapGym 实验中，MPD 在软体操作（BTM 任务）中平滑性次优
4. **无频率异构处理**: 与 SFP 类似，未解决控制频率异构问题

## 代表工作

- Scheikl et al., 2024: "Movement Primitive Diffusion: Learning Gentle Robot Manipulation of Deformable Linear Objects"
- [[FAFM]]（Guo et al., 2026）: 以 MPD 为基线，FAFM 在 LapGym 全部任务上超越 MPD

## 相关概念

- [[Diffusion Policy]]
- [[Action Chunking]]
- [[Flow Matching]]
- [[FAFM]]
