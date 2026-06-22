---
type: concept
aliases: [Hamster robot policy]
---

# Hamster

## 定义

Hamster 是 GeneralVLA-2 评测中使用的机器人操作对比基线方法，在部分任务（Open_jar: 77.67%，Lamp_on: 61%）上表现较强，但在大多数需要精细操作的任务上成功率为 0。

## 核心要点

1. **部分任务优势**: 在 Open_jar 和 Lamp_on 等特定任务上具有竞争力
2. **覆盖范围有限**: 仅覆盖约 7-10 个 RLBench 任务，无法处理全部 14 个任务
3. **对比意义**: 展示了专用方法与通用 VLA 系统的任务覆盖率差距

## 代表工作

- [[GeneralVLA2]] 中作为对比基线

## 相关概念

- [[VLA]]
- [[VoxPoser]]
- [[CAP]]
