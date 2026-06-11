---
type: concept
aliases: [WiLoR, Wild Hand Localization and Reconstruction]
---

# WiLoR (End-to-end 3D Hand Localization and Reconstruction in-the-wild)

## 定义

端到端的野外条件下 3D 手部定位与重建模型，输入 RGB 图像，直接输出 MANO 手部参数（手部姿态 + 形状），支持检测-重建一体化流程。

## 核心要点

1. 端到端：集成手部检测（定位）和 3D 重建，无需独立的手部检测前处理
2. 输出 MANO 参数，与 SMPL-X 的手部部分兼容，可直接插入全身估计流程
3. 在野外数据（遮挡、复杂背景）上具备较强鲁棒性
4. 常与 SMPL-X body 估计方法（HMR2）结合使用，形成全身重建流程

## 代表工作

- [[GRAIL]]: GEM-SMPL 中使用 WiLoR 提供 MANO 手部估计，补充 HMR2 在手部细节上的不足

## 相关概念

- [[MANO]]
- [[SMPL]]
- [[HMR2]]
