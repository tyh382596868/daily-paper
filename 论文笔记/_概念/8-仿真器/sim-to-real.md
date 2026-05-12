---
type: concept
aliases: [sim-to-real, sim2real, 仿真到现实, Real2Sim2Real]
---

# sim-to-real

## 定义
在仿真器里训练（或大量预训练）策略 / 动力学模型，再迁移到真实机器人上部署的范式；核心难点是弥合"现实差距"（reality gap）——仿真与真实在动力学、感知、接触、噪声上的不一致。

## 核心要点
1. 常见手段：domain randomization（随机化仿真参数）、domain adaptation（特征对齐）、system identification（辨识真实参数回填仿真）、real-to-sim（从真实数据重建高保真仿真资产，再 sim-to-real）。
2. real2sim 这一支正在升温：用 3DGS / 物理孪生从真机数据重建场景与物体动力学，让仿真更"像真的"。
3. 评价不能只看仿真指标——必须报真机迁移后的成功率，否则容易自欺。

## 代表工作
- [[PhySPRING]]: 用 GNN 把从视觉重建出的物理孪生（spring-mass 系统）降维到"够用"的复杂度，加速 real-to-sim-to-real 的 forward-dynamics rollout。
- [[Sword]]: 关注 world model 当模拟器时对视觉风格扰动的鲁棒性（也是一种 sim-to-real 风格 gap 的体现）。

## 相关概念
- [[MuJoCo]]
- [[RoboTwin]]
- [[World Model]]
