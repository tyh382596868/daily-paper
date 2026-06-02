---
type: concept
aliases: [可扩展机器人数据采集, scalable robot data collection]
---

# scalable robot data collection

## 定义

让真实机器人演示数据的采集速率/成本-收益比足以支撑大规模 [[VLA]] / [[Diffusion Policy]] 训练的工程与算法路线。瓶颈点在于：**人工遥操作时长 + 物理 reset + 场景/物体多样性**。

## 核心要点

1. 路线一：**远程众包平台**（如 [[DROID]]）— 多实验室协作
2. 路线二：**视觉增广**（[[ROSIE]] / [[RoboEnvision]]）— in-place 换皮
3. 路线三：**世界模型合成**（[[RoboDream]] / DreamGen）— 真正合成新物理配置
4. 路线四：**Prop-Free 遥操作**（[[RoboDream]] 提出）— 无道具采集 + 后期合成物体，2.2× 提速
5. 路线五：**模仿 / 学习视频** — 从 YouTube 等无标签视频提取动作监督

## 代表工作

- [[DROID]]: 跨实验室众包路线代表
- [[RoboDream]]: 同时贡献了合成路线 + Prop-Free 路线
- [[AdaWorld]]: 用 latent action 桥接无标签视频与 embodiment

## 相关概念

- [[DROID]]
- [[Compositional World Models]]
- [[ROSIE]]
- [[RoboEnvision]]
