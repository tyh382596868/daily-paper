---
type: concept
aliases: [Temporal Block Diffusion, 时序块扩散, TBD]
---

# Temporal Block Diffusion

## 定义
扩散模型推理加速技术：将连续若干时间步打包为"块"，整块只执行一次去噪，块内用插值补全，以减少扩散 VLA 的推理延迟。

## 核心要点
1. 牺牲块内精度换取 3-5x 推理加速
2. 块间边界处精度保持与单步相同
3. 配合跨帧时序表示压缩效果更好

## 代表工作
- [[TBD-VLA]]: arXiv 2606.07895

## 相关概念
- [[diffusion policy]]
- [[VLA]]
