---
type: concept
aliases: [OmniPaint]
---

# OmniPaint

## 定义

基于扩散模型的通用图像 inpainting / 物体擦除模型，给定 mask 与背景图后能填补出语义、光照、纹理一致的内容。

## 核心要点

1. Diffusion-based inpainting，对任意 mask 形状都有较好的边缘融合
2. 在机器人数据合成里常用于"清场" — 把桌面物体抹除得到干净背景
3. 与 [[Grounded-SAM]] 串联即可自动化构造「场景先验」
4. 区别于早期 LaMa 类纯 GAN inpainting，更擅长大区域 + 复杂纹理

## 代表工作

- [[RoboDream]]: 用 OmniPaint 把首帧物体抹除得到 scene prior $I_s$

## 相关概念

- [[扩散模型]]
- [[图像编辑模型]]
- [[Grounded-SAM]]
