---
type: concept
aliases: [Bootstrap Your Own Latent, BYOL]
---

# BYOL

## 定义
自监督表征学习方法，用在线网络（online network）预测动量网络（momentum/target network）的潜变量表示，无需负样本，通过 stop-gradient 防止表征坍塌。

## 数学形式
$$\mathcal{L} = \| q_\theta(z_\theta) - z_{\xi} \|_2^2$$

其中 $z_\theta$ 为在线网络编码，$z_\xi$ 为动量网络（EMA 更新）编码，$q_\theta$ 为预测头。

## 核心要点
1. 动量更新：$\xi \leftarrow \tau \xi + (1-\tau)\theta$，目标网络参数缓慢跟随在线网络
2. 无需负样本：通过预测器（predictor）和 stop-gradient 避免表征坍塌
3. 两路视图增强：同一图像两种增强视图分别送入两路网络

## 代表工作
- [[BYOL]]: Grill et al., 2020，提出原始方法
- [[DynaWM]]: 将动量目标思想引入 Teacher-Student 蒸馏，防止 locomotion 策略的表征坍塌

## 相关概念
- [[EMA]] — 动量网络参数更新方式
- [[stop-gradient]] — 防止 collapse 的关键操作
- [[Knowledge Distillation]] — 动量目标蒸馏的相关思想
