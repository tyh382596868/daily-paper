---
type: concept
aliases: [data poisoning, 数据毒化, 数据投毒]
---

# Data Poisoning（数据毒化）

## 定义
在训练数据集中注入精心设计的恶意样本，使训练后的模型在特定触发条件下产生攻击者预期的错误行为，而在正常输入上表现正常。

## 数学形式
$$\min_{\delta} \mathcal{L}_{train}(f_{\theta}; D \cup D_{poison}) \quad \text{s.t.} \quad f_{\theta}(x_{trigger}) = y_{target}$$

## 核心要点
1. 攻击者在训练集中注入少量毒化样本 $D_{poison}$
2. 模型训练后在正常输入上行为正常（隐蔽性）
3. 当输入包含特定 trigger 时才激活恶意行为（后门）
4. 在 robot world model 流水线中，毒化点在 world model 作为数据增强器时被激活

## 代表工作
- [[Targeting World Models]]: 证明 world model 在机器人学习流水线中引入了隐蔽的数据毒化入口

## 相关概念
- [[Backdoor Attack]]
- [[world model]]
