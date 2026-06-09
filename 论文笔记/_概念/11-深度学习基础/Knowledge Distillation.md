---
type: concept
aliases: [知识蒸馏, KD]
---

# Knowledge Distillation

## 定义
让一个学生（student）模型在 teacher 模型的指导下学习，目的是把 teacher 的能力/知识迁移到结构更简单或输入更受限的 student 上。监督信号可以是 logits（soft label）、隐藏态、特征图、注意力图等。

## 数学形式

经典 KD（Hinton 2015）：
$$
\mathcal{L}_{KD} = (1-\alpha)\,\mathcal{L}_{CE}(y, p_S) + \alpha\,T^2\,\mathrm{KL}(p_T^{(T)} \| p_S^{(T)})
$$
其中 $T$ 是 temperature，$p_T^{(T)}, p_S^{(T)}$ 是 teacher / student 在 temperature $T$ 下的 softmax 概率。

潜空间 KD（如本文）：
$$
\mathcal{L}_{latent} = 1 - \mathcal{S}(h_T, \mathcal{R}(h_S))
$$

## 核心要点
1. **Logits 蒸馏**: 用 soft label 包含类间相对关系
2. **Feature 蒸馏**: 在中间层做特征匹配（FitNets、Hint Learning）
3. **Latent / Anchor 蒸馏**: 在 hidden state 上做对齐，常配 cosine 距离
4. **Online vs Offline**: online 共享 backbone 参数，开销低且训练时实时同步
5. 在 VLA 中常用于把 teacher 路径（带 explicit CoT prompt）的空间/规划能力压进 student 路径（直接出动作）

## 代表工作
- Hinton et al., 2015: 原始 KD 论文
- [[CoT-VLA]]: explicit CoT 文本蒸馏
- [[3DThinkVLA]]: latent reasoning anchor 蒸馏
- [[HANDOFF]]: context 条件化多 Teacher KL 蒸馏（velocity-gated 凸组合 + recovery-masked），用于人形机器人全身控制

## 相关概念
- [[Latent Distillation]]
- [[Cosine Similarity]]
- [[Co-training]]
