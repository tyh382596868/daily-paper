---
type: concept
aliases: [Classifier-Free Guidance, 无分类器引导]
---

# CFG (Classifier-Free Guidance)

## 定义
CFG 是扩散模型中一种"训练时随机丢掉条件、推理时把条件分支与无条件分支线性外推"的引导方法，避免了 Classifier Guidance 需要额外训练分类器的成本，同时显著提升条件生成的质量与可控性。

## 数学形式
训练时以概率 $p_\mathrm{drop}$（通常 0.1~0.2）把条件 $c$ 替换为空集 $\varnothing$，得到同一个网络 $\epsilon_\theta(x_t, t, c)$ 既能预测条件噪声也能预测无条件噪声。推理时：
$$\tilde{\epsilon}_\theta(x_t, t, c) = (1+w)\,\epsilon_\theta(x_t, t, c) - w\,\epsilon_\theta(x_t, t, \varnothing)$$

其中 $w \geq 0$ 是 guidance scale，$w=0$ 时退化为条件采样，$w$ 越大条件遵循度越高（但同时多样性下降、容易出现 over-saturation）。

## 核心要点
1. **训练-推理双轨**：训练时学一个"既会条件又会无条件"的网络，推理时手动外推
2. **Scale 是个艺术参数**：$w$ 太小条件弱、太大容易过饱和或失真；视频生成里通常 $w \in [3, 12]$
3. **空集 token 设计**：文本条件常用 null embedding 或 learned null token
4. **缺点**：推理时要跑两次前向（条件 + 无条件），计算开销翻倍——后续有大量工作（如 CFG-Zero、CFG distillation）想去掉这步

## 代表工作
- 原论文：Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPS 2021 workshop
- 应用：[[LongCat-Video]]、Stable Diffusion 系列、视频扩散模型全家桶

## 相关概念
- [[Score Distillation]]
- [[DMD]]
- [[DiT]]
