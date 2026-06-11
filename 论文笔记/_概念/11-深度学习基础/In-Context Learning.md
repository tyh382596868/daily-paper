---
type: concept
aliases: [in-context learning, ICL, 上下文学习]
---

# In-Context Learning (ICL)

## 定义
在推理阶段通过在输入 prompt 中提供少量示例（input-output pairs），使模型无需参数更新即可完成新任务的能力。

## 数学形式
$$p(y|x, \{(x_1,y_1), ..., (x_k,y_k)\}) \approx p(y|x, \text{context})$$

## 核心要点
1. 无需 fine-tune，只需在 prompt 中添加 k 个示例（few-shot）
2. 示例质量和顺序对性能有显著影响
3. 本质是利用预训练模型的隐式元学习能力
4. 在大模型（LLM/VLM）中尤为强大，小模型效果有限

## 代表工作
- [[VICX]]: 将 in-context learning 用于 V2T-ICON，用检索到的 image-state pairs 作为 prompt
- GPT-3: 最早系统展示 ICL 能力的工作

## 相关概念
- [[Transformer]]
- [[Few-Shot Learning]]
- [[V2T-ICON]]
