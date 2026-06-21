---
name: paper-deep-dive
description: |
  把一篇论文「讲透」的精读模式。当用户给出一篇感兴趣的论文（arXiv 链接 / PDF 路径 / 标题），
  并希望深入讲解、吃透原理时使用。输出单个教学式深度笔记到 `论文精读/<方法名>.md`。

  **触发词**: "精读 XXX"、"精读这篇"、"讲透这篇论文"、"深入讲解这篇"、"帮我吃透 XXX"、
  "把这篇讲明白"、"deep dive"、"精讲 XXX"。

  与 paper-reader（"读一下 XXX"）的区别：paper-reader 产出结构化归档笔记，存到
  `论文笔记/分类/`；本 skill 产出更深、偏教学讲解的单文件，存到 `论文精读/`，
  面向「从零把这篇论文的动机、推导、设计取舍全部讲懂」。
context: fork
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch, WebSearch
---

> **开始前**: 先跟用户打个招呼，说一声「开始精读 📖」并复述要读的论文标题/链接。

# 论文精读 (Paper Deep Dive)

把一篇论文当作教材讲透：不只是「写了什么」，而是「为什么这样做、凭什么有效、怎么推出来的、换我会怎么做」。
读者设定为**刚进入这个方向的研究生**——默认他没读过这篇的前置工作，需要你把直觉和推导补全。

## Step 0: 读取共享配置

先读取 `../_shared/user-config.json`，如果 `../_shared/user-config.local.json` 存在，再用它覆盖默认值。

显式生成并在后续统一使用这些变量：

- `VAULT_PATH`
- `NOTES_PATH = {VAULT_PATH}/{paper_notes_folder}`
- `CONCEPTS_PATH = {NOTES_PATH}/{concepts_folder}`
- `DEEP_READING_PATH = {VAULT_PATH}/{deep_reading_folder}`（即 `论文精读/`）
- `AUTO_REFRESH_INDEXES`
- `GIT_COMMIT_ENABLED`
- `GIT_PUSH_ENABLED`（只有 `GIT_COMMIT_ENABLED=true` 时才可能为真）

## 1. 接收论文

| 输入方式 | 处理方法 |
|----------|----------|
| arXiv 链接 `https://arxiv.org/abs/xxxx` | WebFetch `https://arxiv.org/html/{arxiv_id}`（首选 HTML，含图）；失败再 fetch abs 页 |
| PDF 路径 `/path/to/paper.pdf` | 直接 Read |
| 标题 / 模型名 | 先 WebSearch `"{标题} arxiv"` 定位 arXiv ID，再按 arXiv 链接流程 |

> 沙箱环境下 arXiv 可能返回 403。此时退回 WebSearch 收集足够信息，并在 frontmatter
> 标 `source: websearch`、图片标 `image_source: pending`（见 §5）。

**先通读全文**再动笔。统计论文的 Figure / Table / 公式总数，精读笔记要把它们都讲到，不能漏。

## 2. 精读笔记结构（严格遵循模板）

模板：`assets/deep-dive-template.md`。逐节填充，**不可删节、不可用「同上」糊弄**。精读笔记通常比
普通笔记长 1.5–3 倍，这是预期，不是负担。各节要点：

- **TL;DR**：30 秒读完——解决什么问题、核心 idea 一句话、为什么有效、最强的一个结果数字。
- **阅读前置**：列出读懂本文必须先懂的概念，每个用 `[[概念]]` 链接；读者可能不熟的，补 1–2 句直觉。
- **问题是什么 & 为什么难**：任务设定（输入/输出/评价指标）→ 朴素做法为什么不行 → 本文的切入点。
- **核心思想（讲直觉）**：用一个类比或一个具体例子从头走一遍，说清这个思路**为什么**能解决上面的难点。这是精读的灵魂，不能省。
- **方法逐层拆解**：先画数据流（输入→…→输出），再逐个模块讲「是什么 / 为什么这样设计 / 怎么实现」。每个模块尽量回答：**去掉它会怎样？为什么不用更简单的替代？**
- **关键公式逐步推导**：每条公式都要「直觉先行 → LaTeX → 逐符号解释 → 由来/推导 → 极端值代入自检」。论文里的重要公式一条不漏。
- **关键图表精读**：每张图讲清它在论证什么，不是配图。
- **实验：到底证明了什么**：主结果（数字说明什么）+ 消融（哪个组件最关键）+ 局限实验。
- **复现思路**：伪代码/训练流程、关键超参与算力、容易踩的坑。
- **批判与延伸**：真创新 vs 包装、站不住的 claim / 评估盲区、可以接着做的 research idea。
- **常见疑问 Q&A**：3–6 条读者读到一半最可能卡住的问题，自问自答。

## 3. 复用 paper-reader 的质量规范

以下规则与 paper-reader 完全一致，**直接套用，不要另起一套**：

- **公式**：每条含名称、`$$` 块（前后留空行否则 Obsidian 不渲染）、含义、符号说明。
  ⚠️ GitHub KaTeX 兼容：**禁用 `\operatorname` / 自定义 `\def`**，改用 `\mathrm` / `\mathop` / `\text`。
  详见 `../paper-reader/references/quality-standards.md`。
- **图片**：多源 fallback 取真实 URL（arXiv HTML `<figure>` img src → 项目主页 → PDF 提取）。
  ⛔ **禁止编造图片 URL**（如 `/figures/method.png`）。抓不到就用占位行：
  `> 🖼️ **Figure X: 标题** — 图片暂缺（原图见 [arXiv HTML](https://arxiv.org/html/{arxiv_id})）`
  并在 frontmatter 标 `image_source: pending`。详见 `../paper-reader/references/image-troubleshooting.md`。
- **内联概念链接**：正文首次出现的技术术语用 `[[概念]]` 链接，不只在结尾堆。

## 4. 保存到 Obsidian

### 文件命名与路径

- 文件名只用**方法名/模型名**：`{方法名}.md`（如 `Pi05.md`），希腊字母转 ASCII，不加年份前缀。
  方法名判断：标题冒号前 / Abstract「We propose XXX」。
- 路径**扁平**保存到 `{DEEP_READING_PATH}/{方法名}.md`（不分子目录——精读区就是一个平铺文件夹）。
- 若同名文件已存在：先 Read 检查。若是骨架/旧版（行数少、缺章节）则覆盖重写；若已是完整精读则告诉用户已存在、问是否重写。

### frontmatter

```yaml
---
type: deep-reading
title: "论文标题"
method_name: "MethodName"
authors: [Author1, Author2]
year: 2026
venue: arXiv
arxiv: https://arxiv.org/abs/xxxx
tags: [tag1, tag2]   # 小写连字符，3-8 个，第一个是最核心主题
image_source: online # online / mixed / pending
source: arxiv-html   # arxiv-html / pdf / websearch
created: YYYY-MM-DD
---
```

### 保存后图片可达性检查（有嵌图时）

```bash
python3 ../daily-papers/download_note_images.py "{笔记完整路径}"
```
不可达外链自动下载到本地并替换为 wikilink；有本地化则 frontmatter `image_source` 改为 `mixed`。

## 5. 概念库维护（每篇必做）

与 paper-reader 一致：扫描笔记中所有 `[[概念]]` 链接 → 检查 `{CONCEPTS_PATH}/` 各子目录是否已存在 →
对缺失概念自动归类创建（分类规则 + 模板见 `../paper-reader/references/concept-categories.md`），
并把本论文列入该概念的「代表工作」。**不可跳过缺失概念。**

## 6. 刷新索引 + Git（按开关）

1. 仅当 `AUTO_REFRESH_INDEXES=true`：
   ```bash
   python3 ../_shared/generate_concept_mocs.py
   python3 ../_shared/generate_paper_mocs.py
   ```
   > 注：`generate_paper_mocs.py` 默认只扫 `论文笔记/`。`论文精读/` 是否进入 MOC 取决于脚本配置，
   > 不进也无妨——精读区文件少、平铺，本身就好找。
2. 仅当 `GIT_COMMIT_ENABLED=true` 且 `VAULT_PATH/.git` 存在、`git add` 后确有 staged changes：
   ```bash
   cd {VAULT_PATH} && git add "{deep_reading_folder}/{方法名}.md" "{paper_notes_folder}/" && git commit -m "deep dive: {方法名}"
   ```
   只有 `GIT_PUSH_ENABLED=true` 且配置了远端时才 `git push`。

## 7. 完成后自检

- [ ] 模板每一节都填了，没有空节 / 没用「同上」糊弄？
- [ ] 「核心思想」一节真的用直觉/类比讲清了**为什么有效**，不是复述摘要？
- [ ] 论文所有重要公式都在，且每条有「由来/推导」而不只是抄公式？
- [ ] 所有 Figure/Table 都讲到（数量与论文一致）？图片是真实 URL 或占位行，**没有编造路径**？
- [ ] 公式不含 `\operatorname` 等 GitHub 禁用宏？
- [ ] 正文技术术语有 `[[概念]]` 内联链接，缺失概念已创建？
- [ ] 文件存到 `论文精读/{方法名}.md`？

## 输出

完成后告诉用户：精读笔记路径、新增概念数、以及一句话「这篇到底值不值得深挖」的判断。

## 注意事项

- 这是**单篇深读**，一次只处理一篇；用户给多篇时逐篇问是否都要精读。
- 与 paper-reader 的分工：归档式速记走 `读一下`，吃透原理走 `精读`。两者可以并存，互不覆盖文件。
- 默认不做 git，除非配置开启。
