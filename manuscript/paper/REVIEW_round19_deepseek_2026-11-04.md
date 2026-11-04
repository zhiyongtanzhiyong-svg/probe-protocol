# 第十九轮（外部）：投稿前最终检查 — DeepSeek

> 判定：可以提交（Minor Fixes），但有一处必须修正的 Cover Letter–Manuscript 不一致，以及三处需澄清的措辞。这一版相比上一版有实质性改进，且是十一轮中首次达到"可投稿"状态的版本。

## 一、必须修正（阻塞投稿）

### 1. Cover Letter 与 Manuscript 的 Data Availability 不一致
- **Manuscript**（DAS）："…available in the **public repository** `https://github.com/zhiyongtanzhiyong-svg/probe-protocol`"
- **Cover Letter**（Declarations 第 4 条）："…will be made available in an **anonymized repository at the review stage**…"

这是直接矛盾。更严重的是，URL 包含作者名 "zhiyongtan"，**如果 CeA 是双盲审稿，这个 URL 暴露了作者身份**。

**修正方案（二选一）**：
- **方案 A（CeA 单盲，Elsevier 多数期刊是单盲）**：Manuscript 保持 public URL；Cover Letter 改为 "The code and per-seed result files are available in the public repository `…` at submission; the precomputed feature stores and raw image corpora are available from the authors on request."
- **方案 B（CeA 双盲）**：Manuscript DAS 改为匿名链接（anonymous.4open.science / Zenodo 匿名 DOI）；Cover Letter 保持 "anonymized repository at the review stage"；接受后替换为 public URL。

**建议先确认 CeA 的审稿政策，再统一。**

### 2. DAS 中 "raw image corpora available on request" 需要澄清
DAS 说 precomputed feature stores 和 raw image corpora "available from the authors on request"。但 Methods 明确说原始 11 MB TPDD release 已丢失、split 是 re-derived 的。作者能共享的是 re-derived 版本，不是 original release。

**修正建议**："The re-derived TPDD image set (the version on which every number in this paper is computed) is available from the authors on request; the original 11 MB release cited in the Methods is no longer available to us."

### 3. Cover Letter 日期 "October 14, 2026"
若为笔误需修正；建议核对（修回信应使用修回日期）。

## 二、建议修正（非阻塞）

### 4. Limitation (9) 的 "pre-registered" 措辞
1,295/138/138 split 是 re-derived release split，没有证据表明实验前被 pre-registered。想表达的应是"预先固定的、不是根据结果选择的"。
建议改为 "…the 1,295/138/138 split is the **fixed** re-derived release split on which every protocol above was executed…" 或 "…was **fixed before any test-set result was inspected**…"

### 5. AI 使用声明应出现在 Manuscript 中（Elsevier 要求）
Cover letter 已有 AI 声明，但 Manuscript 的 Acknowledgments/Conflicts 中没有。建议增加：
"**AI-assisted tools.** An LLM coding assistant was used for drafting and code assistance during preparation. The authors critically reviewed every claim, verified all reported statistics against the underlying result files, and take full responsibility for the content."

### 6. Keyword "knowledge graph" 建议删除
KG 分析现在只是 Discussion 一段，不是核心 contribution。保留会让读者期待更多 KG 内容。

### 7. Table main caption 的 "Appendix~A" 硬编码
建议改为 `Appendix~\ref{appx:protocol}`。

## 三、投稿前 Checklist（摘要）
- LaTeX 编译：tikz/definecolor、\ref/\cite 无 ??
- figures/fig_*.pdf 全部存在
- CeA 参考文献格式（elsarticle/numcompress）匹配 thebibliography
- 字数限制确认（当前约 25,000 词）
- Highlights（3–5 条 ≤85 字符）✓
- Graphical Abstract（可选）
- Cover letter 与 Manuscript DAS 统一；日期核对
- 科学内容：matched-budget caveat 保留；K=1/2 LAION 15-seed 修正四处一致；DINOv2 95.6% 引用正确；Fig 1 caption "carries no independent evidence"

## 四、最终判断
可以提交，但必须先修正 Cover Letter–Manuscript 的 Data Availability 不一致（第 1 项），并澄清 raw image corpora 可用性（第 2 项）。其余五项建议性修正不阻塞投稿。科学内容已稳定。投稿前最后一步：修正 DAS 不一致 + 确认 CeA 单盲/双盲政策。这是唯一阻塞项。
