# Submission Checklist — Computers and Electronics in Agriculture

> **[TOP-PRIORITY — HUMAN, before submission] Confirm the live CeA "Guide for Authors" in a normal
> browser.** Every automated fetch from this machine returned 403 (webfetch / curl / Jina / CDP), so
> the exact house format **cannot be machine-verified here**. Manually confirm: (a) reference/citation
> style — this paper uses **numbered** `[1]…[32]`; (b) reference **entry format** (e.g. `Authors.;
> Title. Venue, Year.; DOI`); (c) whether the portal wants the **`elsarticle`** class template or
> accepts plain `article`. URL:
> `https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture/publish-your-article/guide-for-authors`

> **Source:** Drafted 2026-10-13. ScienceDirect/Elsevier guidelines page returned 403 to all
> automated fetch attempts (webfetch, curl, Jina, CDP). Items marked **[VERIFY]** must be
> confirmed against the live "Guide for Authors" page before submission:
> `https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture/publish-your-article/guide-for-authors`

## STATUS (2026-10-14)
**Re-target to C&E in Agriculture (Elsevier) COMPLETE:**
- [x] Document class MDPI → `\documentclass{article}` (plain LaTeX); all MDPI macros removed
- [x] Citations author-year → numbered `[1]…[32]` (verified in compiled PDF)
- [x] End-matter MDPI macros → standard `\section*{...}` (Author Contributions / Funding / Data Availability Statement / Conflicts of Interest)
- [x] `\simplesumm` removed; Highlights created (`Highlights.txt`, 5 bullets, 50–200 chars each)
- [x] Compiles clean (pdflatex ×2, 0 errors, 0 undefined refs); checker 189/189; 0 TABs
- [x] K=1 LAION p-value corrected to 1.5×10⁻⁵ throughout (abstract + intro + body)
- [x] LP++ DFN draw `30.5/48.0/56.4` + full gain `+6.5` pp; Appendix-A "Feature-store unification" (LAION 75.4% vs canonical 76.1%, no change of sign)
- [x] **2026-10-14 A+B expansion (reframed):** Lever B = honest readout-geometry characterization (checkpoint-dependent CLS-vs-patch-token signal; DFN→CLS, LAION→patches; ties Shi et al. 2026; "not a new method") — contributions (5), main body, untested-heads line, Limitation (11)
- [x] Cross-crop now **tomato + corn** (both powered); seed-paired powered stats added (tomato K1 +4.3pp p=0.051; corn K1 +8.1pp p=0.021, K10 p<0.001); Highlights bullet 5 updated to two cross-crops
- [x] **Full 55k PlantVillage skipped** (user decision) — tomato+corn already provide powered multi-crop evidence
- [x] M5 gate green: 189/189 checker PASS; clean compile 27pp, 0 errors, 0 undefined refs/cites, 0 overfull>100pt, 0 TABs
 - [x] **2026-10-16 adversarial review pass + apt fix.** Cross-examination flagged P1/P2/P3; each checked against source: **P2 (K=1) already addressed** (abstract L25 "checkpoint-specific… probe-protocol effect… not a representation effect"); **P1 restricted-usage-mode already in abstract** (L30) and title L12 already tobacco-scoped/representation-focused → no retitle needed. **P3 is a research gap** (fine-tuned CLIP not run at K≥5), already in Limitation L553 — not a writing fix. **Sole apt residual fixed:** abstract L26 "…the ordering is a property of the data regime rather than of the model" conditioned on the frozen-probe-vs-fine-tuned-CNN protocol (→ "…rather than of the encoder"), aligning the abstract with the body's own L482 "restricted usage mode" scoping. Re-verified: 189/189 gate + clean compile 27pp (0 errors / 0 undef / 0 overfull>100pt).
   - [x] **2026-10-29 reference-integrity audit (3 questions).** (Q1) Citation **order was wrong** — all 32 `\bibitem`s reordered to **first-citation order** (the 2026 readout-geometry cluster `cxr/traceclip/dinov3mil` had sat mid-list but is cited last). (Q2) All **32/32 references verified REAL** via CrossRef/OpenAlex/arXiv/DataCite (30 papers + 2 datasets; no hallucinated entries); `scold2025` year corrected **2026→2025** (per its DOI + OpenAlex). (Q3) All 32 citation contexts **appropriate** (no wrong-context). Reference count grew **25→32** through revision rounds. Recompiled clean: 26 pp, 0 errors, 0 undefined; `.aux` numbering `1→32` = first-citation order. Backup: `probe-protocol.tex.bak-citeorder-20261029-222648`.
   - [x] **2026-10-30 figure/table audit + caption de-duplication.** Placement verified clean: all 14 floats `[!ht]`, recompile 26 pp / 0 errors / 0 undefined / **0 float warnings**; every float referenced before-or-adjacent to its definition (the two full-width figures sit beside their callout; `fig:tsne` correctly in the Appendix). Trimmed 4 captions that re-narrated the body (number-preserving, caption-only): `tab:prompts` (dropped the finding sentence already in the following RQ1 paragraph, L206–208); `tab:perclass` — dropped the §why-duplicated interpretation tail, **kept** the 16-class index→name list (it is the only copy; the Appendix lists only the 19 field classes); `fig:curve` — dropped the K=1-LAION tie/unstarred clause already in Methods (L159/L162) + Findings (L70/L287), kept the asterisk legend + the "K=1 ResNet = legacy baseline" reading note; `tab:readout` — dropped the L486-duplicated readout interpretation, kept all column definitions. Cross-verification caught and held the tab:perclass class-list removal. Backup: `probe-protocol.tex.bak-captrim-20261030-064059`.
    - [x] **2026-10-30 de-verbosity / redundancy pass (2 batches).** (Batch 1) Abstract re-compressed **625→249 words** and collapsed 7 paragraphs→1 (all 7 finding verdicts + load-bearing numbers 13.0% / 86.2% / 92.5-vs-89.4 retained; per-$K$ number soup and caveat parentheticals dropped — all still in the body); tomato+corn captions' identical verbose split-exclusion sentence → short pointer (full caveat retained in Intro L53 / Methods L154 + result bodies). (Batch 2) Pointer-ized the 4 recurring caveats — #2 full-data gap/CI, #4 CV-folds-not-independent, #6 PlantVillage non-independent, #7 field "untested hypothesis": each stated once in Results, now pointed to (Section~\ref{sec:results}) from Limitation 3, Conclusions, and 2 within-Results near-verbatim dups; every verdict kept, only re-derived numbers dropped, so the canonical CI ($[-0.1,\,+3.1]$) now appears only in Results (L281/L334). Verify: 26 pp, 0 errors / 0 undefined refs / 0 overfull / 0 float; new pointers resolve. Backups: `.bak-deverb-115959`, `.bak-deverb2-121124`, `.bak-deverb3-124257`.
    - [x] **2026-10-30 selective em-dash pass.** Converted the **21 run-on clause-join** em-dashes to cleaner punctuation (commas / semicolons / colons / parentheses) across Intro, Methods, Results, Discussion, Conclusions; **kept** the ~32 tight appositive em-dashes doing real parenthetical work and left the 41 table cell placeholders (`& --- &`) untouched. Prose em-dash density **53→32**; total `---` **94→73**. Every change punctuation-only (no wording / number / claim altered); each target matched exactly once. Verify: 26 pp, 0 errors / 0 undefined / 0 overfull / 0 float. Backup: `.bak-emdash-125826`.
   - [ ] **Still TODO before submission (human-only, 2026-11-02):** (1) confirm CeA live Guide-for-Authors caps (page/word/keyword limits, blinding rules, Highlights spec) against this 29-pp build; (2) real repository URL (insert at submission; anonymized in the PDF per double-blind); (3) author ORCID(s) if the portal requires them; (4) final human read-through of Abstract/Conclusion/Table-2 footnote/Appendix A. Highlights are DONE (Highlights.txt: 5 bullets, 78–85 chars, verified 2026-11-02).


## FINAL GATE — 20-item quick pass (2026-11-02, paper final: 29 pp)

### 最终运行记录（2026-11-05 晚，数据可用性 → Zenodo 持久 DOI）
**变更：公共可引用版本从 GitHub URL 改指 Zenodo 持久 DOI（`10.5281/zenodo.24557268`；记录标题 `zhiyongtanzhiyong-svg/probe-protocol: v1.0.0-CeA_submission_snapshot`）。**
- DAS（`probe-protocol.tex` L684）+ cover letter（L41）：GitHub URL → `https://doi.org/10.5281/zenodo.24557268`；“public repository” → “public Zenodo release (persistent identifier)”；删除 “exceed comfortable repository size”（Zenodo 能放 2GB，该理由不再成立 → 改中性 “not included in the release”）。
- **连锁排版**：DAS 变长触发末行 64→1.25pt 溢出 → 将 DAS 内长文件名改为可断行（`data/tpdd/{train,test,validation}/labels.json` 用 `\allowbreak` 分段；`feature_store_audit.json`/`cache_freshness.json` 用 `\path{...}` 原生下划线）。
- 复验：稿件 0 错误 / **0 overfull** / 0 undefined / 31 页；封面信 0 overfull / 1 页；`check_tex_vs_data.py` 34/34 PASS；双文件 `grep github.com` = 0；包内 source 独立编译 0 overfull / 31 页。投稿包 + zip 已重建（1.9M，`unzip -t` 无错）。
- **盲审已定（单盲，作者须知确认）**：标题页作者块保留 ✓；Zenodo 记录标题含作者用户名（`zhiyongtanzhiyong-svg/...`）**无碍**——单盲下作者本就对评审可见，无需匿名化（round19 “方案 B 双盲匿名”整条作废）。

### 最终运行记录（2026-11-05，投稿包终检 + 排版修复）
**状态：交叉验证（实测非仅信 log）+ 修复完成，投稿包自洽可独立编译（31 页版）。**
交叉验证确认 3 个真实问题并修复（**纯排版，未动任何数字/结论**；`tests/check_tex_vs_data.py` 仍 34/34 PASS）：
1. **p24 Limitation(5) 右溢 62pt**：`\texttt{data/probe_cgrid_15seeds_20261104.json}` 为不可断行单 token，`\emergencystretch=2em` 也救不了（实测正文 xMax=585.96 > 右边界 524）→ 改 `\path{data/probe_cgrid_15seeds_20261104.json}`。**坑：`url` 宏包 `\path` 内必须用原生 `_`，写 `\_` 会渲染出字面反斜杠**（首版误用 `\_` 已纠正）。
2. **p30 底溢 58pt + p29 底部 129pt 空隙 + 尾部挤到 p31**：Appendix B/C 两张高图（278pt+143pt）`[!ht]` 浮动错位，把 Appendix D 正文顶上图页 → Appendix D 前加 `\clearpage`（图落 p30、D 独立 p31）。修复后全文 0 overfull、无溢出、p29 空隙缩至 56pt。
3. **投稿包 source 无法独立编译**：包内图为 `Figure_N_*.pdf`，而 source 引用 `figures/fig_*.pdf` → 包内 5 图改名 `fig_*.pdf`（sha256 与主 `figures/` 一致，内容零变化）。现 source 已在干净目录独立编译通过（0 错误 / 0 overfull / 0 undefined / 31 页）。
- A.1 ✅ 干净重编译 0 错误 / **0 overfull**（本轮实测确认，非 11-04 记录时的 2 处）/ 0 undefined
- A.2 ✅ **31 页**（11-04 记录为 30；31 = Appendix A 文本 + B/C 两图 + D 文本 的固有体量，`\clearpage` 使分页干净无溢出；非内容漂移）
- 投稿包已同步：`01_manuscript.pdf` / `01_manuscript_source.tex` 取最新干净构建；`submission_package.zip` 重建（9 文件 / 1.9M / `unzip -t` 无错）
- **盲审：单盲已确认（作者须知）**，不再作为开放项
- 仍未关闭（**人工**）：作者 ORCID（5 人）、最终通读 PDF（Abstract/Conclusion/Appendix）

### 最终运行记录（2026-11-04，投稿前）
**状态：机器可验证项全部通过（30 页版）**。逐项：
- A.1 ✅ compile 0 错误 / 0 overfull / 0 undefined（修复 Appendix D 长文件名 overfull 79.9pt → 花括号集合记法）
- A.2 ✅ 30 页（29→30：M2/M3/M4 内容增量，已记录）
- A.3 ⏳ 浮动体顺序人工扫 PDF（用户）
- A.4 ✅ M2/M3 不动任何图；figures/ 无变化
- B.5 ✅ 34/34；B.6 ✅（被 B.5 覆盖）
- B.7 ✅ 复核：45.5 出现在 Appendix D（Round 18 M4 新增，gate 写于其前）；6 值全部对 JSON `stage2_final` 核实（45.5±5.4 / 75.3±4.4 / 82.3±2.4 / 76.2±4.8 / 90.6±2.4 / 92.5±1.1）→ 旧 gate 误报
- B.8 ✅ 0；B.9 ✅ 0
- C.10 ✅ sha256 匹配；C.12 ✅ 5 JSON 字段齐全；C.13 ✅ 全部 data 引用存在（花括号集合逐一展开验证）
- C.11 ✅ 提交后树干净
- D.14 ✅ Limitation (1)/(10) 自洽（(10) 已改写：K=1/2 稳健、K≥5 grid coverage、frozen arm 不受影响）
- D.15 ✅ descriptive ×21；nominal-post-selection 限定句在 tex:260（含 and，grep 变体未命中）
- D.16 ✅ cost-weighted/calibration scope 限定在位
- E.17 ✅ 双盲干净（无真实 URL / 0 ORCID）
- E.18–20 ⏳ 人工：CeA Guide for Authors 官网核对、投稿时插匿名 repo URL、最终通读 PDF

---

Run once before submission; items 1–16 are machine-verifiable (commands from repo root), 17–20 human.
Any item fails → fix, then re-run all of 1–16.

## A. 构建与版面（机器）

- [ ] 1. `latexmk -pdf -interaction=nonstopmode -g probe-protocol.tex`（在 `paper/`）exit 0，且 log 中 `!`=0、Overfull=0、undefined=0
- [ ] 2. `pdfinfo probe-protocol.pdf` → 29 页（若内容改动导致页数变化，先向自己说明原因再接受新页数）
- [ ] 3. 浮动体顺序人工扫一眼 PDF：Fig 1 (p3) → T1 (p9) → T2 (p11, 含 DINOv2-FT 行) → Fig 2 (p12) → T3 (p13) → T4 (p14) → Fig 3 (p15) → Fig 4 (p19) → Cost-weighted (p26) → Fig 5+Appendix C (p27)；无孤行孤标题
- [ ] 4. `python3 scripts/generate_figures.py` 重跑后 `git status` 无 figures 变化（图与数据零漂移）

## B. 数值一致性（机器）

- [ ] 5. `python3 tests/check_tex_vs_data.py` → **N/N All checks passed**（34 项锚点：ZS/few-shot/FT/multiplicity/ECE/LP++/cross-crop/DINOv2-FT）
- [ ] 6. Table 2 五行 × 五列与 `data/*.json` 逐格一致（含 DINOv2-FT: 40.2 / 57.5 / 75.3 / 85.1 / 95.6±1.5）——被 #5 覆盖，人工再抽 3 格
- [ ] 7. `grep -c "45.5" paper/probe-protocol.tex` = 0（旧 cross-check split 的番茄 K=1 值不得出现在正文）
- [ ] 8. `grep -c "p2_results" paper/probe-protocol.tex` = 0（已禁止引用的文件）
- [ ] 9. `grep -c "REPLACE\|XXXX\|TBD" paper/probe-protocol.tex` = 0（无遗留占位符）

## C. Provenance 与可复现（机器）

- [ ] 10. `sha256sum checkpoints/dinov2/dinov2_vitl14_pretrain.pth` = `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`
- [ ] 11. `git status --short` = 空（工作树干净）；`git log --oneline | wc -l` 记录在案
- [ ] 12. 5 份 DINOv2 JSON 的 `checkpoint`/`gpu`/`torch` 字段齐全（`python3 -c "import json,glob; [print(f, json.load(open(f)).get('gpu')) for f in sorted(glob.glob('data/dinov2_finetune*.json'))+glob.glob('data/dinov2_kshot*.json')]"`）
- [ ] 13. Appendix A provenance 表中每个 data 文件名在 `data/` 实际存在（`grep -o "data/[a-z0-9_]*\.json" probe-protocol.tex | sort -u` 逐一 `test -f`）

## D. 披露完整性与自洽（人工为主，grep 辅助）

- [ ] 14. Limitations 10 条编号连续、无自相矛盾（重点：(1) 说"三个 backbone 家族"且不再说"DINOv2-FT 需进一步研究"；(10) grid 不对称只影响 FT 臂）
- [ ] 15. `grep -c "descriptive" probe-protocol.tex` ≥ 3（p=0.002 的 5-fold 描述性定性在 Results/讨论/限制三处一致）；"nominal, post-selection, non-inferential" 出现 1 次
- [ ] 16. Cost-weighted 与 Calibration 小节仍带 scope 限定（"descriptive"/"not a primary claim" 措辞未在被后续编辑中冲掉）

## E. 投稿合规（人工）

- [ ] 17. **双盲**：`grep -in "github.com\|our code is\|available at" probe-protocol.tex` 无真实 repo URL（投稿版匿名；附录措辞用 "anonymized repository"）；无致谢、无 ORCID（若 CeA 双盲要求如此——以官网 Guide for Authors 为准）
- [ ] 18. CeA Guide for Authors 官网逐条核对：页数/格式/参考格式(Citeseer? IEEE?)/双盲细则/Highlights 要求 —— **需人工访问**
- [ ] 19. 5 条 Highlights（每条 ≤85 字符含空格）已按期刊模板写好；若需真实 repo URL / ORCID，此刻插入
- [ ] 20. 最终通读一遍 PDF（重点：Abstract、Conclusion、Table 2 脚注、Appendix A）；确认摘要与正文数字一致（95.6/87.7/40.2/10.5pp 等）

## Abstract & Keywords

- [x] Abstract unstructured (no sub-headings) — **done** (single paragraph; 249 words, re-compressed 2026-10-30)
- [x] Abstract ≤ 250 words — **done** (249 words, re-compressed 2026-10-30; it had drifted to ~625 words through revision rounds and was re-cut to 249 with all finding verdicts retained) **[VERIFY: confirm exact CeA cap]**
- [x] Highlights ≤ 85 characters each — **done** (all five were 85–95 characters; re-cut 2026-10-24, now 76–83) **[VERIFY against the live Guide for Authors]**
- [x] Keywords: 7 (CLIP; zero-shot classification; agricultural disease diagnosis; vision-language models; few-shot learning; knowledge graph; prompt engineering) **[VERIFY: confirm C&E-in-Ag keyword range — some Elsevier journals cap at 3–6; may need to trim to 6]**
- [x] No `\simplesumm` (MDPI-specific) — **removed**

## Citation & References

- [x] **Citation style:** Now **numbered** `[1], [2], …` — **done** (article class + `thebibliography` renders `\cite` as `[n]`; verified `[1]…[32]` in compiled PDF)
- [x] **32 references — no orphans** — **pass** (32 distinct `\cite` keys = 32 `\bibitem` entries; parity clean; count grew 25→32 through revision rounds)
- [x] **Numbering follows first-citation order (strict)** — **fixed 2026-10-29** (`[1]` CLIP … `[8]` Shi et al. … `[30][31][32]` = 2026 readout-geometry cluster `cxr`/`dinov3mil`/`traceclip`; `\bibitem` blocks ordered by first `\cite` appearance). The 2026 cluster had previously sat mid-list; reordering applied and verified `1→32` in the compiled `.aux`. Managed by `scripts/audit_references.py --reorder` (committed, idempotent).
- [x] **Reproducible reference audit — pass via `scripts/audit_references.py`** (each `\bibitem` resolved against Crossref/DataCite/arXiv/HF; title + first-author + year checked). `tld` (community HF dataset, no DOI) verified **offline against local ground truth** (`data/tld_local_manifest.json`: 1,392 images / 10 classes / 0 missing on disk). Audit record: `data/reference_audit.json`. **Re-verified 2026-10-29:** all **32/32** entries confirmed REAL (30 journal/papers via CrossRef/OpenAlex/arXiv + 2 datasets: `plantvillage_mendeley` DataCite-confirmed, `tld` HF no-DOI); `scold2025` year corrected 2026→2025.
- [x] DOI present for most references — **pass** (**6 without DOI**: `radford2021learning`, `tld`, `fang2023data`, `dinov2`, `hu2022lora`, `liu2022convnext` — all arXiv/preprint or community dataset with no publisher DOI available)

## Section Structure

- [x] IMRAD: Introduction → Related Work → Materials and Methods → Results → Discussion → Conclusions — **present**
- [x] Results organised by RQ1/RQ2/RQ3 — **present**
- [x] Appendix A (reproducibility protocol) — **present**
- [x] Abbreviations section — **present** **[VERIFY: confirm C&E in Ag requires abbreviations section]**

## End-Matter (required by journal-guidelines skill)

- [x] CRediT authorship contribution statement — **added** (5 authors: Chenfeng Long, Zhiyong Tan, Yangjun Deng, Xiang Luo, Lizhuo Zhang; roles L.Z./C.L./Z.T./Y.D./X.L. — **VERIFY Z.T./Y.D./X.L. role assignment with co-authors**)
- [x] Declaration of competing interest — **added**
- [x] Acknowledgments (with AI-use declaration) — **added**
- [x] Data availability statement — **added** (repository URL placeholder)
- [ ] ORCID for authors — **MISSING** **[VERIFY: confirm required; add ORCID iDs for all five authors]**
- [ ] Repository URL placeholder — **MUST replace before submission**

## Figures & Tables

- [x] Figures in PDF vector format (`\includegraphics{figures/fig_*.pdf}`) — **pass**
- [x] Tables use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) — **pass**
- [x] Captions above figures, below tables — **pass**
- [ ] **[VERIFY: confirm minimum DPI (typically 300 for halftones, 1000 for line art)]**
- [ ] **[VERIFY: confirm figure placement rules (embedded in text vs. separate files)]**
- [ ] **[VERIFY: confirm color requirements (color OK? grayscale required for B&W section?)]**

## Document Class & Layout

- [x] **Document class:** Switched MDPI → `\documentclass{article}` (plain LaTeX) — **done**
  - All MDPI-specific macros removed: `\firstpage`, `\pubvolume`, `\issuenum`, `\articlenumber`, `\pubyear`, `\copyrightyear`, `\datereceived`, `\daterevised`, `\dateaccepted`, `\datepublished`, `\AuthorNames`, `\address`, `\corres`, `\reftitle`, `\Title`, `\Author`, `\simplesumm`, `\authorcontributions`, `\funding`, `\dataavailability`, `\conflictsofinterest`
  - Note: `elsarticle.cls` not installed locally, so plain `article` is used; re-wrap in the official `elsarticle` class at final typesetting if the portal requires it
- [ ] **[VERIFY: confirm page limit (typically no hard limit for C&E in Ag, but ~15–25 pages is typical)]**
- [ ] **[VERIFY: confirm line spacing (single vs. 1.5 vs. double)]**
- [ ] **[VERIFY: confirm font (Times New Roman vs. Computer Modern)]**
- [ ] **[VERIFY: confirm single-column vs. double-column]**

## Blinding

- [ ] **[VERIFY: confirm single-blind (author names visible to reviewers) vs. double-blind]**
  - If single-blind: current author name/affiliation is fine
  - If double-blind: must remove author name, affiliation, ORCID, acknowledgments from manuscript

## Cover Letter

- [ ] Draft cover letter with:
  - Significance statement (why this paper matters for C&E in Ag readership)
  - Confirmation that the paper is not under consideration elsewhere
  - Suggested referees (3–5, with ORCIDs and affiliations)
  - Statement of AI use in manuscript preparation
  - **[VERIFY: confirm C&E in Ag requires suggested referees]**

## Submission Portal

- [ ] **[VERIFY: confirm submission portal URL (likely Editorial Manager at `submit.elsevier.com` or `editorialmanager.com`)]**
- [ ] **[VERIFY: confirm accepted file formats (.tex, .docx, .pdf)]**
- [ ] **[VERIFY: confirm APC (C&E in Ag is hybrid — confirm whether OA is optional or required)]**
- [ ] **[VERIFY: confirm word/page limit for full papers]**
- [ ] **[VERIFY: confirm supplementary material policy (size limit, format)]**
- [ ] **[VERIFY: confirm replication package / code deposit requirements]**

## Pre-Submission Actions

- [ ] Replace `[ANONYMOUS_REPOSITORY_URL_TO_BE_INSERTED_BEFORE_SUBMISSION]` with actual URL
- [ ] Add author ORCID
- [x] Switch document class to Elsevier template — **done** (article class)
- [x] Convert citation style to numbered (if required) — **done** ([1]…[32])
- [x] Remove all MDPI-specific macros — **done**
- [x] Re-compile and verify 0 errors, 0 undefined references — **done** (pdflatex ×2, 0 errors)
- [x] Run `tests/check_tex_vs_data.py` — confirm 189/189 — **done** (189/189)
- [x] Create Highlights (`Highlights.txt`) — **done** (5 bullets)
- [x] Draft cover letter — **done** (`cover_letter_CeA.md`)
- [ ] Proofread for grammar and style (use `latex-paper-en` grammar module)
- [ ] Verify all figure files exist and are referenced correctly
