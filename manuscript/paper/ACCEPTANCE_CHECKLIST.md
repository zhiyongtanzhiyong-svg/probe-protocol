# 投稿前验收清单（20 条）

用法：投稿前逐条勾选。1–16 为机器可验证（附命令，全部从仓库根目录运行）；
17–20 为人工项。任何一条不过 → 修复后整体重跑 1–16。

## A. 构建与版面（机器）

- [ ] 1. `latexmk -pdf -interaction=nonstopmode -g probe-protocol.tex`（在 `manuscript/paper/`）exit 0，且 log 中 `!`=0、Overfull=0、undefined=0
- [ ] 2. `pdfinfo probe-protocol.pdf` → 29 页（若内容改动导致页数变化，先向自己说明原因再接受新页数）
- [ ] 3. 浮动体顺序人工扫一眼 PDF：Fig 1 (p3) → T1 (p9) → T2 (p11, 含 DINOv2-FT 行) → Fig 2 (p12) → T3 (p13) → T4 (p14) → Fig 3 (p15) → Fig 4 (p19) → Cost-weighted (p26) → Fig 5+Appendix C (p27)；无孤行孤标题
- [ ] 4. `python3 scripts/generate_figures.py` 重跑后 `git status` 无 figures 变化（图与数据零漂移）

## B. 数值一致性（机器）

- [ ] 5. `python3 scripts/check_tex_vs_data.py` → **N/N All checks passed**（34 项锚点：ZS/few-shot/FT/multiplicity/ECE/LP++/cross-crop/DINOv2-FT）
- [ ] 6. Table 2 五行 × 五列与 `data/*.json` 逐格一致（含 DINOv2-FT: 40.2 / 57.5 / 75.3 / 85.1 / 95.6±1.5）——被 #5 覆盖，人工再抽 3 格
- [ ] 7. `grep -c "45.5" manuscript/paper/probe-protocol.tex` = 0（旧 cross-check split 的番茄 K=1 值不得出现在正文）
- [ ] 8. `grep -c "p2_results" manuscript/paper/probe-protocol.tex` = 0（已禁止引用的文件）
- [ ] 9. `grep -c "REPLACE\|XXXX\|TBD" manuscript/paper/probe-protocol.tex` = 0（无遗留占位符）

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
