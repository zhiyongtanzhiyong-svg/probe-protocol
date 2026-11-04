
---

## M2/M3 实验与正文更新（2026-11-04 执行完成）

### M3 — 15 种子 val-selected C probe（`data/probe_cgrid_15seeds_20261104.json`）
- LAION K=1/2 "翻转" **消失** → 15 种子下非显著平局（40.8%±4.6 vs 40.4%，p=0.84；53.0%±4.8 vs 52.5%，p=0.66）
- DFN K=1/2 仍显著落后（32.5%±5.0 / 44.9%±5.0，p<0.001）
- C-selection 本身有效（DFN +2.0~+15.1pp、LAION +0.7~+8.5pp）但不产生显著 probe 领先
- LAION K=5 名义领先 +3.2pp（p=0.032，未校正）——与扩展 LR ResNet 68.1% 打平
- DINOv2 K=1 的 45.9% 是 15 种子均值（旧单种子 46.4% 是种子 0 成员）→ 原叙述不变
- **结论实质变化：M3 从"单种子疑似翻转"变为"15 种子稳健平局，方向性主张撤回"**

### M2 — ResNet 低 LR 扩展扫描（`data/resnet_kshot_lr_extend_{1,2,5,10}_20261104.json`）
- **K=1/2：union 胜者不变（3e-4/ep300），published 40.4%/52.5% 维持** → 承重对比稳健
- **K=5：胜者变 1e-4/ep300，15 种子 68.07%±4.14%（vs matched 64.93%±5.20%，+3.17pp）**
- **K=10：胜者变 1e-4/ep300，15 种子 76.67%±3.75%（vs 74.64%±2.33%，+2.07pp）**
- **结论实质变化：Limitation (10) 从"未跑"变为"已跑"：K≥5 部分 margin 是 grid coverage；K=1/2 稳健**

### 正文编辑（9 处，一批执行）
1. Abstract（line 37）：C-reversal → "非显著平局（无 probe 领先），DFN 仍落后"
2. Intro（line 70）：同上 + ResNet 侧扩展扫描 K=1/2 稳健
3. TikZ Figure 1 节点（line 95）
4. Methods C-grid 段（line 180）：重写为 15 种子全量结果（含旧单种子不重复的说明）
5. Methods matched-budget 段（line 196）：加"已跑扩展扫描"结果子句
6. Results Finding 1（line 346）：含 K=5 名义领先与扩展 LR 打平的说明
7. RQ2（line 411）
8. Limitation (10)（line 667）：完整改写——已跑、结果、K≥5 grid coverage、K=1/2 稳健、descriptive 定位
9. Conclusions（line 674）：C-reversal → 非显著平局

### 交叉验证（M2/M3 批次）
- ✅ 9 处编辑全部落地；无残留 single-seed C 断言；"reverses" 审计只剩 field（hedged）与 balanced-accuracy 语境
- ✅ 双编译 0 错误、0 未定义引用，30→30 页
- ✅ PRE（9 处反向重建）/POST 双 PDF 词袋 diff：5×"reverses" 移除、旧单种子数字（46.4/55.1/34.8/42.8/47.8/31.2/C=0.1/10/30/300/Caveat）全移除、新数字（40.8/53.0/68.2/75.0/68.1/76.7/p 值）全进入，其余为换行伪影
- ✅ `check_tex_vs_data.py` 34/34 pass
- ✅ M2/M3 所有正文数字与 JSON 源值核对一致（68.07→68.1、76.67→76.7、+3.17→+3.2、+2.07→+2.1、p 值）

---

## 提交包补全（2026-11-04 第二批）

### Minor 5（标题）— 已执行
- 选候选 B：`Fine-Tuning or Frozen Probes? Empirical Applicability Boundaries for Off-the-Shelf CLIP in Tobacco Leaf-Disorder Diagnosis`（16 词，应用腔，问题式）
- 级联：probe-protocol.tex `\title` + cover_letter_CeA.tex 标题行，均已同步；双文档编译 0 错误，30 页不变
- **级联审计发现并修复**：`Highlights.txt` 第 2 条仍是已撤回的单种子断言 "At K=1 the CLIP-vs-ResNet order is checkpoint-specific: LAION-2B leads, DFN trails" → 改为 "With C selected on validation (15 seeds), LAION-2B K=1/2 is a tie; DFN-CLIP trails."（81 字符，符合 Elsevier ≤85 限制）；同时把缺失的 matched fine-tuning 强正结果补进 Highlights

### Cover letter scope 段 — 已执行
- "Fit and originality" 段追加 2 句：贡献定位为 protocol-controlled 部署边界（非新方法）；现有 CLIP 植物病文献数字互不可比（tuning budget/split/class composition 不一致），实践者缺决策证据——先发制人挡"无新方法"审稿意见
- 已核对 cover letter 无 C-reversal 残留断言

### 遗留（作者人工，需事实输入）
- **Minor 2（field provenance）**：见下方填空模板
- **Minor 3（SOTA 对比）**：需 2–3 个文献报告值（作者核读原文后填），可用锚点数字见下方

### 待作者事实（仓库无信息，不代笔）

#### Minor 2 填空模板（field 246 图集 provenance）
现状：Methods（tex:150）已有 "collected from commercial tobacco fields under natural lighting with variable background clutter"；缺采集者/地点/时段/设备/标注/授权。建议补入 Methods + DAS 的完整声明（作者填 [ ] 后我执行）：

> The 246 field images were collected [by the authors / by X] from commercial tobacco fields in [地点/省/市] during [生长季/年份] using [设备, e.g., smartphone models], with labels assigned in Chinese by [标注者, e.g., two agronomists with X years of experience, adjudicated by Y]. The dataset is [not publicly released / released at URL]; images were [captured with / without] landowner consent; no third-party license applies.

五问清单：①自采还是合作单位采集？②哪个省/市、什么生长季？③什么设备拍的？④谁标的类（中文标签已确认）、是否有多人 adjudication？⑤数据集是否公开（URL/仓库）、授权状态？

#### Minor 3 填空模板（SOTA 定位对比，2–3 个）
我们可用的锚点数字（已核实）：TPDD full-data held-out：frozen probe 86.2%/87.7%（val-selected C），fine-tuned ResNet-18 94.7%（15-seed），fine-tuned CLIP 94.7–95.9%；zero-shot 13.0% 基线下 best template 不显著。作者核读原文后提供（格式：`[文献 key] 数据集/规模/报告值/是否 zero-shot/few-shot`），我负责把对比句写成 protocol-contrast 形式（强调 split/composition/tuning-budget 不可比），并同步 Methods 或 Discussion 定位段 + 参考文献条目。候选文献（已在 bib 中）：PD-CLIP（pdclip2026）、SCOLD（scold2025）、TPDD 原文（Lin 2022，若 bib 中已有则直接引其报告值）。
