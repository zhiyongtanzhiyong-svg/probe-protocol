# Round 18: 外审模拟（SCI 审稿人通读）+ 评估 + 处置 (2026-11-04)

**角色**：以 Computers and Electronics in Agriculture（Elsevier）审稿人身份通读全文并逐条核数。
**判定（外审）**：Major Revision（倾向接收）。
**判定（自我评估，见 §2）**：中肯；5 条 Major 中 4 实 1 半投射；6 条 Minor 全部低成本可满足。

---

## §1 外审意见原文

### 优点
1. **统计严谨性远超本领域平均**——种子配对、Holm–Bonferroni/FDR、a-priori power、检测限、pre-specified vs exploratory 显式标注；外审用仓库 `data/*.json` 复核关键数字（87.7±2.8、10 above+1 equal、p=0.060、CI[-0.1,+3.1]）全部吻合（可验证性卖点首次独立确认）。
2. **claim scoping 诚实**，几乎每个结论都标了 power/provenance 边界，无过度外推。
3. 控制实验覆盖广（DINOv2、SigLIP、LoRA、ConvNeXt、BioCLIP-2、CoOp、LP++、ResNet-50）。

### Major
- **M1 主张重新定位（半实）**：标题级主张"排序由适应协议而非数据量决定"近乎同义反复（ML 社区先验）；真实贡献是 (a) zero-shot CLIP 不高于多数类基线的负结果，(b) 烟草任务的经验边界。建议标题/摘要/讨论向"适用边界 + 部署准则"倾斜。
- **M2 网格边界（实，最高优先）**：唯一强正结果"matched fine-tuning CLIP 全面领先"压在网格边界上——Limitation (10) 自认 ResNet 最优配置位于最低 LR 且验证精度仍在上升。"自认的 limitation 足以吞掉自己的正结果"。处置：跑 ResNet 低 LR 扩展扫描（探索性标注），存活则关闭 Limitation (10)，收窄则量化 grid coverage 占比。
- **M3 单种子 C 反转（实）**：K=1/2 LAION 反转是中心主张的一半，但基于 seed-0 单 support set。"exploratory"标注不能替代实验。处置：K∈{1,2,5,10} × 15 seeds × C-grid 重选。风险：LAION K=5 对比（4.7pp）可能收窄。
- **M4 跨作物 like-for-like（半实）**：tomato/corn 只复制冻结臂且对照单配置（欠调优）ResNet；"Two findings hold across all three crops"在同一基线下不统一成立。处置：把已跑的 matched cross-crop run 以描述性方式报告进附录（数据在 data/*.json）。
- **M5 田间 reversal 表述过强（实，本可先修）**：正文已限定（53 图、SE≈6pp vs gap 6.9pp、单配置、8 seeds），但摘要"the fine-tuned ranking reverses under field conditions"与结论"field conditions reverse the fine-tuned ranking"是断言式。SE≈gap 不足以称 reversal。处置：摘要/结论降级为 may reverse / directional。

### Minor
1. **Power 括界混源**：Evaluation 中"(89.4% vs. 87.7%)"——89.4 是 5-fold CV 探针均值、87.7 是 held-out ResNet 均值，不同协议；同一括界又用 held-out per-class 表收紧下界。结论不受影响，标注来源修。
2. **田间 246 图集来源未声明**：Methods 只说"collected from commercial tobacco fields"，未说明自采还是公开集。需作者补声明（自采者/地点/协议/授权）。
3. **缺烟草 SOTA 定位**：TPDD 原文及文献多有 95–99% 报告值，全文未定位。加 1–2 句。
4. **统计装置搬家（部分驳回）**：预注册声明与检测限是承重墙，不动；power bracket 细节可压缩。
5. **标题过长术语化**（20 词）：向任务/决策倾斜，可选。
6. **`n_seeds_ge_clip` 命名**：实为 strictly-greater 计数（bestC=10，含等应为 11）。正文表述正确，字段名误导，改 `n_seeds_above_clip`。

### 外审给作者的问题
1. M2：ResNet 学习率最优若低于 3e-4，matched margin 会缩到多少？
2. M3：为何探针 C 选择未做 15-seed 复跑？
3. 田间集是否作者自采？
4. 本文 best 数字相对 TPDD 原文报告值处于何位？

### 外审结论
统计上几乎无可挑剔、但主张新颖性偏薄。可复核性极佳。接收关键在 M2/M3：补实验或明确降级。**M1 与 M5 是低成本的必改项。**

---

## §2 自我评估（逐条判定）

| # | 判定 | 处置 |
|---|---|---|
| M2 | 完全中肯，最高优先 | 跑实验——ResNet 低 LR 扩展扫描（小时级）。两种结局都是赢 |
| M3 | 完全中肯 | 跑实验——C-sweep 15 seeds。回答 Q2：设计判断+资源原因，现在补 |
| M5 | 完全中肯（round 14 就该顺手修） | 摘要/结论降级，5 分钟 |
| M1 | 部分中肯（ML 先验投射到 C&EA 读者；但"边界+部署准则"框架低成本高收益） | 摘要/讨论任务导向倾斜；标题可选 |
| M4 | 部分中肯（论文已声明基线差异与 K=1 方向依赖） | 附录描述性报告 matched cross-crop run，透明度净增 |
| Minor 1 | 正确 | 括界改 held-out 同源对，结论不变 |
| Minor 2 | 正确 | 需作者补声明（配合 DAS） |
| Minor 3 | 正确 | 加定位句（不引用未核实文献数字） |
| Minor 4 | 部分驳回（承重墙不动） | 不做结构搬家 |
| Minor 5 | 可选 | 候选标题存档，本轮不改标题（连带 cover letter/Highlights 级联） |
| Minor 6 | 正确 | 字段改名（JSON + 2 脚本） |

**记录**：M5 是本轮唯一"我们本可以先修掉"的项——摘要-正文强度不匹配在 round 14 修零样本范围矛盾时就该顺手处理田间句。最后一个强度不匹配由外审发现，而非自我修正。

---

## §3 交叉验证记录（本轮执行前核验）

| 项 | 验证 | 证据 |
|---|---|---|
| M5 摘要断言式 | ✓ 属实 | tex:37 "and the fine-tuned ranking reverses under field conditions"（无 hedge） |
| M5 结论断言式 | ✓ 属实 | tex:673 "field conditions reverse the fine-tuned ranking" |
| M5 正文已限定 | ✓ 属实 | tex:528 紧随 "The reversal is directionally consistent ... single-checkpoint and directional" |
| Minor 1 混源 | ✓ 属实 | tex:205 "(89.4% vs. 87.7%)"：89.4 = CV 探针均值（cv_protocol），87.7 = held-out ResNet 均值；bracket [1.7%,23.3%]、span 5.8–11.5 |
| Minor 1 修正算术 | ✓ 已验 | held-out 同源对 (86.2, 87.7)：|Δ|=1.5pp；(1-a1)+(1-a2)=13.77+12.27=26.0%；p_d∈[6%,26.0%]；80% 检测限 = 2.8·sqrt(p_d/138)：6%→5.8pp，26.0%→12.2pp；planning 0.16→3.4pp/9.5pp 不变；1.5pp 与 3.5pp 均 < 5.8pp，结论不变 |
| M4 数据存在 | ✓ 属实 | resnet_k{1,5,10}_matched_{tomato,corn}.json 存在（15 seeds, stage2_final）：tomato 45.5±5.4 / 75.3±4.4 / 82.3±2.4；corn 76.2±4.8 / 90.6±2.4 / 92.5±1.1 |
| M4 论文已声明排除 | ✓ 属实 | tex:471（tomato）、tex:494（corn）已有 different-split 排除声明 → 补描述性附录 + 指针 |
| Minor 6 计数语义 | ✓ 属实 | verify_claims.py:215 计算为 `accs > mean`（strictly greater）；bestC 块=10（119/138 种子为 EQUAL 不计），C1 块=15 → 改名准确 |
| 外审核数复核 | ✓ 通过 | 15 种子全精度：10 above / 1 equal(119/138) / 4 below；t=2.043, p=0.0604≈0.060；mean=87.73, sd=2.84；TOST CI[-0.07,3.07]pp |
| M2/M3 脚本 | ✓ 已定位 | M3：scripts/probe_cgrid_valselected_20261026.py（seed-0 版，需 15-seed 变体）；M2：scripts/resnet_kshot_matched_budget.py（LR_GRID=[3e-4,1e-3,3e-3]，需低 LR 扩展 [3e-5,1e-4]） |
| Minor 2 来源 | ✗ 仓库无信息 | data/field_*.json 仅 date/gpu/n_images；无采集者/地点/协议 → 作者 TODO |
| Minor 3 文献数字 | ✗ 仓库无已核实值 | 无 references/ 目录、无 Lin2022 全文、审计文档无 SOTA 数字 → 本轮加自含定位句（不引用未核实数字），文献对比留作者补 |

---

## §4 处置

### 本轮已改（文本批，零结论变更）
1. **M5**：摘要（tex:37）、结论（tex:673）、田间节 (2) 句（tex:528）三处 reversal 断言 → directional 限定。
2. **Minor 1**：power bracket 改 held-out 同源对 (86.2% vs 87.7%)，1.7%→1.5%、23.3%→26.0%、span 5.8–11.5→5.8–12.2pp；"below the detection limit" 结论不变。
3. **M4**：新增 Appendix D（matched cross-crop 描述性数字 + different-split 不可比声明 + 数据文件路径）；tomato/corn 两处排除声明加附录指针。
4. **M1**：Discussion Interpretation 首句加"applicability boundary + deployment rule"定位（指向既有 Practical Implications，无新主张）。
5. **Minor 3**：Related Work 末段加自含定位句（best full-data 数字是 protocol contrast 的参照点，非 SOTA 主张）。
6. **Minor 6**：`n_seeds_ge_clip` → `n_seeds_above_clip`（data/claim_verification.json ×2，scripts/verify_claims.py ×3+1 打印标签，scripts/check_tex_vs_data_v1.py ×2）。
7. **Minor 2 加查发现（内部矛盾，比外审指出的更实）**：正文 tex:504 承诺 "see Appendix A for the class list **and collection protocol**"，但 Appendix A 只有 class list、无 collection protocol——论文许了未兑现的诺言。已移除假承诺（→ "for the class list"）；真实协议信息（采集者/地点/时段/设备/授权）留作者 TODO，不代笔虚构。

### 实验（M2/M3）执行状态
- ✅ **M3 完成**（`scripts/probe_cgrid_15seeds_20261104.py` → `data/probe_cgrid_15seeds_20261104.json`）：15-seed 验证选 C，K=1/2/5/10，与 ResNet matched 15-seed 逐 seed 配对（同一 `kshot_subset_idx(y,K,s+500)` draw）。
  - **结果：LAION-2B 的 K=1/2 "反转"不成立为显著领先**：K=1 40.8%±4.6 vs 40.4%（+0.3pp, paired t=0.20, p=0.84）；K=2 53.0%±4.8 vs 52.5%（+0.6pp, p=0.66）——均为不显著平局。
  - DFN-CLIP 仍显著落后：K=1 32.5%±5.0（−8.0pp, p=0.0005）；K=2 44.9%±5.0（−7.6pp, p<0.0001）；K=5 61.8%（−3.1pp, p=0.011）；K=10 71.5%（−3.1pp, p=0.005）。
  - K=5 LAION 68.2%±3.8（+3.2pp, p=0.032，描述性名义领先）；K=10 LAION 平局（+0.3pp, p=0.64）。
  - **管线自验证**：C=1 列 15-seed 均值与论文 Table 1 完全一致（DFN 30.5/36.5/48.0/56.4；LAION 40.1/48.8/60.2/66.5）。
  - **待改 6 处**：abstract (tex:37)、intro (tex:70)、TikZ 节点 (tex:95)、Methods (tex:180)、Results Finding 1 (tex:346)、RQ2 (tex:411)——"LAION 反转/领先" → "15-seed 下收窄为不显著平局、无显著 probe 领先；DFN 仍落后；probe 侧对 C 敏感（LAION K=1/2 +0.7/+4.2pp，DFN +2.0/+8.4pp）"。等 M2 一起批。
- ⏳ **M2 运行中**（`scripts/resnet_kshot_lr_extend_20261104.py`，后台 PID 2277869，日志 /tmp/m2_lr_extend.log）：低 LR 扩展 {3e-5,1e-4}×{30,100,300}=6 新配置，与已存 9 配置并集重选（selection 规则不变：seeds 0-2 平均 best-val）；仅当新配置胜出才触发 stage-2 15-seed。
  - K=1 阶段数据：3e-5 最高 val 31.2%（ep300）、1e-4 ep100 val 31.9%——均远低于已发表 3e-4/ep300 的 42.3%；K=1 胜者大概率不变。
  - 原则：M2 结果出来前，正文不预写方向。

### 作者 TODO（需作者输入，不可代笔）
- **Minor 2**：田间 246 图集来源声明（自采/公开？采集者、地点、协议、授权）。建议写入 Methods 数据集节（tex:150 现仅有 "collected from commercial tobacco fields under natural lighting"）+ Appendix A（tex:907 现仅有 class list）+ Data Availability Statement（现未列原始田间图像可用性）。**注意：tex:504 原承诺 "Appendix A ... collection protocol" 已移除（假承诺），补齐协议前不得恢复。**
- **Minor 3（完整版）**：2–3 个文献定位对比（如 TPDD 原文 Lin 2022 报告值）——需作者核实引用数字后再入文。
- **Minor 5（可选）**：标题候选（本轮未改，避免 cover letter / Highlights / 已引用处级联）：
  - A: "When to Fine-Tune CLIP and When to Deploy a Compact CNN: A Protocol-Controlled Comparison for Fine-Grained Tobacco Leaf-Disorder Diagnosis"
  - B: "Fine-Tuning or Frozen Probes? Empirical Applicability Boundaries for Off-the-Shelf CLIP in Tobacco Leaf-Disorder Diagnosis"
  - C: 维持原标题

### 验证记录（改完查全文）
- tex diff：恰好 11 处预期修改（10 批 + 1 假承诺移除），无越界改动。
- 双遍编译：0 错误、0 undefined reference/citation、30→30 页。
- 双 PDF 词袋对比（PRE 独立编译 vs POST）：全部差异可归因于预期修改 + 换行重排伪影，无意外内容变更。
- 一致性 grep："reverses under field" 归零；"23.3"/"11.5" 归零；89.4 仅余 5 处全在 CV 语境（正确）；附录 D 引用 3 处（2 指针 + 1 定义）。
- 回归：v2 校验器 check_tex_vs_data.py **34/34 全过**；改名后 JSON 键值正确（bestC 10/15、C1 15/15）；两脚本 py_compile 通过。
- **既有 bug（非本轮引入，未动）**：scripts/check_tex_vs_data_v1.py:104 使用未定义变量 `pk`，git HEAD 原版同样崩溃（早于改名代码行即崩）。留待后续修，不在本轮范围内。
- 新数值复核：Appendix D 的 6 个均值±SD 从 data/resnet_k*_matched_*.json stage2_final 重算一致；Minor 1 括界新值 (86.2, 87.7)→|Δ|1.5pp、上界 26.0%、span 5.8–12.2pp 独立重算一致，"below the detection limit" 结论不变。

### Response letter 要点
- Minor 1 回应时顺带说明：bracket 改为 held-out 同源对后，括界下界 1.5pp 与正文报告的 1.5pp top-1 gap 自洽（原 1.7pp 下界与 1.5pp gap 本身有 0.2pp 内部不一致，已一并消除）。
- Minor 4：引用论文已有限定句原文 + 承重墙说明。
- M1：说明 C&EA 读者画像与 ML 社区先验差异，接受任务导向重述（已改）。
- M5：承认摘要-正文强度不匹配，已三处降级。
