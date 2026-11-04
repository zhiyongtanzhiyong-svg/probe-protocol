# 审稿意见存档 — 十三轮复审 + ChatGPT/GLM 外审模拟 + GLM 点评

> 存档目的：避免遗忘。以下为三份评审意见的原文存档（2026-11-03），随后附交叉验证与处置决定。
> 处置原则（用户指示）：不防御、不辩解、把每条批评转成改进契机；不糊弄、不过度包装；
> 小修不如不改，改就要成批改、改完查全文；改不动结论的内容，加进去只会增加自相矛盾的风险。

---

## 文档 A：十三轮审查（v13 复审）

推荐意见：Minor Revision（终门）。这是十三轮里执行最完整的一版——三方清单几乎全部落实，外加两项新实验；我本轮仅发现一处新的实质问题（代价分析的列归属），另有四条一句话级遗留。P0 改写了但未关闭。

### 一、本轮闭合项（核验通过）
- 行政三项 + 新增 COI。 Funding 节与题目页对齐（三项基金）✓；贡献声明幽灵作者（F.H./X.T.）清除，五作者各归其位 ✓；通讯作者统一为 L.Z. ✓；新增利益冲突披露（X. Luo 任职 + 烟草资助）——这是此前该有的句子。
- ChatGPT 轮全部实质项。零样本范围：摘要限定 "On TPDD"、Practical (1) 与 Conclusion 改为双轨表述（烟草集 ≤ 基线 / 跨作物与田间高于各自的低基线），且 Conclusion 主动给出混杂解释（细粒度 vs 近天花板构成）——比我给的模板更好 ✓。best-of-14：主次对调，且实现了更强的版本——明确写出单调性论证（"取最大值只会抬高精度，故最佳模板 ≤ 基线 ⟹ 全部模板 ≤ 基线"），p 值降为 "nominal, post-selection, non-inferential"，SigLIP p=0.0025 同步标注 ✓。新增配对 McNemar（b=16/c=18 等，内部自洽）+ bootstrap CI ✓。功效段重写：给出 √(p_d/n) 公式、p_d≈0.16 明标为 planning assumption、可推导区间 [6%, 23.3%] → 检测极限 5.8–11.5pp（SE 与极限算术全部吻合）✓。混淆矩阵色阶/计数注明 ✓；TikZ 图"不承载独立证据"注明 ✓；"full protocol"→"frozen-probe arm" 全线统一 ✓；"matched budget" 摘要内联定义 + Methods 操作定义 ✓；field 段新增无检验声明与 SE≈6pp ✓；番茄近重复样本给出界论证（单图 ≤0.06pp << 11.1pp 最小显著对比）✓。
- qwen 轮合并清单全部落实。npz：Appendix A 现披露 DFN-CLIP C=1 探针 legacy store 重拟合 76.1 vs canonical 76.8（0.7pp）——与 qwen 实测的 76.09/76.81 精确对应；"only place" 声明更新为含审计文件的范围表述；LAION 旧 store 标注已废弃 ✓。误差代价矩阵：新增 Discussion 7.2 三档权重（3/2/1）+ 权重鲁棒性——但见第二节（本轮唯一实质新发现）。网格平台曲线：新增 Appendix C + Limitation (10)，诚实命名脆弱方向（ResNet 网格欠覆盖可能高估 FT 领先）并说明为何不扩展（会破坏匹配预算）——以自限代替辩护，正确 ✓。单例类：Limitation (9) 给出"60 图训练池、刻意不再平衡、保护预注册划分"的完整理由 ✓。DINOv2-FT：真的跑了——matched 9-config 全套 few-shot + 全数据（95.6±1.5，K=1 平 ResNet 40.2/40.4，K=2/5/10 领先 5.0/10.4/10.5pp），2×2 矩阵补齐，"非 backbone-pairing 伪影"结论获得实验支撑；硬件差异（A100 vs V100）披露并附 0.05pp 交叉验证 ✓。稳定性限定句：RQ2 答、CV 节、Limitations (3)、Conclusion 四处齐备，且 CV 节新增逐折 gap（−2.3/−2.3/−2.7/−3.5/−4.6；复算配对 t：mean 3.08, SD 0.98, t≈7.0, df=4 → p≈0.002 ✓）。
- 我 v11 台账。K∈{1,2,5,10} ✓；表注 K=1/2 ✓；SigLIP 指针改指 Table 1 注 ✓；34.1 标注 "single deterministic fit, seed-0 support set" ✓；"natural expectation" ✓；readout 表注与 Table 1 互引（66.4 vs 66.5 的 5-seed/15-seed 差异说明）✓；macro-F1 双 83.7 之谜以替换解决——换成定义明确的逐类 F1 平均、探针 vs frozen ResNet 探针的同协议对比（67.5/77.3 vs 64.3/72.4）✓；Conclusion "untested hypothesis" ✓；TPDD matched 同划分运行时断言 ✓；玉米 K=1 跨作物族内 Holm 核验（0.021×2=0.042 存活，与 JSON 挂钩）✓；legacy 混合行拆分为 matched 行 + superseded 行 ✓；还自加了 legacy 种子配对重跑（32.0%，−1.55pp, p=0.26, n.s.）——把 DFN-legacy K=1 对比标记为 instantiation-sensitive 且非承重 ✓；Results 开篇 "two findings carry the paper" 的 signposting ✓。

### 二、本轮新问题（一处实质 + 三处小）

**1.（实质）代价分析用的是 C=1 列，但未标注——而这正是论文自己定义的"欠调参探针"。**
逐格验算：cost-weighted recall 66.2% 与 viral-tier 51.2% 精确匹配 Table perclass 的 CLIP C=1 列（权重和 28：w=1 项 625.3 + w=2 项 737.8 + w=3 项 743.7 = 2106.8/28 = 75.24 是 best-C 的值；C=1 列得 66.16 ≈ 66.2 ✓；viral 层 C=1 (80+25+100+0)/4 = 51.25 ✓；ResNet 81.0 两种列一致 ✓；鲁棒性行 66.8/81.1、65.6/80.3 亦全部复算吻合 C=1）。问题在于：(a) 正文只说 "the two full-data models of Table perclass"，三列中未指明；(b) 全文其余处用 best-C 代表 frozen 探针，此处静默切换到 C=1，敌意审稿人一句就能打中："你们用自己论证过欠调参的探针来放大 CLIP 的代价劣势"。修法（两句话）：明确标注 "default C=1 probe"，并同时给出 best-C 版本——75.2 vs 81.0（+5.8pp），viral 层 62.0 vs 74.1（其中 13.3pp 来自 TSWV 单图）。结论方向两个列都存活（ResNet 仍领先），但量级差 2.5 倍，且 best-C 下 PVY 反超（75 vs 60.8）、viral 劣势转为单图驱动——诚实的版本应该把这个削弱说出来，它仍然支持"不要部署 frozen 探针于高代价场景"，只是从 +14.8pp 缩到 +5.8pp。现在的写法是把七轮拆掉的 C=1 伪影在新区块里又搭了一间。

**2. npz 命名陷阱（仓库侧）。** qwen 实测的文件名是 clip_features_clean.npz，而 Appendix A 现在披露的就是该 store 与 canonical 的 0.7pp 分歧——若 "clean" 命名的文件实际是 legacy store，任何照名取文件复现的人会得到 76.1 而非 76.8。改名或在 feature_store_audit.json/README 顶部一行标明。

**3. "meets fine-tuned CLIP at K=5"**（75.3 vs 77.1）。DINOv2 在 K=5 落后 DFN-CLIP 1.8pp（各自 ±5.7/±4.8），"meets" 略强，改 "statistically indistinguishable" 一词即可；K=10 的 85.1/85.1 是真平。

**4. macro-F1 静默替换。** 旧 83.7 双值消失、新数（67.5/77.3/63.4/76.2 等）定义清楚且内部自洽——修复正确，但数字换血应有溯源（新 JSON + 一句 "recomputed under the per-class-F1 definition"），否则仓库对不上正文。

### 三、遗留台账（全部一句话级）
| 条目 | 轮次 | 备注 |
|---|---|---|
| 81.2−77.6=3.5（应 3.6） | 0/13 | 建议用全局注一次了结："differences computed from unrounded values; printed-value arithmetic may differ by ≤0.1 pp"——同时杀掉下面两条 |
| 番茄 CV −1.6 vs 表 98.0/99.5=−1.5 | 0/13 | 同上 |
| +4.6–12.2（正文/公平段/TikZ 图三处）vs 表 79.3−74.6=4.7 | 0/13 | 同上 |
| Evaluation→sec:why 的 balanced-acc 分解指针仍悬空 | 0/5 | sec:why 补一句（排除四个单图类后余 −0.3pp；仅移除 class 13 使 3.5→0.2）或删指针 |
| 摘要 ~430 词（C&EA ~300） | 0/3 | 投稿排版时处理 |
| P0 仓库 | 第 13 轮，形态变更 | 占位符字符串已消失，改为 "URL withheld for double-blind review"。两个问题：C&EA 惯例为单盲，"double-blind" 理由错配；更实质的是正文引用 30+ 个 data/*.json，仓库不在评审期可见意味着论文的验证品牌在评审期不可验证。标准解法：anonymous.4open.science 镜像。这是投稿前最后一个阻断项 |

### 四、结论
评分：8.5/10（苛刻口径，本轮 +0.5——十三轮首次上调）。上调理由不是编辑执行（那只是还账），而是两项新数据：DINOv2-FT 把 Finding 1 从"两个 CLIP checkpoint 对一个 CNN"升级为跨对比/自监督鸿沟的协议现象；legacy 种子配对重跑把最后一条 instantiate 敏感的对比主动降级。加上 Limitation (10) 对自家招牌数字的自限，这一版的证据-主张对齐达到了它声称的水平。

投稿前清单（终版）：① 仓库匿名链接（一条 URL）；② 代价分析列标注 + best-C 两句话；③ 全局未舍入注（清三条算术）；④ sec:why 一句或删指针；⑤ npz 命名一行；⑥ "meets" 一词；⑦ 摘要压缩。合计半天。

---

## 文档 B：ChatGPT 外审模拟报告（GLM 汇总）

### Computers and Electronics in Agriculture — 修订稿深度模拟同行评审报告｜编辑 + Reviewer 1 + Reviewer 2

稿件：Frozen Probes versus Matched Budgets: Protocol-Dependent Ordering of CLIP and ResNet-18 for Fine-Grained Tobacco Leaf-Disorder Classification
评审基础：probe-protocol.tex（约 885 行，当前上传版本）。性质：学术模拟评审；不是期刊真实编辑决定。

### 一、综合判断与投稿风险
- Handling Editor：Major Revision；若关键公平性及数据溯源问题未解决，可能拒稿
- Reviewer 1（机器学习/统计）：Major Revision；主张需要更严格的搜索空间、数据与统计验证
- Reviewer 2（农业工程/落地）：Major Revision / borderline；田间外部效度、标签可靠性及应用价值证据不足

### 二、编辑决定函（模拟）
Decision: Major Revision.
- **E01 贡献定位与文献增量**：引言同时强调"off-the-shelf CLIP 的边界""模型排序随协议变化""KG prompt""田间部署"，这些并非同一主贡献。建议将主贡献界定为可复核的模型选择/评测协议，并用 1 张差异矩阵说明创新。定位：Introduction, RQ1–RQ3, Findings (1)–(5)
- **E02 决定性比较的有效性**：附录 C 已披露 ResNet 最佳学习率在网格下边界，且 epoch 曲线仍上升。9 对 9 配置数相同，不代表搜索空间对称、模型已收敛或计算开销相等。建议主标题/摘要的"matched budgets"限定为"matched number of candidate configurations"。定位：Abstract, Limitations (10), Appendix C
- **E03 结果适用范围**：PlantVillage 番茄和玉米来自同一数据来源，仅复现 frozen-probe 分支；田间模型用田间训练集重新训练，未测试 TPDD→田间直接迁移。不得把这两部分合称"独立跨域泛化验证"。定位：Cross-crop, Field-condition, Conclusions
- **E04 复现包与数据谱系**：TPDD 原 11MB 图像包已不可得；现用 1,571 张重导版本。数据链接与 manifest、文件哈希、标签映射、损坏样本处理必须公开。当前 Data Availability 的 URL 仍是待插入项。
- **E05 稿件组织**：大量例外与 caveat 被埋在摘要长句、脚注和 Limitations；建议增加 protocol matrix 与 effect-size summary；压缩摘要。

### 三、Reviewer #1｜实验设计、算法比较、统计推断
- **R1-01**（Major）"同配置数"与"同优化机会"混淆：ResNet-18 学习率最优值位于 3×10⁻⁴ 下边界，属诊断性证据。建议报告每格验证性能、最优轮次、最优点边界；新增低学习率与更长 epoch 网格作为 sensitivity。定位：Appendix C, Limitations (10)
- **R1-02**（Major）全参数微调的计算成本：DFN/LAION ViT-L/14 各约 428M 参数，ResNet-18 约 12M；相同 9 个超参组合 ≠ 相同 GPU 小时/训练步数/显存/能耗。建议增加 GPU-hours、训练时长、峰值显存等。
- **R1-03**（Major）C 网格仅 K=1/2：稿件明确小 K 的验证选 C 仅单 seed exploratory，K=5/10 冻结探针沿用 C=1。K=5/10 的"ResNet leads"不能视为 probe 充分优化后的稳定结论。建议全部 K、全部 15 seed 内验证选择 C。
- **R1-04**（Major）支持集/验证集构造：少样本每类 K 张训练图像后如何获得与训练严格分离、每类代表性足够的验证样本？K=1 时调 C 的验证误差可能高度不稳定。
- **R1-05**（Major）预处理不是"等价"：canonical 与 clean path 图像特征均值 cosine=0.90，C=1 probe 测试差 2.2 pp，不能支持"高度相似/鲁棒"这一强结论。
- **R1-06**（Major）种子重复 ≠ 新样本重复：15 seed 配对检验固定在同一 138 张测试图像，测的是训练抽样/初始化不确定性，非来自未来田间样本的总体推断。Holm 校正不能修复错误统计单位。
- **R1-07**（Major）16 重比较家族与探索性分析：为何仅这 16 项为一个 family？是否观察结果后确定？
- **R1-08**（Major）CV p 值的解释：5 折训练集重叠约 80%，fold t 检验 p=0.002/0.03 不应作为独立重复的确认性证据。摘要中"sign-stable"容易产生强证据印象。
- **R1-09**（Major）全数据探针的训练/验证/重拟合描述："fit 80%, validate 20%, refit 100%"易让人误解 100% 是否包含最终 held-out test。
- **R1-10**（Major）测试样本量与类别指标：TPDD 138 test、4 类 singleton；每类召回的 0%/100% 只是一个观测。建议给 per-class support 与 Wilson CI；去 singleton 敏感性。
- **R1-11**（Major）泄漏审计阈值选择：TPDD CLIP embedding 相似度阈值 0.99 高于观察最大值 0.976。建议列出 0.95/0.97/0.98/0.99 阈值下的候选数和人工核查。
- **R1-12**（Major）番茄明确存在 train/test 重复：文中发现一张 cosine=1.0 的完全重复图像。建议移除重复后重算对应测试指标。
- **R1-13**（Major）原始图像版本丢失：稿件承认旧版 TPDD 图像包不可得。建议给新版图像哈希与下载脚本；不要把 Welch p=0.15 解释为两数据版本等价。
- **R1-14**（Major）模型容量和预训练差异：ViT-L/14 428M 与 ResNet-18 12M 差距混合了架构、参数量、预训练数据、训练目标。建议表格逐格标明训练方式、预训练集、参数量、数据预算。
- **R1-15**（Major）LoRA 比较解释：LoRA r=16 仅 0.19% 可训练参数，但其 93.5% 接近全微调不意味着优势"不是由于微调预算"。
- **R1-16**（Major）零样本 best-of-14 的选择偏倚：用 14 个 prompt 的 test best 作为上界可以描述，但若实际根据 test 选模板就不构成独立模型选择。
- **R1-17**（Major）机制性"decision-rule mismatch"仍非因果证据：冻结表征被监督 probe 利用，只说明信息可被某类监督 readout 提取。建议将"indicating"改为"consistent with"。
- **R1-18**（Major）校准与 ECE：τ=100 直接影响 confidence 和 ECE；ECE=0.48 必须清楚指定温度、分箱、样本数及拟合集。
- **R1-19**（Minor/Major）推理 benchmark 设定：RTX 4070 Ti、batch=138、FP32、3 次重复与真实单图田间推理的延迟需求不同。
- **R1-20**（Major）复现性与非确定性：同 seed 的 CNN 复跑跨 86.96–89.86%。建议记录 cudnn.benchmark、deterministic_algorithms、CUDA/torch 版本。

### 四、Reviewer #2｜农业植保意义、外部验证与可部署性
- **R2-01**（Major）任务标签的植保学含义：TPDD 16 类混合病原病害、病毒、虫害、非生物胁迫、遗传异常和 healthy。正文多处泛称 disease。
- **R2-02**（Major）数据来源与真实场景偏差：TPDD 公开谱系存在 2,721/12 类与重导 1,571/16 类的差异。
- **R2-03**（Major）TLD 标签可能重叠：Brown spot/Alternaria alternata、Frogeye/Cercospora nicotianae 等名称可能指向同一或重叠疾病。
- **R2-04**（Major）田间数据设计不支持直接域迁移：246 张田间图像 193 训练、53 测试；因此这是 field-domain in-domain adaptation，不是 TPDD-trained model 在 field test 的 domain shift robustness。建议更名小节并在摘要/结论澄清。
- **R2-05**（Major）田间样本太少且类数太多：53 张 test 覆盖 19 类，69.8±2.7 vs 76.7±1.6 的差异不能凭 seed SD 确认为未来田间总体差异。
- **R2-06**（Major）田间排序反转是核心负面结果：控制域 CLIP 微调领先，田间 DFN-CLIP 69.8% 低于 ResNet 76.7%。建议将建议限定为 TPDD 条件。
- **R2-07**（Major）跨作物仅同源 PlantVillage：番茄与玉米都来自 PlantVillage。不要称 independent external validation。
- **R2-08**（Major）少样本设定的农业成本意义：K=1/2/5/10 每类标注样本是否能由农技人员可靠提供？
- **R2-09**（Major）误诊成本矩阵缺少专家依据：成本权重 3/2/1 只是研究者设定。建议提供专家定义的场景矩阵及多组权重敏感性。
- **R2-10**（Major）拒识与转人工流程缺失：真实植保应用不能强制每张图归入 16 类。
- **R2-11**（Major）部署建议与目标设备不匹配：当前只有桌面 4070 Ti 的吞吐和峰值显存。
- **R2-12**（Major）病害风险与可解释性：仅 top-1 accuracy 难以说明误诊是否集中在高危病种。
- **R2-13**（Major）KG prompt 的贡献不足：7/16 类缺少知识图谱描述而用 fallback。
- **R2-14**（Minor）中英文提示的解释：英文中心 tokenizer 对中文模板的弱表现可能源于语言能力。
- **R2-15**（Major）公平部署选择应含 fine-tuned CLIP：成本表仅列 frozen CLIP probe 与 ResNet，主结论最强的 fine-tuned CLIP 未纳入同一准确率–成本 Pareto 图。
- **R2-16**（Minor/Major）缺少外部专家或实地工作流验证：论文是算法评估，而非农艺师验证的决策支持系统。

### 五、跨章节内部一致性与可立即修正问题
- 摘要称"all code, data and scripts are in the repository"，而 Data Availability 中匿名 URL 尚未插入。
- 引言中"TPDD/TLD 受控烟草数据"与 TPDD 数据来源中 field images 的描述可能让读者误解采集环境。
- 引言 RQ3 询问 zero-shot/few-shot/full-data 的跨作物排序，但 matched fine-tuning 只在烟草开展。
- "15/16 Holm 显著"与"8/8 fine-tune contrasts 显著"均须有完整逐项统计表。
- 表中 balanced accuracy 81.2−77.6=3.6 pp（若按未舍入数据为 3.5 pp，应标注"基于未舍入值"）。
- 田间段落和结论中的"generalization"须统一为"field-condition evaluation / within-field adaptation"，并写明 193/53。
- 模型选择中的"matched 9-config"要与"equal compute"在每张图表标题/脚注中重复区分。
- 各数据集的 test-best template、validation-selected prompt、default C、validation-selected C 需要不同符号和表头。
- 作者贡献与匿名评审政策需以期刊投稿系统选择的审稿模式核实；不要未经确认声称本刊一定双盲。
- 建议检查所有引用键、图文件、表格交叉引用及代码仓库路径。

### 六、分级返修路线图
- **P0**（决定主结论）：① ResNet 学习率/epoch 扩展；② CLIP 全 K 全 seed C 搜索；③ 官方 preprocessing 审计；④ 可复现数据清单与预测输出；⑤ 结果表述严格限定到指定协议。
- **P1**：① 去除 PlantVillage 重复并复算；② seed/image 统计单位分离；③ per-class uncertainty；④ 田间域内/跨域概念修正；⑤ 实验矩阵和准确率–成本 Pareto。
- **P2**：① 农业标签专家核验；② KG 覆盖率消融；③ 改写摘要和贡献；④ 图表与附录索引；⑤ 投稿格式与数据共享声明核验。

### 七、建议新增的关键表格与图件
- 表 S1：数据谱系；表 S2：模型×数据×K×适配×调参×seed×验证×GPU-hours 全矩阵；表 S3：16 个主对比的每 seed 差值/均值/95% CI/原始 p/Holm p。
- 图 1：防泄漏流程图；图 2：超参验证曲面；图 3：跨受控/田间排序；图 4：精度–推理延迟–显存 Pareto。

### 八、建议修改的核心论断（英文示范）
"On the re-derived TPDD split, model ordering varied substantially with adaptation and model-selection protocols. Under the specified nine-candidate fine-tuning searches, CLIP outperformed ResNet-18 across the evaluated few-shot settings; however, the ResNet-18 optimum occurred at the search boundary, and this result should not be interpreted as an architecture-level or compute-matched superiority claim. Frozen-probe comparisons were sensitive to regularization selection. Cross-crop experiments replicated only the frozen-probe arm on two subsets of PlantVillage, while within-field adaptation yielded a reversed ranking on a small independent field dataset. These observations motivate protocol-aware model selection rather than a universal preference for one backbone."

### 九、最终模拟建议与复审门槛
- 编辑：Major Revision。复审门槛为 P0 五项的可核验证据；如作者不能补实验，应重新定位为"指定协议下的探索性观察"。
- Reviewer 1：Major Revision。优先检查 R1-01、03、05、06、11、20。
- Reviewer 2：Major Revision / borderline。优先检查 R2-02、04、05、06、09、15。

---

## 文档 C：GLM 对 ChatGPT 审稿意见的点评（triage）

### 一、为什么感觉越改越多——三层原因，没有一层是"论文变差了"
1. 可审查面积在扩大（正文→图件脚本→TikZ→附录→仓库清单）。
2. 模拟评审的激励结构（35+ 条意见 + P0/P1/P2 + 7 表 4 图是"评审"文体的最大化产物；真实 C&EA 审稿人典型产出 2–4 条 Major + 若干 Minor）。
3. 真正的新洞确实有——但只有两个，而且都便宜。

核心问题数不是在膨胀，是在单调收敛。十三轮前约 10 条实质问题，现在剩 2 条可执行的实验问题 + 1 个 URL。

### 二、这份报告的 triage（~50 条的真实构成）
- **v13 已完成，报告滞后（~9）**：R1-16、R1-08、R2-07、E03、R1-06 大半、R1-10 的 Wilson CI 等
- **一句话/一段话文本修复（~11）**：R1-17 "indicating"→"consistent with"；R1-09 refit-100% 指代；R1-05 "highly similar"→如实；R2-04 田间节改名 "within-field adaptation"（采纳）；R2-06 Practical 加 TPDD 限定；R2-12 非诊断声明；R2-10/R2-16 future work；未舍入全局注；双盲问题
- **便宜的分析/测量（~8）**：R1-11 泄漏阈值敏感性表；R1-12 番茄重复图删除复算；R2-13 KG 覆盖子集消融；R1-19/R2-15 batch=1 延迟 + Pareto；R1-02 GPU-hours；表 S1/S2/S3
- **应当驳回的（~10）**：E02 标题改名；R2-03；R2-09 专家权重验证；R2-11；R1-07 时间线；R2-01/02 最小化响应
- **真实验（2 个）**：见下

### 三、两个真实验——诚实的风险评估
- **实验 1：C-sweep 扩到 K=5/10 × 15 seeds（R1-03）。** 成本小时级。风险：K=5 的 LAION 差距只有 4.7pp（64.9 vs 60.2），而已观测的 C 选择增益 +3.6 到 +10.1pp——K=5 的这一个对比完全可能收窄甚至翻转。DFN 侧（16.9/18.2pp）几乎必然存活。两种结局都是赢。
- **实验 2：ResNet 网格下界扩展（R1-01/E02）。** 成本几个 GPU 小时。风险：8 个对比的最小 margin 是 K=10 LAION 的 4.7pp；ResNet 再涨 2–3pp 合理——8 个对比里最弱的 1 个可能被抹掉；DFN 侧（8.5–12.2pp）难以撼动。更干净的框架是"ResNet-extended sensitivity"。

这两个实验是全文最后两个真正的科学不确定性。

### 四、合并后的终版清单
科学（投稿前）：两个实验（1–2 天）；代价分析列标注 + best-C 两句话；R2-04 改名 + R2-06/R2-12/R1-17 等一句话批；未舍入全局注 + sec:why 分解句；便宜分析（KG 子集消融、番茄去重、阈值表）。
投稿阻断：仓库 URL；"double-blind" 措辞与 C&EA 单盲惯例对齐。

### 五、何时停止
停止条件：当边际实验不再可能改变任何印出的数字时。跑完上面两个实验后，这个条件即满足。

---

## 文档 D：交叉验证决策报告（三方意见 × 数据核验 × 实际修改）

原则：只改「核验为真 + 不改变结论」的项；改一处就成批改，改完查全文；不为辩护而辩护、不糊弄、不过度包装。

### D.1 已改（7 处，全部经数据核验 + 全文一致性检查 + 重新编译通过）

| # | 意见来源 | 问题 | 核验结论 | 修改 | 位置 |
|---|---------|------|---------|------|------|
| 1 | v13 实质 #1 / B-R1-成本列 | 代价加权召回只给默认 C=1（66.2% vs 81.0%），显得 CLIP 劣势过大 | 真：best-C 下 75.2% vs 81.0%（+5.8pp），viral tier 62.0% vs 74.1%，PVY 75%>ResNet 60.8%；TSWV 两臂均 0% | 标注默认 C=1 为 under-regularized，补 best-C 两列，说明方向不变、margin≈2.5× 来自 C=1 | line 546 |
| 2 | B 跨章节 #5 | 3.5/3.6、4.6/4.7、−1.6/−1.5 未舍入算术 | 真：balanced gap=3.54pp，K=10 LAION FT=4.637pp | Evaluation 节加全局未舍入注 | Evaluation 子节 |
| 3 | B-R1 / v13 | sec:why 平衡准确率分解 | 真：去 4 个 singleton → CLIP −0.28pp 反超；只去 TSWV → +0.22pp | 补分解句，点明 singleton artifact | sec:why |
| 4 | v13 小 / B-R2-04 | "Field-Condition Generalization" 名不副实（实为 within-field 适配） | 真：field 集是独立采集、193 训练/53 测试 | 改名 "Field-Condition Evaluation (within-field adaptation)"，label 不变 | line 460 |
| 5 | B 投稿阻断 / v13 | "double-blind" 与 C&EA 单盲惯例不符 + 匿名镜像措辞 | 真：C&EA 单盲 | Data Availability + Limitation(7) 改措辞，指定 anonymized mirror | 相应处 |
| 6 | B-R1 / v13 | "meets the ResNet"（DINOv2 K=5）过度宣称 | 真：75.3% vs 77.1% 在误差内 | 改 "statistically indistinguishable"；摘要 "indicating"→"consistent with" | line 317 + 摘要 |
| 7 | v13 小 | npz 命名歧义（clip_features_clean.npz 为 legacy） | 真：canonical_clip_features_dfn.npz 为 canonical DFN | feature_store_audit.json 加 naming_note + canonical_files | data/ |

编译：pdflatex 两遍，29 页，无 undefined ref/citation，无 rerun。

### D.2 不改（理由）—— 避免过度包装/自相矛盾

- **macro-F1 provenance 指针**（77.3/64.3/72.4/63.4/76.2 的来源 JSON）：无法在 data/ 定位唯一来源文件。加一个未核验的指针 = 糊弄。→ 走回复信说明。
- **~40 条 ChatGPT 其他项**：绝大多数「已满足 / 应拒绝 / 廉价分析」。逐条塞进正文会增加自相矛盾风险且不改变结论。→ 全部进回复信逐点回应。
- 凡「不能改变任何已印数字、只增加表述复杂度」的项，一律不动。

### D.3 需真实验（本轮不跑，投稿前 1–2 天，跑完边际实验不再改变任何印出数字即停）

1. **C-sweep K=5/10 × 15 seeds**：验证 few-shot 下 best-C 与结论稳定性。
2. **ResNet grid 下界扩展**：确认 grid 搜索下界不是人为压低 ResNet。

### D.4 投稿阻断项（非本轮正文范围）

- 仓库 URL 落地（Data Availability 已改为 anonymized mirror 措辞，需补真实链接）。

### 结论
7 处已改且全文一致、编译通过；其余或进回复信、或需实验。停止条件满足：正文已无「可安全修改且能改善」的悬项。
