# Paper2: CLIP vs ResNet - Tobacco Dataset
## TODO List

---

## 【2026-10-31 第八轮】13 轮评审（ChatGPT）处置 —— 已完成

### 评审定性
- 它审的是**旧快照 `(4).tex`**（第七轮前，URL 占位符仍在）、附件无图无数据、其 TeX 环境编译失败、脚本是旧文件 `generate_figures_optimized.py` → E1.2/A3 等是**附件管道假象**；R1-5/R2-6/K=1 星号/逐类 CI/BioCLIP 0.051 均已在前面轮次修掉（它自己确认）。
- 真问题 = A1（best-of-14 名义显著性 vs L239 FWER 自相矛盾）+ 7 个小项，全部已执行。

### 已执行（18 处 tex + 2 处脚本，28 页 clean build，备份 `probe-protocol.tex.bak-rev8-20261031`）
- [x] **A1**：L231 重写为 "nominal, post-selection, non-inferential"（模板同集选择，p 不控制 14 路搜索）；SigLIP p=0.0025 同步；L237 后加**配对 McNemar+bootstrap** 段（方向：否定结论对 marginal/paired 选择稳健）；RQ1 答案的 p≤0.051 加 "nominal post-selection and descriptive"。
- [x] **新脚本 `scripts/paired_zs_baseline.py`** → `data/paired_zs_baseline_20261031.json`：从归档逐图 preds 算（**验证门：42/42 模板重算精度与归档一致**）：DFN b=16/c=18 p=0.86、LAION 17/18 p=1.0、BIO 14/18 p=0.60；配对差 CI 跨零偏负；**p_d 观测 0.23–0.25**。SigLIP 无逐图归档（保持 descriptive）。
- [x] **A4**：结论 "tobacco-specific" → "descriptive contrast across our own evaluation settings…not a claim that tobacco as a crop causes a zero-shot deficit"；摘要加 "On TPDD" 限定。
- [x] **A5**：混淆矩阵图注加 "Colour scale: row-normalised recall…numbers inside cells are raw image counts, not proportions"。
- [x] **R1-3 功效段**：SE=√(p_d/n) 公式 + planning p_d≈0.16 声明为假设（full-data 对无逐图归档）+ bracket [1.7%, 23.3%] + per-class 表推下界 p_d≳6%（Σn_c|Δ_c|）→ 检测限 5.8–11.5pp（planning 9–10pp 在内）+ 零样本对观测 p_d=0.23–0.25。1.5/3.5pp 缺口全 bracket 下不可分辨 ✓。
- [x] **⚠️ `data/p2_results.json` 是旧时代产物**（clip_full=75.4% 为 canonical 前 LAION C=1 值）→ **禁止引用**；其 McNemar 不对应当前 C=100 probe。
- [x] **R1-6**：L129/L410 "full protocol" → "frozen-probe arm across all four regimes"。
- [x] **R2-2**：4 处 "determined by" 软化（L73 首现加 "in the settings measured here…within-task regularity…not a universal law"；L310/372/622 → "is set by"）。
- [x] **R2-4**：TikZ 图注加 "summarizes the reported results…no independent evidence; CV cell rests on five non-independent ≈80%-overlapping folds"。
- [x] **B1（顺手）**：摘要首现 "matched fine-tuning budget---a matched 9-config search procedure, not equal compute cost (Methods)"。
- [x] **R1-4（定量界限版）**：L134 → 单图 ≤0.06pp vs 番茄最小显著对比 11.1pp（K=5；K=10 14.4pp）→ 不可能翻转显著性。
- [x] **R1-8**：L456 跨作物 3-家族 Holm 句加 `data/m2_powered_stats.json` 引用（raw p 可复算）。
- [x] **R4.3**：`generate_figures.py` `--no-gpu` help/docstring 改为 "legacy alias…no longer GPU-dependent"。
- [x] 构建：28 页，0 错/0 overfull/0 未定义；pdftotext 18/18 验证。
- [x] **R2-1（导读句版，用户确认“不做结构大改”）**：Results 开头 L189 后补两句——两条承重结论（零样本 decision-rule mismatch；ordering set by adaptation protocol）+ “其余全部分析为支撑证据，逐条 scoped，无一条结论依赖其中任何一项”。28 页 clean build 验证通过。
- [ ] **回复信素材**：按上文“假象/已修/本轮修/待人工”四类逐条回应 13 轮意见。

---

## 【2026-10-30 第七轮】投稿前收尾 —— 状态与待办

> 队列进度（10-31 早）：K=1 stage-1 完成（winner lr=2e-5 ep=300, val 43.72%），stage-2 15 seeds 进行中（seed0 test 40.58 / seed1 36.23）；随后 --full（大头 2–4 天）→ K=2/5/10。

### 已完成（本轮，全部已验证）
- [x] **项1 npz/canonical 对齐**：根因 = `clip_features_clean.npz`（DFN legacy，重建前随机增强）漂移（cosine 0.83–0.97，C=1 → 76.09 vs canonical 76.81）；LAION legacy（75.4）已被取代、仓库内两个 LAION store 均为 fresh。Appendix A "only place" 句已重写为当前事实（DFN C=1 为唯一受影响报告值；0.7pp 无符号/排名翻转；legacy 76.09 = 重建前 full_provenance.json 记录值）。
- [x] **项2 错误代价**：Discussion 新增 §Cost-weighted diagnostic reliability（病毒级 PVY/TMV/TSWV/CMV w=3；重症 w=2；其余 w=1；DFN 66.2% vs ResNet-FT 81.0%；病毒层 51.2 vs 74.1；权重 2/1.5/1 与 2/1.2/1 下排名不变；healthy→0 不变；权重为显式假设已声明）。
- [x] **项3 双网格平台曲线**：`scripts/fig_grid_plateau.py`（新）→ `figures/fig_grid_plateau.{pdf,png,svg}`；新增 Appendix C + Methods matched-budget 段加一句 + **Limitation (10)**。如实结论：CLIP 面平坦（span 5.3pp、选中在内部）；ResNet 面陡峭（span 24.2pp、选在最低 lr 边界、epochs 仍在升）→ 声称 matched 的是搜索程序而非调优质量；若 ResNet 真最优低于 3e-4 则 matched-FT 边距可能被高估（frozen 臂不受影响）。数据源：`data/p3b_clip_finetune_kshot.json`（laion2b K=5）、`data/p3b_resnet_kshot_matched.json`（K=5）的 stage1_search。
- [x] **项4 split 核查**：四个 n=1 测试类（anthracnose/black_shank/genetic_abnormality/tswv）各 60 张训练池，再平衡可行但刻意不做 → **Limitation (9)** 已加（预注册 re-derived split 不可事后改）。
- [x] **项6 稳定性限定**：三处 "descriptive" 核实齐全（RQ2 答案 / CV 段 / Limitation 3），无需改动。
- [x] **项7 行政**：ghost authors（F.H./X.T.）已不在作者列表；对应作者仅 L.Z.；资金声明标题页与 Funding 一致无矛盾；**URL 占位符已改为 double-blind 措辞**（DAS L626 + Limitation 7 同步；DAS 补 manifest 与审计文件路径）。
- [x] 论文构建：**28 页**（27→28，因成本小节+Appendix C+两条 Limitation）、0 错/0 overfull/0 未定义。备份 `probe-protocol.tex.bak-rev7-20261030`。

### ⏳ 进行中：DINOv2-FT 补 2×2 缺失格 —— **已迁移至 16/A100**（2026-10-31）
- **迁移原因**：35/V100 上 full 需 ~1.5–2 天 + K 收尾 0.5 天；用户指认 16 上 A100-80G×2（GPU1 空闲）可用 → 总 ETA 缩至 **~12–18h**。
- **35 状态**：队列已 kill（pid 2364237）；**K=1 成品已回 143**：`data/dinov2_kshot_ft_k1.json`（40.24%±6.10, n=15，winner lr=2e-5 ep=300，V100/torch2.4 产物，留作跨硬件备份）；partial full checkpoint 存 `/tmp/dinov2_full_partial_35.json`（仅 5e-6 列，不作数）；35 现空闲备用。
- **16 环境**：`user@10.78.65.16`（143 直连可达，35 不通需经 143 中转）；容器 `dinov2-ft`（基于 `mzyc-llm-base:v1`，torch 2.9.1+cu129/py3.12；`--device /dev/nvidia1` 手工直通——该 docker 无 nvidia runtime 别名；`-v /usr/lib/x86_64-linux-gnu:/hostlibs:ro`+`LD_LIBRARY_PATH`；容器内 pip 装了 sklearn 1.9.1/open_clip_torch 3.3.0/scipy）；工作区 `/home/user/probe-protocol`（=容器 `/work`）；权重 SHA256 `d5383ea8…` 已验证一致；**冒烟已通过**（K=1 单 seed 跑通，验证门正常）。
- **队列**：`/work/run_queue_16.sh`（容器内 nohup，pid 1006，10-31 启动）；顺序 `--k 1 → --full → --k 2 → --k 5 → --k 10`（K=1 重跑 = 与其余 job 同硬件同 torch，保同族可比；35 版 K=1 留作跨硬件复现点）；guard：GPU1 空闲显存 <20GB 则睡 300s；失败即停。
- **监控**：`ssh user@10.78.65.16 "tail -20 /home/user/probe-protocol/logs/queue_nohup.log"`（⚠️ **16 时钟慢 1 天**：日志 10-30 14:53 = 实际 10-31）；输出 schema 同前（stage1 9 配置×3 seeds + selected + stage2 15 seeds）。
- [x] **收尾（11-02）**：queue DONE 后 `docker stop dinov2-ft` 释放 GPU1（nvidia-smi 确认 14 MiB 空载；GPU0 为别人的负载）；队列日志已拉回 143 `logs/dinov2_16/queue_nohup.log`。本地化审计：代码（git）+ 5 份结果 JSON（git，含 provenance/权重 SHA256）+ 日志全部在 143；per-seed 模型权重本就不落盘（脚本训完即评只存指标）；16 磁盘工作区保留（~2GB），需现场 rerun 时 `docker start dinov2-ft && docker exec -it dinov2-ft bash` 一条命令恢复环境。
- **provenance 注意**：A100/torch2.9 产物与 35 的 K=1（V100/torch2.4）跨硬件，Appendix A provenance 按 artifact 分别注明硬件/torch。
- [x] **full 出数（11-02 早）**：**95.60% ± 1.41%（n=15，winner lr=5e-6/300ep）**；K=1 40.19±6.29（A100）vs 40.24±6.10（V100）差 0.05pp 跨硬件复现✓；K=2 **57.49±4.78**。full+K1+K2 JSON 已回 143（`dinov2_finetune_full_16.json` / `dinov2_kshot_ft_k2_16.json`，K1 为 35 版 `dinov2_kshot_ft_k1.json`）。
- [x] **full 入论文（4 处，备份 bak-rev9-20261102）**：① Results L313 DINOv2 句重写（frozen 68.1 / single-30ep 77.5 / matched 95.6±1.4，"not an artifact of the specific CLIP-vs-ResNet-18 pairing"）；② Fairness control 加 DINOv2 句（"tying fine-tuned CLIP…contrastive/self-supervised divide"）；③ Limitation (1) → "Three backbone families"（全数据轴三家族均覆盖）；④ Appendix A provenance 加 DINOv2-FT 行（A100/PyTorch 2.9 + V100 K1 跨硬件 0.05pp 互证）。29 页 clean build，pdftotext 6/6 验证。
- [x] **TikZ 确认无需改**：Intro 2×2 图只含 CLIP–ResNet，无 DINOv2 格。
- [x] **K=5/K=10 出数（11-01 23:21 16 时钟，queue DONE，5/5 rc=0，GPU 释放）**：K=5 **75.27±5.72**（winner lr=2e-5/100ep）；K=10 **85.07±3.18**（winner lr=2e-5/30ep）。JSON 已回 143（`_k5_16` / `_k10_16`，partial=None）。
- [x] **Table 2 终版（备份 bak-rev11-20261102）**：新增 `DINOv2 fine-tuned (ViT-L/14)` 行（40.2/57.5/75.3/85.1/95.6，FT 组）；脚注删 "DINOv2 K=2 not executed" + 加 A100/PyTorch2.9 + V100 K1 互证注；Results 加 few-shot 单调句（K1 打平 → K2 +5.0 → K5/10 +10.4/+10.5 vs ResNet，K5/10 追平 CLIP-FT，K1/2 落后）；SD 统一舍入 1.5。29 页 clean build，pdftotext 9/9 验证。
- [x] **git 分组 commit（7 个，working tree 零残留）**：① gitignore（bak/r1/审计工作文件）② manuscript（rev8–11 + DINOv2 矩阵）③ scripts 迁移 ④ data DINOv2+paired ⑤ canonical 对齐 ⑥ dinov2/grid/paired 脚本 ⑦ 根 figures/ 删除。
- [x] **em-dash 复查（B 方案撤销）**：70 个 `---` 中 41 个是表格占位符，行文真实密度 0.24%（每 413 词一个）低于 AI 典型值，29 个全为功能性括号 → 不改。
- [x] **We/Our 统计**：we=38 / our=9 / 12.7k 词 = 0.37%，方法节集中、Results 近零，正常克制区间 → 不改。
- [x] **AI 痕迹扫描**：40+ 指纹词零命中（crucial/pivotal/leverage/delve/moreover/notably/worth-noting… 全 0），mosaic=病毒名合法 → 不改。
- [ ] 若队列失败：看 `logs/queue_nohup.log` traceback；容器内 `cd /work && export LD_LIBRARY_PATH=/hostlibs DINOV2_PTH=/work/checkpoints/dinov2/dinov2_vitl14_pretrain.pth && python3 scripts/dinov2_finetune_kshot.py <job>` 重跑。

### ⏳ 待办（投稿前，按序）
- [ ] **项7 收尾：20 项 one-line checklist**（源自 12 轮评审 verdict `manuscript/paper/REVIEW_2026-10-29/12_twelfth_review_verdict.md` + `SUBMISSION_CHECKLIST.md` 的 "Still TODO"）→ 写成每项一行的状态表，放进 SUBMISSION_CHECKLIST.md 或 REVIEW 目录。
- [ ] **Git commit**（分组）：① 本轮 tex（14 意见+7 项）② 图/脚本（fig_grid_plateau、generate_figures 合并、perclass ×100 修复）③ dinov2_finetune_kshot.py ④ 数据 JSON；注意 `data/canonical_results.json` 是**先前遗留**的未提交修改，单独一组；`*.bak-*` 备份文件加 .gitignore 或不提交。
- [ ] **人工项（只能作者做）**：CeA Guide for Authors 活页确认（引用格式/模板/关键词上限 3–6/摘要字数）；仓库真实 URL（double-blind 前可匿名，投稿前必须插入）；ORCID；Highlights 5 条复核。
- [ ] 最终：latexmk 重建 + `scripts/audit_references.py` + checker + 页数/overfull 终验。

### 关键事实备忘（防遗忘）
- 论文 28 页；float 布局 p3 Fig1(TikZ 2×2) / p9 T1 / p11 T2 / p12 Fig2 / p13 T3 / p14 T4 / p15 Fig3 / p19 Fig4 / p27 Fig5(t-SNE)+Appendix C。
- DFN-CLIP full C=1 canonical = **76.81**；legacy store = 76.09（漂移 0.72pp，已披露）；LAION = 76.09；ResNet-FT = 87.7±2.8；CLIP-FT = 94.7/95.9。
- matched-FT 8 contrasts 全过 Holm（+4.64~+12.17pp）；p=0.002 是 5-fold 配对 t（df=4，descriptive，非 16 家族成员）。
- 143 = lizhuo@192.168.50.143（4070 Ti，sshfs 挂在 /mnt/gpu-4070）；35 = broadsense@192.168.20.35（V100×4，DINOv2 队列已迁走，现空闲备用）；16 = user@10.78.65.16（A100-80G×2，GPU0 他人占用，GPU1 归 `dinov2-ft` 容器；143↔16 直连，35↔16 不互通需经 143 中转；**16 时钟慢 1 天**）。
- 35 上的仓库副本只有 scripts/ + data/（tpdd 52M + 108 个 json + checkpoints/dinov2），**无 .git、无 manuscript**——tex 改动不要指望 35 副本，一切以 143 为准。

---


> **状态 (2026-09-23)**: 大修订已完成并编译通过 —— 新标题/摘要/引言、番茄跨作物复制、
> BioCLIP 2 领域VLM零样本对照、DINOv2 自监督对照、CoOp 完整配方对照、5-fold CV 协议、
> ZPE 加权模板集成（作为 in-sample bound 报告）。排版 overfull 已修复（含 Abbreviations 表
> 218pt 溢出，根因是 \binpenalty 阻止 tabular 前换行，用 \par 解决）。
> 剩余决策点：① "You Are What You Prompt" 已从上版稿中整体移除（正文3处+bibitem），
> 因其"领域prompt池优于通用池"的结论与本文"无可靠提升"的零样本结论存在张力——
> 需确认是保持移除还是重新并置讨论；② LP++ 仍未实测（仅引用）。

### Related Work Updates (Required before submission)

- [x] **引用 SCOLD (2025)**: "[SCOLD: A Vision-Language Foundation Model for Leaf Disease Identification](http://arxiv.org/abs/2505.07019)" （已引用：Intro L66、Related Work L92、Discussion L459、Limitations L517）
    - Paper提出18.6万图文对 + 97概念的叶片病害专用VLM
    - soft-target对比预训练，直接竞品/相关工作
    - Paper2结论"off-the-shelf CLIP zero-shot不行"的直接反向支撑

- [x] **引用 BioCLIP 2 (2025)**: "[BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](http://arxiv.org/abs/2505.23883)" （已引用 + 已实测：作为领域VLM零样本对照，TPDD 最佳 15.2% / TLD 21.1%，见 L216 与 data/domain_vlm_zeroshot.json）
    - 2.14亿生物体图像训练，zero-shot泛化到栖息地分类、性状预测
    - "领域预训练带来涌现能力"的最强正面证据
    - 作为领域预训练VLM谱系的一部分

- [ ] **引用 "You Are What You Prompt" (2026)**: "[You Are What You Prompt: Prompt Quality, Domain Shift, and Uncertainty in Agrifood VLMs](http://arxiv.org/abs/2608.18116)"
    - ⚠️ **当前状态：已从论文中整体移除**（正文3处 + bibitem，上一轮会话的编辑决定）
    - 移除原因推断：其"领域特定prompt池稳定优于通用池"的结论与本文"零样本无可靠提升"结论存在张力
    - **需作者确认**：保持移除，还是在 Discussion 中重新并置（作为"prompt池规模/质量"的领域证据）
    - 农业/食品领域直接证据
    - 领域特定prompt池(51-52条)稳定优于通用池(247-426条)
    - ZPE(按判别信号加权prompt集成)在domain shift下显著提升精度与校准
    - 将Paper2的单数据集观察升级为领域规律

- [x] **引用 LP++ (2024)**: "[LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP](http://arxiv.org/abs/2404.02285)" （已引用：Intro L66、Discussion L459、Limitations L517；未实测，仅作为"已知更强probe"文献引用）
    - 明确指出"LP在few-shot CLIP文献中常被当作弱基线"
    - 提出可学习的图像-文本知识混合线性分类器
    - 显著超越朴素LP

### Experimental Additions (High Priority)

- [x] **补LP++或CoOp对照实验**
    - 回答"CLIP特征弱还是用法弱"的核心问题
    - CoOp 已完成：最小配方(4tok/10ep)与完整配方(16tok/20ep、8tok/15ep)均报告，完整配方 collapse 到接近随机，CoOp 不是救援路径（L145-148、L246、L266；data/coop_*.json）
    - LP++ 未实测（仅引用）——若审稿人追问"CLIP用法弱"，这是仍可选的补实验

- [x] **运行ZPE (Prompt Quality Enhancement)**
    - 已完成：data/zpe_experiment.json（14模板 + 判别信号加权集成，K∈{1,2,5,10,full}、τ扫描）
    - 论文中以"accuracy-weighted mixture（in-sample bound）"形式报告：最佳 14.5%（95% CI 8.7–20.3），不改变"无可靠提升"结论（L202、data/zero_shot_clean.json）

### Manuscript Revision Items

- [x] **Related Work Section**: 补全SCOLD, BioCLIP 2, LP++ 三篇关键论文（"You Are What You Prompt" 已移除，见上方决策点）
    - 正面处理领域预训练VLM作为off-the-shelf CLIP的直接反例方向 ✓（L66、L91-92）
    - 在related work里并置Paper2结论与领域预训练结果 ✓（L66 明确"off-the-shelf"边界 + BioCLIP 2 实测对照 L216）

- [x] **Limitations Section**: 
    - 引用LP++ (2404.02285)，明确朴素probe是已知弱基线 ✓（L459、L517）
    - 承认朴素LP是已知弱基线，防止审稿人解读为"用了CLIP的弱用法" ✓（CoOp 对照 + DINOv2 对照 + frozen-ResNet 对照共同支撑"特征 vs 用法"的区分）

- [x] **Domain Pretraining Discussion**: 
    - 阐明Paper2的定位是"资源受限场景下不做领域预训练的实证边界" ✓（L66、L459）
    - 虽然有SCOLD/BioCLIP 2等领域预训练工作，但Paper2在资源受限情况下的实验更具代表性 ✓（BioCLIP 2 已实测作为对照 L216，结论：领域VLM仍处 near-zero-shot，失败是 CLIP-style 冻结图文决策规则的性质）

- [~] **Prompt Quality Acknowledgment**: 
    - ~~融入"You Are What You Prompt"的发现~~ → 已移除（决策点，见上方）
    - 提到ZPE作为比手工模板更系统的prompt评估方法 ✓（L202，ZPE加权集合作为 in-sample bound 报告）

### Code/Script Updates

- [x] **验证现有脚本覆盖**:
    - [x] `prompt_ablation.py` - 已完成prompt消融实验
    - [x] `coop_experiment.py` + `coop_full.py` - CoOp对照实验已完成并报告
    - [x] `kg_tld_validation.py` - KG TLD验证逻辑检查
    - [x] `text_length_ablation.py` - 文本长度消融已完成
    - [x] `pv_tomato_protocol.py` - 番茄跨作物复制已完成并报告（Abstract/Intro/Results 均含）
    - [x] `domain_vlm_zeroshot.py` - BioCLIP 2 零样本对照已完成并报告
    - [x] `dinov2_protocol.py` + `dinov2_native518.py` - DINOv2 对照已完成并报告

- [~] **确保LP++对照实验代码就绪**:
    - LP++ 未实测（仅引用）；若需补实验，这是性价比最高的下一步
    - 当前以 CoOp（prompt学习）+ DINOv2（表征泛化）+ frozen-ResNet（公平基线）三重对照覆盖"特征 vs 用法"问题

### Documentation

- [x] **完善Abstract**: 明确提及"资源受限场景下CLIP的实证边界" ✓（新摘要：zero-shot fails / regime-dependent / not tobacco-specific 三段式）
- [x] **更新Introduction**: 增加领域预训练VLM的讨论框架 ✓（L66 off-the-shelf 边界定位 + BioCLIP 2 实测对照预告 + RQ3 跨作物）
- [x] **完善结论**: 总结CLIP在农业领域的适用边界与局限 ✓（L517-539：off-the-shelf 边界 + 实践建议，限定到 off-the-shelf CLIP）

### Quick Wins (Can Do Immediately)

- [x] 在 manuscript/related work 中添加关键论文的引用（SCOLD、BioCLIP 2、LP++；You Are What You Prompt 已移除）
- [x] 在 Limitations 中添加LP++引用和probe结果下界说明
- [x] 在 Introduction 中阐明资源受限场景的定位

### Typography (2026-09-23, done)

- [x] 修复 Abbreviations 表 218pt overfull（根因：\binpenalty=10000 阻止 tabular 前行断行；改用 \par 分隔 intro 句与表格）
- [x] 修复 RQ 列表 overfull（enumerate→itemize）
- [x] 修复 tab:cost 表 overfull（加 \small）
- [x] 剩余 overfull 均为 ≤14pt 的正文/参考文献行内溢出，属可接受范围