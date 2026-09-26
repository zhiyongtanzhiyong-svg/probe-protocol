# Paper2: CLIP vs ResNet - Tobacco Dataset
## TODO List

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