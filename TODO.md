# Paper2: CLIP vs ResNet - Tobacco Dataset
## TODO List

### Related Work Updates (Required before submission)

- [ ] **引用 SCOLD (2025)**: "[SCOLD: A Vision-Language Foundation Model for Leaf Disease Identification](http://arxiv.org/abs/2505.07019)"
    - Paper提出18.6万图文对 + 97概念的叶片病害专用VLM
    - soft-target对比预训练，直接竞品/相关工作
    - Paper2结论"off-the-shelf CLIP zero-shot不行"的直接反向支撑

- [ ] **引用 BioCLIP 2 (2025)**: "[BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](http://arxiv.org/abs/2505.23883)"
    - 2.14亿生物体图像训练，zero-shot泛化到栖息地分类、性状预测
    - "领域预训练带来涌现能力"的最强正面证据
    - 作为领域预训练VLM谱系的一部分

- [ ] **引用 "You Are What You Prompt" (2026)**: "[You Are What You Prompt: Prompt Quality, Domain Shift, and Uncertainty in Agrifood VLMs](http://arxiv.org/abs/2608.18116)"
    - 农业/食品领域直接证据
    - 领域特定prompt池(51-52条)稳定优于通用池(247-426条)
    - ZPE(按判别信号加权prompt集成)在domain shift下显著提升精度与校准
    - 将Paper2的单数据集观察升级为领域规律

- [ ] **引用 LP++ (2024)**: "[LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP](http://arxiv.org/abs/2404.02285)"
    - 明确指出"LP在few-shot CLIP文献中常被当作弱基线"
    - 提出可学习的图像-文本知识混合线性分类器
    - 显著超越朴素LP

### Experimental Additions (High Priority)

- [ ] **补LP++或CoOp对照实验**
    - 回答"CLIP特征弱还是用法弱"的核心问题
    - 如果算力允许，跑一个LP++对照——性价比最高的补实验
    - 或运行CoOp实验作为prompt学习对照

- [ ] **运行ZPE (Prompt Quality Enhancement)**
    - 用代替手工14模板的系统化对照方法
    - 基于判别信号加权的prompt集成
    - 作为prompt消融的系统化对照

### Manuscript Revision Items

- [ ] **Related Work Section**: 补全SCOLD, BioCLIP 2, "You Are What You Prompt", LP++四篇关键论文
    - 正面处理领域预训练VLM作为off-the-shelf CLIP的直接反例方向
    - 在related work里并置Paper2结论与领域预训练结果

- [ ] **Limitations Section**: 
    - 引用LP++ (2404.02285)，明确probe数字(28.4%@K=1 vs ResNet 31.4%)是CLIP特征质量的下界
    - 承认朴素LP是已知弱基线，防止审稿人解读为"用了CLIP的弱用法"

- [ ] **Domain Pretraining Discussion**: 
    - 阐明Paper2的定位是"资源受限场景下不做领域预训练的实证边界"
    - 虽然有SCOLD/BioCLIP 2等领域预训练工作，但Paper2在资源受限情况下的实验更具代表性

- [ ] **Prompt Quality Acknowledgment**: 
    - 融入"You Are What You Prompt"的发现：领域特定prompt质量决定专业领域成败
    - 提到ZPE作为比手工模板更系统的prompt评估方法

### Code/Script Updates

- [ ] **验证现有脚本覆盖**:
    - [ ] `prompt_ablation.py` - 已完成prompt消融实验
    - [ ] `coop_experiment.py` - CoOp对照实验已准备就绪
    - [ ] `kg_tld_validation.py` - KG TLD验证逻辑检查
    - [ ] `text_length_ablation.py` - 文本长度消融已完成

- [ ] **确保LP++对照实验代码就绪**:
    - 检查是否需要添加LP++基线对比代码
    - 或准备好引用现有实验结果作为LP++对照

### Documentation

- [ ] **完善Abstract**: 明确提及"资源受限场景下CLIP的实证边界"
- [ ] **更新Introduction**: 增加领域预训练VLM的讨论框架
- [ ] **完善结论**: 总结CLIP在农业领域的适用边界与局限

### Quick Wins (Can Do Immediately)

- [ ] 在 manuscript/related work 中添加四篇关键论文的引用
- [ ] 在 Limitations 中添加LP++引用和probe结果下界说明
- [ ] 在 Introduction 中阐明资源受限场景的定位