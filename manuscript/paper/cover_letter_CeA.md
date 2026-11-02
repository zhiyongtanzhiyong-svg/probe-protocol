# Cover letter — Computers and Electronics in Agriculture

**Date:** 2026-10-14

**To the Editor,**
Computers and Electronics in Agriculture
(Elsevier)

**Manuscript title:** Frozen Probes versus Matched Budgets: Protocol-Dependent Ordering of CLIP and ResNet-18 for Fine-Grained Tobacco Leaf-Disorder Classification

**Authors:** Chenfeng Long (College of Information and Intelligence, Hunan Agricultural University; Yuelushan Laboratory, Changsha 410128, China); Zhiyong Tan (College of Information and Intelligence, Hunan Agricultural University; Yuelushan Laboratory, Changsha 410128, China); Yangjun Deng (College of Information and Intelligence, Hunan Agricultural University; Yuelushan Laboratory, Changsha 410128, China); Xiang Luo (Information Department, China Guangdong Tobacco Meizhou Co., Ltd, Meizhou 514000, China); Lizhuo Zhang (College of Information and Intelligence, Hunan Agricultural University; Yuelushan Laboratory, Changsha 410128, China; corresponding author, zhanglizhuo@hunau.edu.cn)

---

Dear Editor,

We are pleased to submit the above manuscript for consideration as a Research Article in *Computers and Electronics in Agriculture*.

**Why this work matters for your readership.** Whether off-the-shelf vision–language models can classify fine-grained crop diseases without domain-specific training is a practical question for the resource-constrained deployments that *Computers and Electronics in Agriculture* addresses. On a primary 16-class tobacco leaf-disorder dataset (TPDD), with cross-crop checks on tomato and corn (PlantVillage) and a field-condition evaluation, we report findings of direct practical consequence. First, generic CLIP is unreliable zero-shot for this fine-grained task: on the two tobacco sets no prompt template is detectably better than the majority-class baseline, and knowledge-graph-augmented prompts give no reliable gain. Second, the CLIP-versus-ResNet ordering is determined by the adaptation protocol, not the data regime: at the default probe C=1 the apparent few-shot CLIP advantage over the under-tuned CNN is an artifact that disappears under a matched fine-tuning budget, whereas under matched fine-tuning fine-tuned CLIP leads fine-tuned ResNet-18 at every K. Third, at full data fine-tuned ResNet-18 holds a small, sign-stable edge over the validation-regularized frozen CLIP probe under 5-fold cross-validation. Together these give practitioners an empirical boundary for when off-the-shelf CLIP is and is not a reasonable, training-light option for fine-grained leaf-disorder diagnosis.

**Fit and originality.** The work is empirical and methods-comparative, in line with the journal's scope on intelligent and electronic systems for agriculture. Its contribution is a within-crop empirical boundary, tested under seed-paired statistics and multiplicity correction, with a cross-crop replication — rather than a new architecture.

**Declarations.**
- This manuscript is original, has not been published, and is not under consideration by any other journal or conference.
- All authors have read and approved the submission. X. Luo is employed by China Guangdong Tobacco Meizhou Ltd, and part of this work was supported by the Meizhou Tobacco Science Research Project (Grant 202404); the authors declare no other conflicts of interest.
- AI-assisted tools (an LLM coding assistant) were used for drafting and code assistance during preparation; the authors critically reviewed every claim, verified all reported statistics against the underlying result files, and take full responsibility for the content.
- The code, precomputed features, and per-seed result files supporting all findings will be made available in an anonymized repository at the review stage so that the reported statistics can be independently re-verified.

**Suggested reviewers** *(to be completed before submission — names, affiliations, ORCID iDs, and emails; 3–5 recommended):*

1. [TO BE ADDED]
2. [TO BE ADDED]
3. [TO BE ADDED]

Thank you for your consideration. We look forward to your response.

Sincerely,

Lizhuo Zhang (on behalf of all co-authors: Chenfeng Long, Zhiyong Tan, Yangjun Deng, Xiang Luo)
College of Information and Intelligence, Hunan Agricultural University

Corresponding author:
Lizhuo Zhang, College of Information and Intelligence, Hunan Agricultural University, Changsha 410128, China
zhanglizhuo@hunau.edu.cn
