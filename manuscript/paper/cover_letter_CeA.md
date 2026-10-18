# Cover letter — Computers and Electronics in Agriculture

**Date:** 2026-10-14

**To the Editor,**
Computers and Electronics in Agriculture
(Elsevier)

**Manuscript title:** Regime-Dependent Performance of Generic CLIP Representations for Fine-Grained Tobacco Leaf-Disorder Classification

**Author:** Lizhuo Li, School of Information Science and Engineering, Yunnan University, Kunming 650091, China (lizhuo@ynu.edu.cn)

---

Dear Editor,

We are pleased to submit the above manuscript for consideration as a Research Article in *Computers and Electronics in Agriculture*.

**Why this work matters for your readership.** Whether off-the-shelf vision–language models can classify fine-grained crop diseases without domain-specific training is a practical question for the resource-constrained deployments that *Computers and Electronics in Agriculture* addresses. On a primary 16-class tobacco leaf-disorder dataset (TPDD), with a directional cross-crop check on tomato (PlantVillage), we report three findings of direct practical consequence. First, generic CLIP is unreliable zero-shot: no prompt template is detectably better than the majority-class baseline, and knowledge-graph-augmented prompts give no reliable gain. Second, the CLIP-versus-ResNet comparison is regime- and checkpoint-specific rather than systematic — at K=1 the outcome flips between checkpoints (LAION-2B leads, DFN-CLIP trails), partly as a probe-protocol effect, and fine-tuned ResNet-18 leads at K≥5 with a gap that grows significantly with K. Third, at full data fine-tuned ResNet-18 holds a small, sign-stable edge under 5-fold cross-validation, and the frozen-CLIP-probe versus fine-tuned-CNN gap is best read as a usage-mode effect (fine-tuned CLIP itself exceeds ResNet-18 under a matched budget). Together these give practitioners an empirical boundary for when off-the-shelf CLIP is and is not a reasonable, training-light option for fine-grained leaf-disorder diagnosis.

**Fit and originality.** The work is empirical and methods-comparative, in line with the journal's scope on intelligent and electronic systems for agriculture. Its contribution is a within-crop empirical boundary, tested under seed-paired statistics and multiplicity correction, with a cross-crop replication — rather than a new architecture.

**Declarations.**
- This manuscript is original, has not been published, and is not under consideration by any other journal or conference.
- All authors have read and approved the submission. The author declares no conflicts of interest.
- AI-assisted tools (an LLM coding assistant) were used for drafting and code assistance during preparation; the author critically reviewed every claim, verified all reported statistics against the underlying result files, and takes full responsibility for the content.
- The code, precomputed features, and per-seed result files supporting all findings will be made available in an anonymized repository at the review stage so that the reported statistics can be independently re-verified.

**Suggested reviewers** *(to be completed before submission — names, affiliations, ORCID iDs, and emails; 3–5 recommended):*

1. [TO BE ADDED]
2. [TO BE ADDED]
3. [TO BE ADDED]

Thank you for your consideration. We look forward to your response.

Sincerely,

Lizhuo Li
School of Information Science and Engineering, Yunnan University
Kunming 650091, China
lizhuo@ynu.edu.cn
