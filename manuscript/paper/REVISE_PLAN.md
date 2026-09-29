# Revision Plan — CLIP vs. ResNet Tobacco (pre-submission)

Date: 2026-09-09 · Status: **EXECUTED 2026-09-11 (Phase 1) + EXTERNAL REVIEW REVISIONS 2026-09-11 (Phase 2, P0+P1 text fixes done; P2 experiments pending)**
Inputs: 3 LLM reviews (Qwen / GLM / DeepSeek) + prior `KILL_ARGUMENT` (WARN) + `PAPER_CLAIM_AUDIT` (FAIL→WARN).
Method: every item below was checked against `clip_vs_resnet_tobacco.tex` (line refs) and `data/*.json`.

---

## 0. Executive summary
- **Verdict:** The three reviews are *directionally right* on structure but **overstate severity**, duplicate each other, and contain **several factual errors** (listed in §3). None is a real referee; this is an internal audit.
- **End state we can defend:** *"An honest benchmark of one generic VLM (DFN-CLIP) + one strong baseline (ResNet-18) on two small tobacco datasets."* That is publishable at MDPI *Agriculture* **only after** the P0 blockers below are resolved.
- **Not defensible as-is:** anything implying "systematic study of generic VLMs," "competitive," or "KG prompts are the effective method" — all three are overclaims on current evidence.
- **Two landmines that could sink the paper if left** (both need facts only you/the repo can confirm): the **TLD "84 images"** contradiction and the **EN-Label 72.46% / class-imbalance** signal.

---

## 1. P0 — BLOCKERS (resolve before any prose work)

| ID | Issue | Evidence | Why it's blocking | What you must decide |
|----|-------|----------|-------------------|----------------------|
| P0-1 | **TLD "full split = 84 images" contradicts "TLD = 1,392 images"**; full (56.2%) < K=5 (62.4%) with an inverted "underfit" cause | L117 vs **L267**; Table `tab:tld` L255–258 | Internally inconsistent; the "84" is unexplained; the causal story is backwards (underfit improves *with* more data). Reviewers read this as data-leakage or a broken split. | What is TLD's real train/test count and how was it split? Then either fix the number, fix the split, or **drop the TLD full-data row + the "underfit" sentence**. |
| P0-2 | **EN-Label = 72.46%** zero-shot (per-class `anthracnose=100`, rest 0) in `prompt_ablation.json` | `data/prompt_ablation.json` line ~4–23 | If the test split is imbalanced (anthracnose-dominant), the **6.25% random baseline (L282) is wrong** and a **majority-class baseline (~72%) beats the paper's "best 15.2%"** → RQ1's core claim inverts. Several templates degenerate to a single class = degenerate predictors. | Report the **per-class test-set counts**. If imbalanced: remove/replace the 6.25% baseline, add a majority-class baseline, and reframe RQ1 ("marginal over random, below a trivial majority baseline"). |
| P0-3 | **ResNet-18 TLD column is "from prior work," uncited** | L249 caption; L256–258 (44.2/85.0/89.8) | RQ3's "consistent rankings" rests on half of an unverifiable table. | Either (a) cite + protocol the source and re-run under our seeds, or (b) **remove the ResNet TLD column** and make RQ3 a CLIP-only directional check. |

> P0-1 + P0-3 together: if both stand, **RQ3 (cross-dataset) should be demoted or cut**, not defended. See §5.

---

## 2. P1 — CONCEDE + REFRAME (text-only, no new experiments)
These are the cheap, high-value fixes that remove most of the "overclaim" smell.

| # | Fix | Location |
|---|-----|----------|
| P1-1 | **Title:** drop question form → declarative, and drop "Systematic"/"generic VLMs" plural (single VLM). e.g. *"Evaluating a Generic Vision–Language Model for Fine-Grained Tobacco Disease Classification: A CLIP Case Study"* | title |
| P1-2 | **Abstract:** stop number-dumping; lead with the 2–3 takeaways + the honest boundary. Keep at most 2–3 figures. | L42–48 |
| P1-3 | **"competitive" → bounded claim** everywhere: "within 3 pp at K=1 (not shown equivalent)" / "within 10.7 pp at full data." Replace the 4× "strong baseline/competitive" framing. | L47, L82, L230, L318, L362, L381, L384, L386 |
| P1-4 | **K=1 equivalence:** p=0.12 ≠ "same band." Either run **TOST** (equivalence, ±5 pp) or state "no significant difference; not demonstrated equivalent; d=0.6 (moderate)." Do not assert equivalence. | L230, L243 |
| P1-5 | **CoOp causal claim:** L241 "gap is primarily representational" is not supported by a control that scored *below zero-shot* (10.8 < 15.2) at K=1 with batch 32 > 16 support imgs. Downgrade to: "a learnable text prompt did not close the gap; we cannot attribute this to representation vs. optimization; stronger adapters not tested." Fix "evaluated zero-shot" → "evaluated on the test set after prompt training." | L142, L239–243, L275 |
| P1-6 | **KG confound:** 7/16 classes use generic fallback → 15.2% is not clean "KG knowledge." Reframe RQ1 to "KG-augmented templates (9/16 with KG, 7/16 fallback) scored 1.4 pp above the best non-KG template; attribution to KG specifically is limited by fallback coverage." Ideally report **9-KG-covered-classes-only** accuracy. | L131, L205–207, L324–326, L370 |
| P1-7 | **Calibration reframe:** ECE 0.46 (full) and 0.14 (zero-shot, 141 imgs, no CI) → state "not deploy-calibrated; zero-shot confidence barely separates correct from incorrect (0.32 vs 0.29)." Drop "rank-relevant / reliable / typical of L2 LR" (no citation). | L307–312 |
| P1-8 | **Cost narrative:** table already reports size/speed (good) — but L330–333 leads with "CPU LR" and underweights 1.7 GB / 4× slower. Add one balanced sentence: CLIP is cheaper to *train* but heavier/slower to *serve*. | L330–333 |
| P1-9 | **"21× spread"** on a 0.7% (1-image) cell → use a range, not a ratio. | abstract/discussion |
| P1-10 | **Inverse ImageNet normalization (L124):** clarify *why* (dataset stored pre-normalized) and *where*; don't delete, but make it reproducible. | L124 |
| P1-11 | **Scope honesty:** single VLM + single scale. If we don't add a 2nd VLM (P2-1), explicitly say "one VLM; results may not generalize to other VLMs." | L42, discussion |

---

## 3. P2 — ADD EVIDENCE (moderate effort, highest credibility gain — pick 2–3)
| # | Item | Effort | Value |
|---|------|--------|-------|
| P2-1 | Add **1 more generic VLM** (SigLIP or EVA-CLIP) in zero-shot + probe, **or** drop "generic VLMs" plural. | M (1 GPU-hr-ish) | Directly answers the #1 "systematic" critique. |
| P2-2 | Add **BioCLIP/AgroCLIP off-the-shelf baseline** (they're downloadable; "resource-constrained" doesn't exclude them), **or** tighten the "pretraining infeasible" framing. | M | Removes the "you ignored the obvious domain model" hit. |
| P2-3 | **Paired/bootstrap stats** for the 14-template comparison (McNemar + bootstrap CI on the 1.4 pp) + **TOST** for K=1. | S (analysis only) | Turns "not significant" into a defensible statement. |
| P2-4 | **Per-class attribution on the 9 KG-covered classes** (vs. 7 fallback). | S | Cleans the KG claim (P1-6). |
| P2-5 | **Reliability diagram** + ECE bin count for the calibration section. | S | Fixes P1-7 with a figure. |
| P2-6 | **Report class distribution** (all 16/10) + majority baseline. | S | Required by P0-2 regardless. |

---

## 4. P3 — HYGIENE (submission compliance)
- **MDPI boilerplate:** fill `\simplesumm{}`, `\datereceived{}`, add Author Contributions / Funding / Data Availability / Conflict of Interest / Abbreviations.
- **References:** fix `et al.` in entries, add DOIs/volume/pages, replace `Joker5800` (HuggingFace personal) with a proper citation + license + provenance, verify `tpdd2024`.
- **Figures:** ensure `fig_perclass.pdf` + `fig_confusion.pdf` exist and are cited (currently unverifiable).
- **Class lists:** list all 16 TPDD / 10 TLD classes + per-class counts (also satisfies P0-2/P2-6).
- **Reproducibility:** full prompt strings (appendix), seed list or fixed seeds, LR `C=1.0` justification, ResNet augmentation + early-stopping spec, data-split script, code repo URL.

---

## 5. Go / no-go on RQ3 (cross-dataset)
- **Recommendation: DEMOTE, don't defend.** Given P0-1 (84-image contradiction) + P0-3 (uncited ResNet column), the quantitative "consistent rankings" claim is not currently supportable.
- **Keep the genuine negative result:** "KG prompts do **not** transfer to TLD (10.8% vs 17.7%)" (L270–273) is a *real, defensible* finding — promote it to a discussion point.
- If you fix P0-1 and P0-3, RQ3 can stay as a *directional* check with the ResNet column either re-run under our protocol or clearly labeled as a literature reference.

---

## 6. Do NOT change (reviewer errors — verified)
- **Cohen's d values are correct.** Recomputed from `data/`: K=1 Δ0.030 / pooled SD 0.052 → d≈0.6 ✓; K=5 Δ0.151 / SD~0.05 → d≈3.0 ✓. DeepSeek's "calculation error" is not supported.
- **"Inverse normalization is amateur code" (Qwen)** — overreach; it's a clarification item (P1-10), not a defect.
- **"Train ViT-L/14 from scratch" baseline demand** — infeasible overreach for this scope; a stated limitation suffices.
- **Data artifact to reconcile (not a reviewer point, ours):** `coop_experiment.json` full `probe_clean` = 0.7608 with **std = 2.3e-16 (zero variance)**, vs `clip_fewshot.json` full = 0.7502±0.013. **RESOLVED (2026-09-11):** `probe_clean` / `clip_features_clean.npz` (deterministic clean features + deterministic LR) is the *correct* protocol → 76.1% (quad-corroborated by `dfn_clip_full`, `linear_C1`, E4 `full`); `clip_fewshot.json` full 75.0±1.3 was per-seed re-encode augmentation noise. Manuscript now reports **76.1%** everywhere.

---

## 7. Suggested order of work
1. **Answer the 3 P0 facts** (TLD split, TPDD test-set class counts, TLD ResNet source). ← *you*
2. P0 fixes in `.tex` + data (may cut RQ3 rows). 
3. P1 text passes (one sweep, then recompile).
4. Choose 2–3 of P2 (suggest **P2-1 + P2-3 + P2-6** as the best effort/impact set).
5. P3 hygiene + recompile + re-run `paper-claim-audit` to confirm FAIL→clean.

---

## 8. Open questions (only you can answer)
1. **TLD:** real total images, real train/test counts, and how was it split? Is "84" a typo?
2. **TPDD test split:** per-class counts? Is `anthracnose` ≈ 70%+ of the test set? (Decides P0-2.)
3. **TLD ResNet column:** which paper/protocol, or should it be dropped?
4. **Scope:** willing to add a 2nd VLM and a domain-VLM baseline (P2-1/P2-2), or keep single-model and reframe the title/scope (P1-1/P1-11)?

---

## 9. Revision log (2026-09-11)

**P0 (all resolved — no new edits needed):**
- **P0-1 (TLD "84 images"):** already fixed in the manuscript before this session — "84 images" absent from current `.tex`; L120/L246 "232 distinct base test images" verified correct against `tld_test_manifest.json` (232 base + aug variants = 1,392 files); TLD test-only split documented at L269-270.
- **P0-2 (EN-Label "72.46%"):** **FALSE ALARM — units misread.** `prompt_ablation.json` stores accuracy in percent units: EN-Label = 0.7246, i.e. **0.72%**, not 72.46%. Authoritative `zero_shot_clean.json` (source of Table `tab:prompts`) gives EN-Label = 1/138 = 0.72% and carries an explicit `reconciliation` block (old KG-Symptom 15.2% run was a single-run artifact; "CI overlap explains prior best-template disagreement"). Paper L118 describes the true split: 138 test images, four rare classes (anthracnose, black shank, genetic abnormality, TSWV) with 1 image each, wildfire 18 (13.0%) = majority. **Paper correct as written; no change.** User retracted the "72.5% anthracnose" answer (which was based on the misread framing).
- **P0-3 (ResNet TLD column):** already fixed — column removed; L269-270 explicitly documents the removal ("drawn from an unverifiable prior split and are removed rather than cited second-hand"); RQ3 demoted at L272.

**P1 (executed 2026-09-11):**
- P1-1 title → `Evaluating a Generic Vision-Language Model for Fine-Grained Tobacco Disease Classification: A CLIP Case Study` (declarative, single-VLM scope per user decision).
- P1-3 bounded: L240 "competitive at K=5" → "trails by 4.3 pp (59.1% vs 63.4%; DFN variant trails by 15.1 pp)"; L331 "strong, training-light foundation rather than a weak one" → "competitive, training-light baseline at low data (K=1: 28.4-38.5% vs 31.4%)"; L374 item heading "strong low-data baseline" → "low-data baseline" + "DFN variant trails by 3.0 pp, not statistically distinguishable".
- P1-5: L143 "evaluated zero-shot on the test set" → "evaluated on the held-out test set (the trained text prompt alone selects among the class labels; no learned classifier)".
- P1-7: calibration paragraph — dropped "reliable" → "informative"; dropped uncited "typical of L2-regularized logistic regression" → "In neither regime are the absolute confidence values calibrated for deployment: they should not be used as decision thresholds without recalibration."
- P1-10: L125 inverse normalization now states the reason ("because the TPDD release ships pixels that are already ImageNet-normalized").
- P1-4/P1-6/P1-8/P1-9/P1-11: verified already handled in current text (L229 honest p-value wording; L132/L203 KG fallback caveat; L368 + `tab:cost` balanced cost narrative; "21x" absent from current text; Limitation (1) states single-VLM-family scope).

**P2-3 (TOST, executed 2026-09-11):** paired per-seed analysis (n=15, `clip_fewshot.json` K=1 vs `resnet_k1_n15.json`): mean diff (DFN-ResNet) = -3.04 pp, 90% CI [-6.01, -0.08] pp. **Equivalence within ±5 pp NOT established** (lower bound exceeds margin). Paper's existing "not statistically significant (Welch p=0.12; paired p=0.09)" retained as the correct statement — no equivalence claim added. (Cross-check: Welch t=-1.60, df=27.7; paired t=-1.81, df=14 — matches paper.)

**P3 (executed 2026-09-11):**
- MDPI boilerplate added before References: `\authorcontributions`, `\funding` ("no external funding" — **verify with author**), `\dataavailability` (repo URL placeholder — **insert real URL**), `\conflictsofinterest`, Abbreviations table; `\simplesumm{}` filled.
- `\bibitem{tld}` (Joker5800) now includes the verified dataset URL (`https://huggingface.co/datasets/Joker5800/Tobacco_Leaf_Diseases`, confirmed in `scripts/download_tld.py`).
- 9 references verified via Crossref/OpenAlex exact-title matches and updated with full author lists + DOIs: `he2016deep`, `huang2021gloria`, `liu2024remoteclip`, `stevens2024bioclip`, `zhai2023sigmoid`, `cherti2023reproducible`, `scold2025` (now Expert Syst. Appl. 2026), `bioclip2` (now NeurIPS 2025), `lpplus` (now CVPR 2024).
- **11 references resolved (2026-09-11, session 2):** 7 already had full author lists + DOIs (spot-checked 3 via Crossref DOI resolution, all correct): `tpdd2024`, `lu2024agroclip`, `asif2024multimodal`, `ge2024fine`, `cropdpkg2025`, `argueso2019fewshot`, `li2021fewshot`. 4 were incomplete ("et al.", no DOI) — now verified and updated: `zhou2022learning` (Zhou/Yang/Loy/Liu, IJCV 2022, 10.1007/s11263-022-01653-1), `fang2023data` (Fang/Jose/Jain/Schmidt/Toshev/Shankar, arXiv:2309.17425), `sun2023eva` (Sun/Fang/Wu/Wang/Cao, arXiv:2303.15389), `youareprompt` (Morales-Garzón/López-Joya/López-Pérez/Martin-Bautista, arXiv:2608.18116).

**Recompiled 2026-09-11:** `latexmk -pdf` clean, 12 pages, new title verified in PDF.

---

## 10. External Review Revisions (2026-09-11, Phase 2)

**Context:** 4 external reviewer reports (DeepSeek, Qwen, GLM, Kimi) all recommend Reject or Major Revision. Consensus issues extracted into P0(5 critical text errors) / P1(7 strengthen caveats) / P2(9 new experiments). User directive: "认真对待，我们是做真研究！！！！" Approved A+B execution.

**P0 (all 5 executed 2026-09-11):**
- **P0-1 (kNN "ceiling"):** L303 "tops out at 72.5%, which is the practical ceiling" → "reaches 72.5%, confirming that the classes overlap substantially in feature space even for the simplest decision rule".
- **P0-2+3 (CI overlap factual error + statistical logic):** L199-201 "all 14 CIs overlap" → "top templates' CIs all include the 13.0% baseline; 4 lowest templates' CIs exclude it". Added: "overlapping CIs do not establish equivalence; McNemar would be required; FWER not controlled". L336 (KG discussion) synchronized.
- **P0-4 ("earlier drafts"):** L271-272 "figures that appeared in earlier drafts were drawn from an unverifiable prior split and are removed rather than cited second-hand" → deleted; replaced with clean statement: "the TLD release provides only a test split, so a probe/ResNet-18 comparison under our protocol is not possible".
- **P0-5 (edge deployment framing):** L343-346 rewritten: "In a typical deployment workflow, both models are trained once and deployed to edge devices. ResNet-18 is ~33x faster at inference and uses ~3x less memory, making it the more practical choice for real-time field deployment. CLIP's advantage is confined to the training phase." L369 (last sentence) synchronized.
- **P0-1b (428M vs 12M):** Already applied in Phase 1 (L237-238, L305).

**P1 (all 7 executed 2026-09-11):**
- **P1-6 (KG confounding):** L203 expanded: "7 of 16 classes lack KG descriptions and use generic fallback text, so the KG templates are a mixture of KG and non-KG prompts; a clean KG comparison would require complete KG coverage or a paired evaluation on the 9 classes with KG descriptions only."
- **P1-7 (tau=100):** L126 added: "a fixed value; the model's native learned logit_scale was not used, and a sensitivity analysis with the native scale is left to future work."
- **P1-8 (83.3% no error bar):** L240 added: "The 83.3% and 81.9% figures are single-run point estimates without seed-level variance; a multi-seed evaluation of the tuned head would be needed to quantify the uncertainty of the 2-4 pp residual."
- **P1-9 (multiple comparisons):** L199-201 (P0-2+3 fix) and L153 (Evaluation section) both note FWER is not controlled.
- **P1-10 (balanced accuracy):** L153 added: "on the imbalanced TPDD test set, top-1 accuracy is dominated by the majority class; balanced accuracy or macro-F1 would provide a more even per-class view, and we leave this to future work."
- **P1-11 (PlantVillage + FGVR refs):** L90 added ImageNet + FGVC context; L106 added PlantVillage citation; 3 new bibitems added (`plantvillage2016`, `deng2009imagenet`, `mountzias2017fgvc`).
- **P1-12 (CoOp protocol):** L143 added: "4 tokens and 10 epochs is a minimal CoOp configuration; the original CoOp paper uses 16 tokens and up to 20 epochs, so a more thorough CoOp evaluation would be needed to rule out under-training as the cause of the poor CoOp results."

**Recompiled 2026-09-11 (Phase 2):** `pdflatex` clean, 13 pages, no errors.

**P2 (executed 2026-09-11):**
- **P2-1 McNemar (K=1):** n_disc=56, n_clip_only=21, n_resnet_only=35, p=0.081 (not significant at 0.05). Supports paper's "not statistically significant" claim. Paired bootstrap: diff=-10.15pp, 95%CI=[-20.3, +0.7] includes zero.
- **P2-1 McNemar (full data):** n_disc=28, n_clip_only=5, n_resnet_only=23, p=0.0009 (highly significant). ResNet wins at full data.
- **P2-2 Balanced accuracy:** CLIP K=1: 33.0% vs ResNet K=1: 41.2%; CLIP full: 67.5% vs ResNet full: 84.3%. Macro-F1: CLIP full 62.8% vs ResNet full 83.7%. ResNet's per-class balance advantage is larger than top-1 accuracy suggests.
- **P2-3 C-grid:** 9 values {0.01,...,100}. Best C=30.0 → 83.3%. Accuracy flat across C∈[0.1,30] (81-83%), not an artifact of extreme hyperparameter.
- **P2-5 t-SNE:** `fig_feature_space.pdf` generated. Shows 16 overlapping clusters, confirming feature-space overlap. Added to §4.4 (Why section) as Figure `fig:tsne`.
- **P2-6 τ sensitivity:** Accuracy completely flat across τ∈{10,...,1000}. τ only affects confidence (ECE), not classification. Paper updated: "τ affects only confidence values, not predicted class or accuracy."
- **P2-4 DINOv2:** Skipped (requires Python 3.10+, environment has 3.8).
- **P2-7/8/9 (reliability diagram, ResNet-50, domain VLM):** Pending (low priority).

**Recompiled 2026-09-11 (Phase 2 + P2):** `pdflatex` clean, 13 pages, t-SNE figure included.

---

## 11. Phase 3 — Numeric Consistency (the #1 reviewer blocker) — RESOLVED 2026-09-15

**Context:** 4 AI reviewers (chatgpt/deepseek/qwen/glm) flagged the full-data CLIP numbers as the top issue. Root cause (traced to source): the paper's full-data CLIP figures were a **"chimera"** stitched from ≥3 different feature-extraction protocols (archived `why_vlms`/`full_provenance` 76.1%, augmented `r4b` 72.5%/82.6%, clean `r4` 84.1%), and — critically — even the "clean" `r4b` protocol extracted its **training features with `TRAIN_TRANSFORM` (random flip/rotate/jitter/crop)**, making the frozen features **non-deterministic across runs** (observed 3–4pp run-to-run swings). This also made the paper's "CLIP side is deterministic (no seed-level variance)" claim false.

**Fix (this session):**
1. `scripts/canonical_protocol.py` now extracts **deterministic clean features** (EVAL_TRANSFORM: resize + center-crop + normalize, no augmentation) for **both** checkpoints (DFN-CLIP primary + LAION-2B), runs C=1 / validation-selected best-C / MLP / kNN(1), both-split geometry, and TOST (training-run + per-image) in one run → single source of truth `data/canonical_results.json`.
2. **Reproducibility verified:** two independent clean runs produced **byte-identical** features and test accuracy.
3. **Canonical numbers locked in** (DFN-CLIP primary): C=1 **76.8%**; validation-selected **C=300 → 85.5%**; MLP **82.6%**; kNN(1) **71.0%**; frozen ResNet linear **73.2%** / MLP **72.5%**; fine-tuned ResNet **86.4% ± 1.9%** (n=15). LAION-2B: C=1 76.8%, best-C 86.2%, MLP 87.0%. Geometry now reported on **both** splits (CLIP 0.62 train / 0.90 test; ResNet 0.56 / 0.88) — drops the earlier train-only cherry-pick.

**Consequences (all reflected in the manuscript):**
- The **non-inferiority claim is now robustly supported** on top-1 accuracy (best-C 85.5% vs 86.4% = 0.9pp; training-run 95% CI [−0.1, +2.0]pp within ±5pp), instead of the fragile 4–5pp borderline it was on the augmented numbers.
- **Balanced-accuracy gap stays in ResNet's favor** and is the honest core caveat: 84.3% (fine-tuned ResNet) vs 77.4% (CLIP best-C) / 68.3% (CLIP C=1), concentrated on the 4 single-image rare classes.
- Every stale number replaced: 76.1→**76.8**, 68.1→**73.2**, 82.6(best-C)→**85.5**, 79.0(MLP)→**82.6**, 74.6(RN-MLP)→**72.5**, 72.5(kNN)→**71.0**; gaps 8.0pp→**3.6pp** (CLIP vs frozen ResNet) and 3.8pp→**≈0.9pp** (residual to fine-tuned); balanced 67.5→**77.4**, macro-F1 62.8→**77.2**.
- **Appendix A** reworked from "protocol-sensitivity (two protocols)" to a **single canonical protocol + reproducibility** statement (the two-protocol distinction is now moot).
- **Methods (L141)** now states clean/deterministic extraction explicitly; the "deterministic" claim is now true.

**Recompiled 2026-09-15:** `latexmk -pdf` clean, **17 pages**, no errors. All canonical numbers cross-checked against `data/canonical_results.json`.

**Still open (not part of the numeric-consistency blocker):** P2-7/8/9 (reliability diagram, ResNet-50, domain VLM baseline) and BioCLIP2/SCOLD/LP++ direct comparisons — text-only scope this round; named in Limitations.

## 12. Phase 4 — Sixth-round Reviewer Triage (deepseek/glm/qwen) — RESOLVED 2026-09-16

**Context:** All 3 sixth-round reviewers rated the Phase-3 fixes excellent; consensus verdict ≈ **Minor Revision / "next round Accept."** Key realization: the *detailed* sections (5.2, 5.5, Methods L158, geometry L372, Appendix) already carried the corrected narrative — the real gaps were **stale headlines**, one **dead geometry number (`0.65`)**, and one **implementation concern (inverse-normalization, raised 5×)**.

**Actions (this session):**
1. **Headline alignment (parity-first), all 7 locations:** abstract (trimmed to **199 words**; `0.65`→`0.62/0.90`; geometry reframed to *text-space decision boundary*), intro findings 2&3, Fig-1 caption, RQ2 narrative + answer (default probe **76.8** vs fine-tuned 86.4), Practical #4 (now framed on **balanced accuracy + cost**, not top-1), Conclusion (3) (regime-dependent + parity + balanced-acc edge).
2. **Blast-radius (glm#2) — resolved by code audit:** K-shot *support* = `TRAIN_TRANSFORM` (intentional few-shot augmentation) / *test* = `EVAL_TRANSFORM` (clean); CoOp = `EVAL_TRANSFORM` on both (clean); full-data canonical path fixed. Noted in Appendix (L492).
3. **Inverse-normalization (raised 5×) — verified NON-ISSUE:** the `EVAL_TRANSFORM` tensor is **already ImageNet-normalized** (measured range `[-1.60, 2.64]`), so the inverse-norm is the **correct denormalization** back to `[0,1]` before OpenCLIP's own re-normalization (only cost = uint8 rounding). Sensitivity test (`scripts/inverse_norm_sensitivity.py` → `data/inverse_norm_sensitivity.json`): canonical vs clean path ⇒ **cos(A,B)=0.90**; C=1 probe **76.8% (canonical) vs 74.6% (clean)** (−2.2pp); zero-shot best-template **<1pp**. Reported result is robust to—and marginally favored by—the canonical path. Documented in **Methods (L124)**.
4. **Citations verified (web/DOI):** SCOLD = Nguyen Quoc, Le Thi Thu, Quach — *Expert Syst. Appl.* 2026, `10.1016/j.eswa.2025.130084` ✓; BioCLIP 2 = Gu, Stevens, Campolongo, … Su — *NeurIPS 38* 2025, 113880–113913, `10.52202/085713-3436` ✓. Both real and correctly attributed.
5. **Residuals:** Practical #4 bracket clean; Table-4 measurement date (2026-09-10) retained as a legitimate provenance note; abstract = **199 words** (≤200).

**Recompiled 2026-09-16:** `pdflatex` clean, **18 pages**, no errors, **no undefined citations/references**. No stale `0.65` or dead "ResNet leads at full data" claims remain (grep-verified).

**Still open (low priority, text-only scope):** direct empirical BioCLIP2/SCOLD/LP++ baselines — named in Limitations, not required for acceptance.

## 13. Phase 5 — Seventh-round / final reviewer pass (glm 6th + deepseek 7th + qwen final) — 2026-09-16

**Verdict from all three:** Minor Revision / "next round Accept." All three say **do not add experiments**; the remaining work is text-only + 1 user action.

**Result-to-claim gate (verified against `data/canonical_results.json`):** the four "hard gates" the reviewers flagged are **already satisfied by the canonical data + existing tables** — no new analysis needed:
- **C=300 is validation-selected, not test-set selection:** val curve C=100/300/1000/3000 = 89.96/90.35/89.96/90.35; argmax C=300 (tied C=3000), interior plateau. Full 10-point curve already in Table 3 (`tab:cgrid`).
- **Head-matched ResNet C-grid:** ResNet-frozen val peak C=10 (80.69%, test 76.09%); both backbones already head-matched in Table 3.
- **Balanced-acc arithmetic + 16-class table:** Table 4 (`tab:perclass`) has all 16 classes; gap decomposes to class 13 (singleton) + class 10 (n=14), offset by CLIP-higher 7/8/9/11/12; "not reducible to singletons" stated.
- **Geometry both splits + Fisher + silhouette + kNN:** §4.5 reports CLIP 0.62/0.90, ResNet 0.56/0.88; Fisher 0.38/0.53 & 0.30/0.44; kNN 71.0%.

**Text edits applied this round (all data-verified, in-place):**
1. **Abstract — neutralize "non-inferior"** (deepseek 必改1): now "within … a pre-specified 5 pp margin" (kept only in RQ2/Methods/Conclusions, each with a top-1 qualifier).
2. **Abstract — TLD is zero-shot-only** (deepseek 必改3): "primary dataset (TPDD) across zero-shot/few-shot/full-data; replicate the zero-shot result on an external dataset (TLD)."
3. **Sharpest claim made explicit** (glm): abstract + already in §4.2 (L308): top-1 passes the 5 pp margin (0.9 pp < 5 pp) but **balanced accuracy fails the same margin (6.9 pp > 5 pp)**; overall accuracy masks the minority-class deficit.
4. **TOST δ=5pp justification** (deepseek 必改2, L158): conservative relative to the 0.9 pp gap; not met on balanced accuracy (6.9 pp); hence restricted to top-1.
5. **Table 2 caption** (deepseek 建议1 / qwen 1): Full column = default C=1; DFN-CLIP 85.5% / LAION-2B 86.2% at validation-selected C (Table 3).
6. **Table 4 note column** (deepseek 建议2): now a consistent per-class "higher" indicator (CLIP / ResNet / $=$) for all 16 classes, with the convention stated in the caption.
7. **§4.5 Fisher test>train** (deepseek 建议4): added that CLIP test-split Fisher (0.53) > train (0.38), reinforcing text-space (not feature) misalignment.
8. **Limitations (1)** (deepseek 建议5): now names BioCLIP 2, SCOLD (and LP++) as not evaluated.
9. **Conclusions (3)** (qwen 3): "so" → "making".

**Recompiled 2026-09-16:** `pdflatex` clean, **18 pages**, no errors; abstract exactly **200 words**; `non-inferior` absent from the abstract (present only in qualified body locations).

**Needs user (cannot fabricate — academic-baseline #1/#2):**
- **Data availability (deepseek 必改4):** `\dataavailability` still reads "URL will be inserted upon acceptance." Need a **real anonymous repo / Zenodo DOI** (upload `scripts/canonical_protocol.py` + `data/canonical_results.json`) to insert the URL. Flagged, not fabricated.
- **Response letter (glm 6):** old→new→reason traceability table for the reviewers — not yet generated.

**Decision (2026-09-16):** user deferred **both** — data-availability URL stays as-is (real anonymous repo / Zenodo DOI to be inserted at submission), and the response letter is not generated this round. All *text* reviewer items are otherwise complete and the manuscript compiles clean (18 pp, abstract 200 words).

**Minor, verified-OK (no change needed):** n_train=1295 matches `canonical_results.json` (the 82/9/9 in the reviewer's note is an approximation; the stratified split yields 1295); `90.4` already `\textbf`; Table 4 already lists all 16 classes; `sun2023eva` already removed.

---

## 14. CEA "苛刻审稿人" 复核 + geometry 代码 bug (2026-09-16)

**触发:** chatgpt 按"CEA 编辑 + 极苛刻审稿人"标准复审 + glm 逐条核实（发现 1 个代码 bug）。结论: Major revision, 但属"统计与论证层 final polishing", 不需加模型。

**代码 bug (glm 核实, 已验证+修复):** `compute_geometry` 的 intra-class 距离用 `D.mean()` 把 n 个对角线 0 也算进均值 → intra 低估 (n-1)/n 倍 → **inter/intra ratio 被高估**。inter 已正确排除对角线; **Fisher(scatter) 与 silhouette(sklearn) 不受影响**。已在 `canonical_protocol.py` / `geometry_recompute.py` / `r4_experiments.py` 三处改为 `D[~np.eye(n)].mean()`。

**重算 (scripts/geom_fix.py, 同一特征提取路径, GPU):** ratio OLD→NEW
- CLIP: 0.62→**0.62**(0.616) / 0.90→**0.81**(0.815)
- ResNet: 0.56→**0.55**(0.549) / 0.88→**0.79**(0.793)
- train 几乎不变, test 降 ~1pp (小类/单样本类对角线影响大); CLIP>ResNet 序与"broadly similar"结论不变。`canonical_results.json` 的 `geometry` 子块已用新值覆盖(备份 `.bak`); `data/geometry_fixed.json` 留档。

**文本修改 (20 处, 全部应用, 编译通过):**
1. **C-grid (Methods L141):** 8→10 值, 并诚实说明 1000/3000 是"验证 argmax 在边界时加的两值以确认 plateau", test 从不参与 C 选择。
2. **K=1 非配对 (Abstract/Intro/Results/RQ2/Discussion/Practical/Conclusion):** "leads/exceeds" → "attains a higher mean K=1 accuracy … independently sampled support sets / unpaired, protocol-level (非纯模型效应)"; p=0.0003 保留在正文。
3. **text-space 过强 (Abstract/Intro/§4.5):** "reflects a misalignment … rather than poor feature geometry" → "consistent with a mismatch between the zero-shot text–image decision rule and the discriminative information …; measured global geometry metrics do not indicate poor separability alone explains it"。
4. **head capacity (§4.5):** "head capacity is not the lever" → "under the tested heads, adding nonlinear capacity did not outperform regularization tuning"。
5. **TLD (RQ3 标题 + Answer):** "External Zero-Shot Replication" → "**Consistency Check**"; "directional external consistency, not an independent replication"。
6. **"decisive" → "substantial" (4 处) + singleton 敏感说明** (balanced-acc 对 4 个单样本类敏感, 1 个单样本类 = 6.25pp)。
7. **residual gap (§4.5 L254):** "a representation/fine-tuning difference, not an artifact of the probe head" → "persists under the tested probe configurations; whether stronger CLIP-side adaptation (adapters/LoRA, not tested) can close it remains unresolved"。
8. **Deployment 表 (tab:cost):** 拆成 Backbone training / Feature extraction / Head training 三行, 不再把 CLIP feature extraction 藏进"training time"。
9. **TLD 17.7 vs 19.0:** 标注 "descriptively below … not a formal failure test"。
10. **Title:** "Evaluating" → "**Diagnosing**" (CEA 语境更强; 可一键回退)。
11. **geometry 数值 + 定义:** test ratio 0.90→0.81 / 0.88→0.79, ResNet-train 0.56→0.55; §4.5 新增 inter/intra ratio 的严格定义(centroid 均值 / 类内 off-diagonal 均值, 跳过单样本类); silhouette ResNet-train +0.03→+0.02(与重算一致)。

**编译:** `pdflatex` clean, **19 页**, 无 error; abstract **146 词**; cost 表 overfull 75pt→32pt(余为原"Inference speed"长标签)。

**验证 (grep):** `decisive`/`head capacity is not the lever`/`reflects a misalignment`/`exceeds ResNet-18 at K=1` 均已清除; `Replication` 仅存于合法语境; geometry 无残留 0.90; 0.81/0.79 已就位。

**未处理 (user 决定 / 用户身份):**
- **作者信息** (Lizhuo Li / Yunnan University / lizhuo@ynu.edu.cn): 投稿前须与投稿系统一致 — 属用户身份, 未改动。
- **Data availability**: 仍为"URL upon acceptance"(上轮已 defer); 建议投稿时给 anonymous repo(Zenodo)。
- **Response letter**: 仍未生成(上轮 defer)。

**结论:** P0(4 项, 含 geometry bug) + P1(4 项) 全部处理; 论文现处"可投 CEA"的论证/统计层面。剩余均为 user 身份/投稿动作项。

---

## 15. 第 9 轮: chatgpt"苛刻 CEA 审稿人" + glm 复核 (2026-09-17)

**触发:** 又一轮最严苛的"统计/论证层"审查, 合并 chatgpt + glm。核心: 论文"解释走得比证据快半步", 需把 claim 降到 evidence 支持的位置 + 补 3 个无需新数据的分析。

**先审计(坐实 reviewer 的"critical"指控):**
- **A3 ensemble 泄漏?** 查 `zpe_experiment.py`: 权重来自 **validation / support-K / static**("T is swept on the signal source (validation)… Final evaluation always uses the held-out test split") → **非 test 泄漏**, 只需在正文注明。
- **B2 摘要 selective:** 确认摘要只报 LAION 38.5%, 漏 DFN 28.4%。
- **A1 pre-specified 矛盾:** 确认 L259 "grid pre-specified" 与 L141 "1000/3000 是后加" 自相矛盾; 另有 L158 "pre-specified 5pp margin"。
- **non-inferior:** 9 处; **A2 "sits at baseline":** 8 处; **McNemar "first completed seed":** L251(与 L277 "seed 0" 不一致)。

**新增 3 个分析(无新数据, 已算 + 写入正文):**
1. **A6 image-text alignment(RQ1 机制, 关键补强):** 用 DFN 文本编码器 + 缓存 test 特征, 测 16 类文本原型 vs 138 test 图的余弦。结果: 正确类文本均值 **0.287 < 最佳错误类 0.303**, 均值 margin **−0.016**; 正确类 rank-1 仅 **13.0%**(=zero-shot 精度), top-3 34.1%, top-10 60.9%, 平均 rank **7.8/16** → **直接量化了"text-image 决策规则失配"而非"表征缺信息"**(同一特征 supervised probe 85.5%)。存 `data/align_a6.json` + `scripts/align_a6.py`。
2. **A5 singleton 敏感性(§4.5):** 剔除 4 个单样本类后 BA: CLIP 77.4→**86.5%**, ResNet 84.3→**87.3%**, gap 6.84→**0.79pp**; 仅去掉 **class 13(1 张图, 0% vs 100%)** gap 就降到 **0.63pp** → 诚实揭示"ResNet 少数类占优"主要是单样本类假象, 12 个多图类上两者基本打平。
3. **B5 duplicate 计算筛查(Sec.3.1):** CLIP 嵌入近邻, 无 test 图对 train 余弦 >0.99(max **0.976**, 2/138 >0.97) → 升级为"计算筛查 + 目视"。

**文本降级(应用, 编译通过 19pp):**
- **B1 删 non-inferior:** 9→2(2 处为"not a formal non-inferiority/equivalence claim"否定式, 正确), 全部改为"descriptive ≈0.9pp gap, no equivalence claim"。
- **A1 消矛盾:** 删 "grid pre-specified" + "pre-specified 5pp margin"; C=300 → "validation argmax 在 {300, 3000} 平局, 取低复杂度成员 300"。
- **A2 at-baseline:** 8 处 "sits at / equals baseline" → "shows no detectable improvement over the majority-class baseline"。
- **B2 摘要:** K=1 报 **28.4–38.5%**(两 CLIP 变体 bracket ResNet 31.4%), 补 singleton 洞见。
- **B3 过度负面结论:** KG "Do not expect KG prompts to help" → "KG-augmented prompts did not improve ... under the evaluated implementation"; MLP "not a capacity effect" → "the tested MLP did not outperform the tuned linear probe"。
- **A3 ensemble 澄清:** 权重由 validation 确定, 只在 test 上评估一次。
- **McNemar seed:** "first completed seed" → "designated representative seed (seed 0)"。

**编译:** `pdflatex` clean, **19 页**, 无 error; overclaim 短语(sits at / pre-specified / grid pre-specified / first completed seed)全部清零。

**未处理 / 待定(user 决定或需重跑):**
- **A4 paired K-shot(唯一需重跑的核心实验):** CLIP 与 ResNet 用**相同** support set(K=1/5/10), 做 paired t / Wilcoxon / bootstrap。当前是独立采样(unpaired), 已在正文标注"protocol-level, not pure model effect", 但 reviewer 要求 matched。
- **A7 更强 frozen-feature 基线(可选, 无新数据):** RBF-SVM / nearest-centroid on `clip_features_clean.npz`(已有), 验证"85.5% 是否 frozen 表征天花板"。
- **C 级(低优先):** t-SNE 移 supplementary; "fine-grained" 给操作性定义; Appendix A bug 史中性化; citation/date 卫生(pubyear 2026, tpdd 年份)。
- **user 身份/投稿:** 作者信息; data-availability URL; response letter。

**结论:** 9 项 A 级中 6 项(含 3 个新分析 + 4 组 claim 降级)已处理; 剩 **A4 paired K-shot(需重跑)** 与 **A7 基线(可选)** 为实验项, C 级为低优先。论文"解释≤证据"的目标基本达成, 主结果重心已由 85.5 vs 86.4 转向 **13.0→85.5(表征可恢复)** + 直接测量的 text-image 失配。

---
## §16. Round-10: A4 PAIRED K-SHOT (reviewer's "must" item) + A7 RBF-SVM — DONE

**A7 (RBF-SVM, `scripts/heads_a7.py` → `data/heads_a7.json`):** RBF-SVM C=50 (val-selected) 87.0% top-1 / 84.8% balanced — above the 85.5% linear probe, ≈ResNet 86.4%. Nearest-centroid 8.7%. ⇒ "frozen DFN features are not exhausted by a linear head."

**A4 (paired K-shot, `scripts/paired_kshot_a4.py` → `data/paired_kshot_a4.json` + `_both.json`):** re-ran fine-tuned ResNet-18 on the *identical* K-per-class support set the CLIP probe uses per seed (shared stratified seed offset, seed+500), so both models train on the same images per seed → seed-paired. Paired t / Wilcoxon / bootstrap on per-seed Δ:
- K=1: LAION-2B 38.5% vs ResNet **32.1%** → **+6.4pp** (paired t p=0.0003, Wilcoxon p=0.0009, boot CI [+4.0,+8.9]); DFN-CLIP 28.4% trails by 3.7pp (p=0.064, borderline).
- K=5: ResNet **64.7%** leads LAION-2B 59.1% (−5.6pp, p=0.005) and DFN 48.3% (−16.5pp, p<0.0001).
- K=10: ResNet **75.2%** leads LAION-2B 66.5% (−8.7pp, p<0.0001) and DFN 57.5% (−17.6pp, p<0.0001).
⇒ Clean regime story: **CLIP (LAION-2B) leads at K=1; ResNet leads at K≥5**, now with PROPER PAIRED stats (was unpaired). Table 3 ResNet K=1/5/10 updated 31.4→32.1, 63.4→64.7, 72.4→75.2; abstract/intro/RQ2/results/discussion/contributions all re-framed seed-paired. Compiles clean, 19 pp.

**All reviewer A/B-level items now addressed.** Remaining: C-level (t-SNE→supplementary, "fine-grained" operational definition, Appendix A bug-history neutralize, citation/date cleanup) + user items (target journal, data-availability URL, response letter, author block).

---
## §17. Round-11: FINAL pre-submission tightening (chatgpt "可以准备投 CEA" review) — DONE

Reviewer verdict: **Minor-to-Major, ~85% mature, ready for CEA first submission** after the compressed list. No new experiments/models. All applied:

**🔴 Must-fix (4/4):**
- ① Section-4 C-grid `C∈{0.01,…,300}` → full **10-value grid** `{0.01,0.1,1,3,10,30,100,300,1000,3000}` (was inconsistent with Methods/Table).
- ② All "artifact of the four singleton classes" → **"largely driven by the four singleton test classes, with a smaller contribution from a mid-frequency class"** (abstract + RQ2; dropped "artifact").
- ③ "capacity does not" → **"under the tested heads"** (section title + body; added "we did not test a broader nonlinear-head family").
- ④ "safer choice for the rare fine-grained classes" / "handles the rare" → **"higher aggregate class-balanced performance on this test split, strongly influenced by singleton classes"** (conclusion + practical implication #4).

**🟠 Strongly-recommended (4/4):**
- ⑤ 0.90 cosine "near-identical" → **"highly similar … strongly aligned but not identical"** (Sec 2.2).
- ⑥ RBF-SVM 87.0% → clarified as **"exploratory nonlinear-head robustness check, not the primary figure"**; logistic probe retained as headline (Sec 4).
- ⑦ **Figure 2 regenerated**: now uses the **seed-paired** ResNet values (32.1/64.7/75.2) and adds a **tuned-C (C=300) CLIP point (85.5%)** at full data (star marker) — no longer "hides" the 85.5 result; caption updated.
- ⑧ McNemar seed-0 already framed as a "descriptive sample-level check against a designated representative seed" — kept de-emphasized.

**Soft:** abstract "leads" → "achieved a higher mean accuracy"; "best accounted for" → "strongly suggests".

Compiles clean, **20 pp**, exit 0. All reviewer A/B/C-level items now addressed.

**Remaining (user decision only):** target journal (CEA first submission), data-availability URL, response letter, author block (Lizhuo Li / Yunnan University).

## §18. Round-11 Experiment-Integrity Audit (experiment-audit skill) — DONE

Independent read-only subagent (fresh context; **same model family** — no GPT-5.5/Codex backend available, so the cross-model guarantee is NOT satisfied; treated as a strong safety net). Executor collected artifact paths only. Verdict **WARN** (material items all resolved before close).

- **A (GT provenance): PASS** — labels from TPDD `test/labels.json`; majority baseline 13.0% = `bincount` of true test labels (`p2_experiments.py:219-220`).
- **B (normalization): PASS** — plain accuracy/ECE/F1/McNemar, no self-normalized scores.
- **C (result existence): WARN → resolved:**
  - **C1 (material, FIXED):** 14.5% template ensemble is weighted by each template's **own test-set** accuracy (`scripts/zero_shot_clean.py:131-154`); prior text falsely claimed "a held-out validation split (never the test set)". Reframed as a *descriptive in-sample bound on template mixing (not an out-of-sample estimate)* (`tex:199`). Conclusion (no template/mixture beats 13.0%) unaffected.
  - **C2 (false positive):** "TLD 100.6%" existed only in the audit brief, not the paper; the paper's real 17.7% vs 19.0% is correct.
  - **C3 (provenance added):** singleton-exclusion BA (86.5 vs 87.3) had no stored artifact → `scripts/singleton_a5.py` → `data/singleton_a5.json`: best-C CLIP 77.41/ResNet 84.25 (gap 6.84 pp); excl 4 singletons 86.54/87.33 (gap 0.79 pp); excl class-13 alone 82.57/83.20 (gap 0.63 pp). Matches the paper.
  - **C4 (FIXED):** C-grid "flat top {100,300}" → true flat top `{100,300,1000,3000}` (90.0–90.4%), argmax tied `{300,3000}` (`tex:387,476,265`).
- **D (dead code): PASS** — no newly-referenced output missing; prior superseded-script items remain repo hygiene only.
- **E (scope): minor note** — few-shot *train* features under the disclosed stochastic augmentation (`clip_fewshot_probe.py:56`), reported mean±SD over 15 seeds; full-data protocol deterministic (byte-identical features). No population-level overclaim after the round-11 wording pass.
- **F (geometry phrasing): PASS** — within-backbone distance ratio (not cosine); consistent with `data/canonical_results.json`.

**Resolution:** C1 + C4 corrected in `.tex`; C3 provenance artifact added; C2 confirmed false positive. **Paper recompiled clean: 20 pp, `pdflatex` ×3 exit 0, no undefined references/citations.** Audit recorded in `EXPERIMENT_AUDIT.md` (§ Round-11 Re-Audit) and `EXPERIMENT_AUDIT.json` (`round11_reaudit`).

## §19. Round-12: Full experiment-audit skill run (strict reviewer independence) — DONE

**Protocol:** executor collected artifact paths only (no content read); a **fresh, separate-context reviewer subagent** was given *only* the paths + A–F checklist + adversarial mandate (no prior findings passed) — the skill's reviewer-independence workflow. Cross-model backend unavailable → same-model-family reviewer (noted as limit).

**Verdict: PASS** (up from prior WARN). A (GT) PASS · B (normalization) PASS · C (existence) PASS · D (dead code) PASS · E (scope) PASS · F = real_gt.

Resolved items (all verified against source before editing):
- **C-a (REAL, FIXED)** — L394 said "None of the evaluated heads reached ResNet-18 … untested heads (e.g., RBF SVM)", contradicting L261 (RBF-SVM 87.0% > ResNet 86.4%). Rewrote L394: pre-specified heads (logistic/MLP/kNN) at or below ResNet; exploratory RBF-SVM modestly exceeds it (87.0% vs 86.4%, within noise); adapters/LoRA/attention-pooling untested.
- **C-b (REAL, FIXED)** — L350 per-class caption "11 of 16" → **12 of 16** (verified: `canonical_results.json.per_class.clip_linear_C1` has 12/16 classes ≥50%).
- **C-c (method nuance, paper CORRECT)** — K=1 LAION Wilcoxon `w_p=0.0009` uses `zero_method='pratt'` (ties-inclusive; appropriate for the 2 tied seeds); the reviewer's 0.0015 uses scipy's default `'wilcox'` (ties-dropped). Both legitimate; now documented. No paper change.
- **C-d (provenance, FIXED)** — `data/paired_kshot_a4_both.json` (headline K=1 LAION-vs-ResNet p=0.0003) had **no writer**. Added `scripts/paired_kshot_a4_both.py` (LAION accs `two_model_e4.json`, DFN+ResNet `paired_kshot_a4.json`; paired t, Wilcoxon `pratt`, bootstrap CI `RandomState(10)`, B=10000). Verified it reproduces **every cited number exactly** (t_p 0.0003, w_p 0.0009, CI [+4.0,+8.9], delta 6.38; K=5/K=10 p-values).
- **ECE (advisory, not edited)** — ECE 0.14 (zero-shot, overconfident) / 0.46 (probe, underconfident) / conf 0.31>0.21 are internally consistent with `calibration_results.json` (runs at 15.2%/73.9% acc, distinct from headline 13.0%/76.8%); optional one-line clarification.

**Recompile:** 20 pp, `pdflatex` exit 0, no undefined references. Recorded in `EXPERIMENT_AUDIT.md` (Round-12) and `EXPERIMENT_AUDIT.json` (`round12_independent`).

## §20. De-first-person pass (We/Our → neutral/passive) — DONE

Per author request ("论文中尽量不要用We/Our"): every remaining **We/we/Our/our** in `clip_vs_resnet_tobacco.tex` was converted to a neutral passive/nominal form, preserving meaning and keeping the prose natural.

- **Scope:** all ~90 first-person instances across abstract, intro, related work, materials & methods, results, calibration, why, TLD, discussion, limitations, conclusion.
- **Strategy:** `We/Our X` → `X is/are …` (passive) or nominal subject; `We note that` → `Note that`; `Our results/work/key findings` → `The results/This work/The key findings`; `We do not claim/test/report` → `… is/are not claimed/tested/reported`. No number, claim, or citation altered.
- **Verification:** `grep -wE '\b(We|we|Our|our)\b'` → **0** matches. No double-verbs introduced. **Recompiled clean: 20 pp, `pdflatex` exit 0, no undefined references/citations.**
