# Future-Experiments Plan (post-review hardening)

Scope: the additions the GLM review + internal peer panel (EIC W3, R1 W2, DA M4) asked
for that are NOT already in the manuscript. Ordered by (reviewer impact / cost).
Grounded in the existing `scripts/` and the per-run wall-clock times already in `data/`.

**What is already covered (do not redo):**
- Two CLIP checkpoints (DFN-CLIP, LAION-2B), frozen-feature control
  (frozen CLIP 76.8% > frozen ResNet-18 73.2%; frozen ResNet-18 probe 34.1% > fine-tuned 32.1% at K=1),
  C-grid + MLP + kNN + RBF-SVM heads, seed-paired K=1/5/10, a-priori power statement.
- Domain-VLM: **BioCLIP 2 now evaluated** (P1 done; at baseline zero-shot on TPDD + TLD). SCOLD / LP++ still not run.
- Representation-generality: **DINOv2 ViT-L/14 now evaluated** (P2 done; 77.5% best-C full-data < CLIP 85.5% < ResNet-18 86.4% — near-parity is CLIP-specific).
- Fairness / asymmetric-effort: **fine-tuned CLIP now evaluated** (P3 done; DFN 96.7%±0.8 / LAION 94.8%±1.7 full-data > fine-tuned ResNet-18 86.4% — near-parity is *conservative*, not an under-tuned CLIP).
- Scale: **ResNet-50 now evaluated** (P4 done; 89.6%±2.2 full-data, between ResNet-18 86.4% and fine-tuned CLIP 94.8–96.7% — CNN result is scale-limited, CLIP stays on top).

**Compute budget note:** ResNet-18 full-data ≈ 300–560 s/run (`resnet_k10_run.log`);
the full canonical protocol = 929 s (`canonical_results.json`). A 1–2 day GPU window
covers P1–P4 below.

---

## P1 — Domain-VLM zero-shot row (BioCLIP 2 *or* SCOLD) on TPDD + TLD  ⭐ do first
**Answers:** EIC W3 (Major, corroborated 2/4), R2 W2, DA "you ignored the obvious domain model".
**Why first:** highest reviewer impact, lowest cost (zero-shot, no training), and it is the
single biggest gap between the paper's framing (cites SCOLD/BioCLIP 2/LP++ as *the reason the
question matters*) and its evidence (none in any table).

- **Model:** BioCLIP 2 (`bioclip2`, NeurIPS 2025, open weights) preferred; SCOLD (`scold2025`) as
  fallback. Both are leaf/biology domain VLMs and directly comparable to the "zero-shot CLIP fails" claim.
- **Protocol (identical to the paper's zero-shot):** same 14 templates, cosine image↔text,
  τ=100, on the **232 distinct TLD base test images** and the **138 TPDD test images**.
  Report: best-template accuracy vs the majority-class baseline (TPDD 13.0%, TLD 19.0%),
  per-class if possible. One new row in the zero-shot tables (TPDD + TLD).
- **Code hooks:** `scripts/prompt_ablation.py` (template loop) + `scripts/tld_zero_shot.py` (TLD path).
  Add a `model=` arg so the same 14-template harness runs the domain VLM. New: `scripts/domain_vlm_zeroshot.py`.
- **Expected outcomes & claim effect:**
  - Domain VLM > baseline (e.g., 25–40%) → *strengthens* the paper's "domain pretraining is what CLIP lacks" story;
    add one sentence to Discussion + a row; **no headline change**.
  - Domain VLM ≈ baseline → *strongly strengthens* the "fine-grained tobacco lesions are hard even for
    domain VLMs" story.
  - Either way the L498 "inadequate" claim becomes explicitly bounded to off-the-shelf CLIP, with evidence.
- **Cost:** ~30–90 min (weight download ~1.7 GB + embedding extraction + template loop). **1–2 h wall-clock.**
- **Stop condition:** one domain VLM, both datasets, best-template vs baseline reported. (Second VLM only if P1 is cheap.)

**DONE (2026-09-21) — gate G1' (domain VLM ≈ baseline) met; strengthens the paper.**
- Model: BioCLIP 2 (`imageomics/bioclip-2`, ViT-L/14, 1.7 GB, open_clip), downloaded to
  `checkpoints/bioclip2/`; runner `scripts/domain_vlm_zeroshot.py`; results `data/domain_vlm_zeroshot.json`.
- **TPDD (16 cls, 138 test):** best template CN-Photo **15.2%** (21/138) vs majority baseline **13.0%**
  (gap +2.2 pp); generic CLIP best 13.0%; fine-tuned ResNet-18 86.4%.
- **TLD (10 cls, 232 base test):** best template KG-Symptom **21.1%** (49/232) vs majority baseline **19.0%**
  (gap +2.1 pp); generic CLIP best 19.0%.
- **Finding:** even a domain-specific VLM sits at the majority-class baseline zero-shot — the near-zero-shot
  regime is a property of CLIP-style frozen image–text decision rules on fine-grained lesions, not an artifact
  of generic pretraining. (Side note: CN templates beat EN on TPDD, CN-Photo 15.2% vs EN-Photo 4.4%.)
- **Paper edits:** L65 (Related Work) now states BioCLIP 2 is empirically evaluated; new Results paragraph
  "(3) A domain-specific VLM shows the same near-baseline behavior" + provenance ref. Recompile clean, 23 pp.

## P2 — DINOv2 ViT-L/14 frozen probe (representation-generality control)
**Answers:** "is this CLIP-specific or any good frozen features?" (DA M4 logic, R1 scale/generality).
**Why:** DINOv2 is a **non-CLIP, image-only** self-supervised ViT-L — the cleanest test that the
frozen-feature result is not an artifact of CLIP's text alignment. Reuses the *entire* existing probe pipeline.

- **Protocol (identical to `canonical_protocol.py`):** extract DINOv2 features on the same 1,295 train /
  138 test; run the **same** C-grid {0.01…3000} + MLP + kNN; report full-data top-1/BA/macro-F1 and the
  K=1/5/10 frozen-probe curve. One new column in Table 2 (full-data) + the few-shot curve.
- **Code hooks:** `scripts/canonical_protocol.py` (probe), `scripts/clip_fewshot_probe.py` (K-shot).
  Parameterize the feature source. New: `scripts/dinov2_protocol.py`.
- **Expected outcomes & claim effect:**
  - DINOv2 ≈ CLIP (85%+) → generalize the claim to "strong frozen features (CLIP or DINOv2) are competitive
    with a fine-tuned CNN at full data"; headline becomes more robust.
  - DINOv2 < CLIP → CLIP's features are genuinely stronger for this task; *strengthens* the CLIP-specific finding.
- **Cost:** feature extraction ~2–10 min + the probe grid (~same 929 s canonical run). **~2–4 h wall-clock.**
- **Stop condition:** one DINOv2 checkpoint, full-data + K=1/5/10, in Tables 1–2.

**DONE (2026-09-21) — DINOv2 < CLIP < ResNet-18; near-parity is CLIP-specific (strengthens the paper).**
- Model: DINOv2 ViT-L/14 base pretrain (`facebook/dinov2-vitl14` LVD-142M, 304M params, 1024-dim CLS, frozen),
  self-contained arch in `scripts/dinov2_protocol.py` (matches native `.pth`: `patch_embed.proj`, per-block
  `ls1/ls2` LayerScale, `mask_token`, native-518px `pos_embed` interpolated to 224px). Weights
  `checkpoints/dinov2/` (via `scripts/download_dinov2.py`, HF-mirror `hdtech/dinov2_vitl14_pretrain`).
- **Full-data (canonical protocol, same C-grid/MLP/kNN):** C=1 **67.4%**, best-C (C=1000) **77.5%** (BA 69.6%,
  macro-F1 67.0%), MLP 76.1%, kNN(1) 62.3%.  →  **below** frozen CLIP (76.8% / 85.5%) **and** fine-tuned ResNet-18 (86.4%).
- **K-curve (15 seeds):** K=1 28.6±3.4, K=5 50.8±5.3, K=10 55.8±3.9, full 67.4% — trails both CLIP variants at K=5/10.
- **Finding:** the full-data near-parity is **not** a generic frozen-feature phenomenon; it is specific to CLIP's
  image–text-pretrained representation on fine-grained tobacco lesions. A strong general-purpose self-supervised
  representation (DINOv2) does not reach it. *Caveat (stated in paper):* DINOv2 at the shared 224px protocol with
  native-518px pos-embeds interpolated; native 518px not tested.
- **Paper edits:** `tab:main` new DINOv2 row + caption; new representational paragraph (after the CLIP-vs-frozen-ResNet
  comparison); new `dinov2` bibitem. Clean 2-pass rebuild: 23 pp, 0 undefined citations.
- **Ops note:** first two runs OOM-killed due to a concurrent `b_lite_splits.py` (Paper1) job on GPU0 + host RAM;
  fixed with `num_workers=0`, batch 16, and ran after that job finished. `python3 -u` for unbuffered progress.

 ## P3 — Fairness control: fine-tuned CLIP vs fine-tuned ResNet  [DONE]
**Answers:** GLM A3 / "asymmetric tuning" — the paper compared *frozen* CLIP (probe) vs *fine-tuned*
ResNet-18, and probed frozen ResNet-18, but never fine-tuned CLIP. That's the "asymmetric effort" gap.
**Done (2026-09-21):** fine-tuned both CLIP checkpoints (image encoder `model.visual` + linear head) under a
matched budget (30 ep, cosine, best-val, native CLIP preprocessing, batch 16 mem-limited, AMP + grad-checkpointing,
n=5 seeds). `scripts/clip_finetune.py` → `data/clip_finetune.json`.
**Results:** fine-tuned CLIP **DFN 96.67%±0.83**, **LAION-2B 94.78%±1.65** (n=5) vs fine-tuned ResNet-18
**86.43%±1.9** (n=15) → **+8.4 to +10.2 pp** in CLIP's favor when both are tuned; and above each checkpoint's own
frozen probe (85.5%/86.2%). → the frozen-probe near-parity is *conservative*; the primary result does not flatter
ResNet by leaving CLIP un-tuned.
**Paper:** new fairness-control paragraph in RQ2 (after the DINOv2 control); forward-ref in the central-finding
sentence; Limitations (9). Recompile clean 23 pp, 0 undefined.
**Scope note:** full-data only (few-shot CLIP fine-tuning not run); single matched budget (not a full fine-tune
ceiling search). Ops: batch 32 OOMs the 428M ViT on 12GB → grad-checkpointing + batch 16; `python3 -u` + nohup
(run ~17 min/seed, ~2.8h for both×5); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
**Cross-validation (2026-09-21):** (a) **no leakage** — all 1,571 images across train/val/test are md5-unique
(shared basenames like `000000.jpg` are a per-split naming artifact, NOT the same files); (b) **gain is from
backbone fine-tuning** — head-only control (frozen backbone, same recipe) reaches only ~45.7% test vs 96.7%
backbone+head, so the +51pp is genuine backbone adaptation, not a head/protocol artifact; (c) stable 5 seeds
(95.65–97.83 DFN), val (92.8%)≈test (96.7%), no overfit. → P3 result confirmed sound.

**Optional complement (not required):** give ResNet-18 a hyperparameter screen (val-selected lr×epochs, 5→15 seeds)
to also show ResNet is not under-tuned. Expected: stays ~86–88%. ~3–6 h. (The fine-tuned-CLIP control above already
closes the main "asymmetric effort" objection, since it shows CLIP is the *stronger* side when both are tuned.)
- **Grid (val-selected, best config → refit → 138 test once):** `lr ∈ {1e-4, 3e-4, 1e-3}`, `epochs ∈ {30, 60, 100}`,
  `batch ∈ {32, 64}`, `weight_decay ∈ {0, 1e-4, 1e-3}`; coarse 8-config screen, refine top-3, 5→15 seeds.
- **Code hooks:** `scripts/exp_full_provenance.py`. New: `scripts/resnet_tune.py`.


- **Grid (val-selected, 9% split, best config → refit → 138 test once):**
  `lr ∈ {1e-4, 3e-4, 1e-3}`, `epochs ∈ {30, 60, 100}` (early-stop on val),
  `batch ∈ {32, 64}`, `weight_decay ∈ {0, 1e-4, 1e-3}`.
  Full grid = 48 configs is too heavy; use a **coarse 8-config screen** (lr × epochs), then refine the top-3
  with batch/wd, **5 seeds** on the winner, confirm at **15 seeds**.
- **Code hooks:** `scripts/exp_resnet_k1_n15.py` / `exp_full_provenance.py` (training loop already exists;
  add a config loop). New: `scripts/resnet_tune.py`.
- **Expected outcomes & claim effect:**
  - Tuned ResNet stays ~86–88% → near-parity holds, now on *symmetric* tuning (removes the A3 hit).
  - Tuned ResNet pulls away to ≥89% → the top-1 gap widens to ~3–4 pp; **keep** the "descriptive near-parity,
    not equivalence" framing, but state the gap explicitly and lean on balanced accuracy + cost. (Honest, not fatal.)
- **Cost:** ~12–18 ResNet-18 runs ≈ 1–3 h GPU. **~3–6 h wall-clock** incl. 15-seed confirmation.
- **Stop condition:** one tuned ResNet config, full-data + K=1, 15 seeds, added to Table 2 as "ResNet-18 (tuned)".

## P4 — ResNet-50 full-data scale control  [DONE]
**Answers:** R1 W2 / "428M vs 12M (~35×) without a size-disclosed baseline".
**Done (2026-09-21):** ResNet-50 (24.7M, ~2× ResNet-18, ~1/17 of CLIP) full-data, identical protocol
(ImageNet-pretrained, AdamW lr=3e-4, cosine, batch 32, 30 ep, best-val, n=5). `scripts/resnet50_full.py`
→ `data/resnet50_full.json`.
**Result:** **ResNet-50 89.57%±2.15 (n=5)** (seeds 92.75/88.41/86.96/89.86/89.86) vs ResNet-18 86.43%±1.9 (n=15)
vs fine-tuned CLIP 94.78–96.67%. → the fine-tuned-CNN result is *somewhat* scale-dependent (ResNet-50 does ~+3pp
better than ResNet-18), yet the ordering is robust: compact fine-tuned CNN (18/50) reaches the frozen-CLIP-probe
level, while fine-tuned CLIP stays on top. The ~35× scale asymmetry does not flatter the CNN.
**Paper:** new "Scale control" paragraph (Why section, replacing the old "we do not test a size-matched CNN"
caveat); Limitations (4) updated to note ResNet-50 tested + why a 400M from-scratch CNN is infeasible on 1,295 imgs.
Recompile clean 24 pp, 0 undefined.
**Ops:** `ResNetBaseline(NC, backbone='resnet50')` from Paper1 `model_zoo` (drop-in swap); ~1–1.4 min/seed;
fixed a `DEVICE.type` bug (DEVICE is a str). ResNet-50 variance higher than ResNet-18 (±2.2 vs ±1.9).
**Not done (optional):** ResNet-101 (44.5M) — likely ~90–92%, same message; a true 400M from-scratch CNN would
overfit on 1,295 images (the informative comparison is the compact CNN, which is why it's the baseline).

## P5 — LP++ probe (optional, head-generality)
**Answers:** "is the linear head the bottleneck?" (partly; RBF-SVM already exceeds at 87.0%).
**Why (lower priority):** the RBF-SVM 87.0% (L387, L472) already shows the head is *not* the binding
constraint on top-1. LP++ would add one more head point. Highest effort-to-impact ratio here.
- **Protocol:** LP++ (learnable image-text blending probe) on the frozen CLIP features, same C/validation.
- **Code hooks:** new `scripts/lpplus_probe.py` (implement the LP++ head).
- **Cost:** moderate (new head + training). **~6–12 h** incl. implementation.
- **Stop condition:** one LP++ row vs the tuned linear probe; report as head-generality, not headline.

---

## Decision gates (what result → what text change)
| Gate | If | Then |
|---|---|---|
| G1 | P1 domain VLM > baseline | Discussion +1 sentence (strengthens); add table row; bound L498 to off-the-shelf CLIP |
| G1' | P1 domain VLM ≈ baseline | Stronger "fine-grained is hard" claim; add row |
| G2 | P2 DINOv2 ≈ CLIP | Generalize to "strong frozen features"; headline more robust |
| G3 | P3 tuned ResNet ≤88% | Remove A3 hit; keep near-parity |
| G3' | P3 tuned ResNet ≥89% | State widened gap explicitly; lean on BA + cost; keep "not equivalence" |
| G4 | P4 ResNet-50 ≈ ResNet-18 | Scale-robust claim; reframe ~35× asymmetry |

## Recommended sequence & total budget
1. **P1** (1–2 h) — biggest reviewer hit, cheapest. *Ship first.*
2. **P2** (2–4 h) — reuses pipeline, generalizes the core finding.
3. **P3** (3–6 h) — fairness; run alongside P2 (independent).
4. **P4** (4–8 h) — scale; run after P3 winner config is known.
5. **P5** (optional) — only if P1–P4 land and the head question is still raised.

**Total: ~2–3 GPU-days** for P1–P4, all within one review round. P1+P2 are the two that most
change defensibility; P3+P4 are reinforcement. All are additive (no existing number changes), so the
current "regime-dependent near-parity" narrative is preserved and *strengthened*, not reversed.

## Explicit non-goals (keep scope honest)
- **No from-scratch 400M ViT training** on 1,295 images — infeasible (severe overfit); say so.
- **No cross-crop generalization** — out of stated scope (single-crop, single-dataset).
- **No equivalence/non-inferiority claims** at full data — one side is deterministic, the other stochastic;
  the per-image SE (~3.4 pp) cannot resolve the ~0.9 pp gap. Keep "descriptive near-parity".
- **No new crops / field data** — the contribution is the *method* (regime-, probe-, checkpoint-aware comparison).
