# Experiment Plan — A+B expansion

**Problem.** Fine-grained crop-lesion classification: does frozen CLIP's *default readout* underperform a fine-tuned CNN in the low-data regime, and is that a *readout* effect we can fix — validated at powered scale, not just a 138-image set.

**Method thesis.** CLIP's CLS-pooled readout is a weak default on fine-grained foreground lesions (consistent with the background-aggregation finding of Shi et al., CVPR 2026). A **foreground-anchored patch-attention head on the frozen CLIP encoder** recovers a significant fraction of the CLIP↔fine-tuned-CNN gap, and this holds on a large multi-crop benchmark.

**Date.** 2026-10-14
**Compute.** Single RTX 4070 Ti (12 GB), 12 cores, 15 GB RAM. ResNet via torchvision; CLIP ViT-L/14 (DFN, LAION-2B) + DINOv2 local.

> ## OUTCOME (2026-10-14, as executed) — supersedes the C1 "method win" framing below
> - **M0 gate STOP (true negative).** DFN-CLIP foreground-anchored attention head overfits only 51.5% train and loses 6.5pp at K=1 to the CLS probe (`data/m0_gate.json`). Anti-false-negative diagnostic (`data/m0_diagnostic.json`) shows the STOP is a true negative: DFN-CLIP patch tokens are near-constant (mean pairwise ℓ1 0.011) and carry no extra class signal (patch-only 18.8%); LAION-2B reverses (mean-pool 87.0% vs CLS 76.1%, +10.9pp).
> - **Lever B REFRAMED** from "cheap frozen-encoder fix / method win" (C1) to an **honest readout-geometry characterization**: the CLS-vs-patch-token signal split is checkpoint-dependent, directionally consistent with Shi et al. 2026, and explicitly **not a new method**. C1 claim dropped; no overclaim.
> - **Lever A (powered multi-crop): DONE for tomato + corn.** Both already on disk and in paper; seed-paired powered stats added (tomato K1 +4.3pp p=0.051, K5/K10 p<0.001; corn K1 +8.1pp p=0.021, K5 n.s., K10 p<0.001).
> - **M3 (full 55k PlantVillage): SKIPPED** (user decision) — tomato+corn already provide the powered multi-crop evidence.
> - **M5 gate: GREEN.** 189/189 checker PASS; clean compile 27pp, 0 errors, 0 undefined refs/cites, 0 overfull>100pt, 0 TAB. Limitation (11) scopes the readout-geometry analysis (TPDD-only 138 test, single attn-head variant).
> - See `EXPERIMENT_TRACKER.md` decision log for the full per-run record.

## Claim Map
| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|-------|----------------|------------------------------|---------------|
| **C1 (method, primary)** | Turns "CLIP is worse" into "CLIP's *default readout* is a weak default; here is a cheap frozen-encoder fix" | Patch-attention head ≥ CLS-probe at K=1/5 and narrows the gap to fine-tuned ResNet, **beyond** equal-capacity heads (MLP-on-CLS, CoOp, LP++), improving confusable-class errors | B1, B2, B4 |
| **C2 (powered scale, supporting)** | Fixes the load-bearing weakness (138-image test set); makes magnitude claims defensible | Full protocol (zero-shot, K-shot, full-data) reproduced on full PlantVillage (14 crops / 55k / 39 classes) with per-seed difference SE ≤ ~1.2 pp (15 seeds detect ~1.2 pp) | B3 |
| Anti-claim (capacity) | Reviewer: "the gain is just more parameters" | Equal-param MLP on the SAME pooled CLS features, CoOp, LP++ do NOT match the patch-attention head | B2 |
| Anti-claim (dataset-specific) | Reviewer: "only works on tobacco" | C1 holds on tomato + full PV, not only TPDD | B3 |

## Paper storyline
- **Main paper must prove:** C1 (head closes the gap on fine-grained lesions) + C2 (it is powered, multi-crop).
- **Appendix can support:** ablations (B5), full per-class tables, PiB/foreground-alignment probe.
- **Intentionally cut:** re-implementing Shi et al. pretraining (LaSt-ViT) — out of scope; we cite it as motivation, not a baseline to reproduce.

## The new head (B) — design
- **What:** learned attention-pooling over ViT-L/14 **patch tokens** (196×1024) instead of the CLS token. `q` learnable query; out = softmax(q W · patchT) · patch → (optionally a 2-layer MLP) → class logits. Image encoder **frozen**; train `q,W` (+MLP) with Adam, batch 32, 10–30 epochs, **same K-shot support sets and seeds as CoOp/LP++** (fair head comparison).
- **Fairness:** compare against (a) equal-param MLP on pooled CLS features, (b) CoOp (text-side), (c) LP++ (blend head) — isolates "reading patch tokens / foreground" from "head capacity."
- **Direct LAST-ViT hook (B4):** a PiB-style foreground-alignment probe showing the head shifts readout mass toward lesion/foreground patches vs the CLS baseline.

## Experiment blocks
### B1 — Anchor (MUST)
- Claim: C1. Dataset: TPDD (16 cls, 1295/138/138), K∈{1,5,10}+full, 15 seeds, 2 ckpts (DFN, LAION).
- Systems: patch-attn head vs CLS-probe (C-swept) vs fine-tuned ResNet-18.
- Success: head ≥ CLS-probe at K=1/5 and narrows the CNN gap. Failure: head ≤ CLS-probe → "readout effect" claim dead; report honestly, drop C1.
- Table: main K-shot table (replaces/augments existing Table 1).

### B2 — Novelty isolation (MUST)
- Claim: C1 (not capacity). Same data/seeds.
- Systems: patch-attn vs MLP-on-CLS (equal params) vs CoOp vs LP++.
- Success: patch-attn (reads patch tokens) beats pooled-feature heads → the *mechanism* matters. Failure: pooled head matches → it's just capacity; reframe.
- Table: head-capacity ablation table.

### B3 — Powered scale / lever A (MUST)
- Claim: C2. Dataset: **full PlantVillage** (39 classes, 55,447 imgs, 14 crops; clean "no augmentation" zip on disk), fixed stratified split; also reuse tomato (1635 test) + corn (346 test) as powered single-crop slices.
- Systems: full protocol (zero-shot 14-template, K∈{1,5,10}, full-data probe, fine-tuned ResNet-18/50) + the new head.
- Success: regime ordering reproduces with per-seed diff SE ≤ ~1.2 pp (tomato) / ≤ ~0.8 pp (corn); head still narrows the gap.
- Cost note: 55k × 2 ckpts CLS-feature extraction is the biggest single job (~2–6 h); do **not** cache 55k patch tokens (stream), only CLS (~170 MB).

### B4 — Failure / mechanism analysis (MUST, cheap)
- Claim: C1 (where it helps). Per-class + confusable-class breakdown (tobacco: nematodes/PVY/CMV/sunscald; PV: the analogous confusable crop pairs).
- Add PiB-style foreground-alignment probe (echoes Shi et al.) on cached patch features: does the head move readout mass to foreground?
- Figure: per-class bar + foreground-mass shift.

### B5 — Ablations (NICE, appendix)
- attention temperature; top-K patch selection; patch-size; "head without text" check.

## Run order & milestones
| M | Goal | Runs | Gate (stop/go) | Est. cost (4070 Ti) |
|---|------|------|----------------|---------------------|
| M0 sanity | patch+CLS extraction TPDD (2 ckpt); new head overfits a toy split; features reproducible | 1 | head > CLS-probe on overfit toy | ~0.5–1 h |
| M1 baseline | CLS-probe + fine-tuned ResNet-18 on TPDD (re-baseline) | 1 | matches published in-paper numbers | ~1–2 h |
| M2 main | B1+B2: patch-attn + 3 head controls, K∈{1,5,10}+full, 15 seeds, 2 ckpt | 2 | C1 holds at K=1/5 | ~4–8 h |
| M3 scale | B3: full PV feature extract (2 ckpt) + heads + ResNet on PV | 3 | C2: SE≤~1.2 pp, ordering reproduces | **~1–3 days wall** (extract + ResNet on 55k) |
| M4 analysis | B4: per-class + PiB probe | 4 | mechanism figure coherent | ~1–2 h |
| M5 polish | B5 ablations + write-up + re-run 189-check + recompile | 5 | checker green, PDF clean | ~2–4 h |

**Separation:** M0–M2 are cheap/reversible (start now). M3 is the expensive, multi-day commitment (confirm before launch). M4–M5 follow.

## Power justification (a priori)
- Per-seed SE of a paired accuracy difference scales as ~1/√N_test. At **TPDD N_test=138**, 15 seeds detect only a **~4 pp** gap — the current 0.6–1.7 pp few-shot diffs are *inside noise* (confirms the weakness).
- At **tomato N_test=1635**, 15 seeds detect **~1.2 pp**; **corn N_test=346** → **~0.8 pp**. Hence the powered benchmark must be PlantVillage (tomato/corn/full), not TPDD.
- Unit of inference = **15-seed paired test** on the *overall* accuracy (not per-class). Per-class is reported descriptively (B4); any per-class hypothesis testing is Holm-corrected **within** the pre-registered confusable set only.

## Compute & data budget
- **Total:** ~2–4 days wall-clock on one 4070 Ti (M3 dominates).
- **Data prep:** extract full-PV zip (828 MB → ~1–2 GB); build fixed stratified split (82/9/9) with seed 42; no augmentation.
- **Biggest bottleneck:** 55k × 2-ckpt CLS extraction + ResNet fine-tune on 55k at 12 GB (use bf16 + grad-accum; ResNet-18 first, then -50).

## Risks & mitigations
- **12 GB VRAM on ResNet-50 / 55k:** bf16 + gradient accumulation; if OOM, batch 16. ResNet-18 is the primary CNN baseline.
- **Patch-token RAM (55k):** do not cache; stream forward, train head in mini-batches (pattern: `clip_finetune.py`).
- **Leakage:** use the clean no-augmentation PV release; single fixed split (seed 42) shared by all systems.
- **Multiple comparisons:** inference unit = 15-seed paired overall-accuracy test; per-class descriptive only; Holm within confusable set.
- **Honesty gate:** if B1/B2 show the head does NOT beat the pooled-feature heads, we do NOT claim C1 — we report the negative and the paper stays a (now powered) evaluation.

## Final checklist
- [ ] C1 (head closes gap) covered by B1/B2
- [ ] Capacity anti-claim isolated (B2)
- [ ] C2 (powered multi-crop) covered by B3
- [ ] Dataset-specific anti-claim covered by B3 (tomato+corn+full)
- [ ] Nice-to-have (B5) separated from must-run
- [ ] No results fabricated — all numbers come from fresh runs vs `data/*.json`
- [ ] 189-check re-run + clean recompile after results land
