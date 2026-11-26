# P3c — Pre-registration: DFN-CLIP few-shot fine-tuning under the matched adaptation budget

**Status:** PRE-REGISTERED (committed before any GPU time is spent). Nothing below may be
edited after the first run without recording the amendment and its reason in
`EXPERIMENT_TRACKER.md`.

**Date:** 2026-10-18
**Re-opens:** the DFN-CLIP @ K=5/10 fine-tune cells that `P3B_KSHOT_PREREGISTRATION.md`
(§2, A5) **deferred** as "confirmatory." This prereg corrects that deferral rationale (§0)
and re-opens the cells as decision-relevant.

---

## 0. Why the P3b deferral rationale was wrong (recorded so it cannot recur)

P3b §0/A5 deferred DFN-CLIP @ K=5/10 on the ground that "DFN-CLIP @ K=5/10 (gaps 17.5/19.3 pp
≫ gain) is confirmatory." That comparison used the **LAION-2B** full-data fine-tuning gain
(+7.2 pp) as the reference gain for **both** checkpoints — a category error. DFN-CLIP has its
**own** fine-tuning gain, and it is larger than LAION-2B's:

| K | DFN-CLIP frozen probe (C=1) | matched ResNet-18 | **DFN gap** | DFN-CLIP full-data fine-tune gain |
|---|---|---|---|---|
| 5  | 47.97% | 64.93% | **17.0 pp** | 76.8 → 95.5 = **+18.7 pp** (C=1 base); 86.2 → 95.5 = +9.3 pp (best-C base) |
| 10 | 56.43% | 74.64% | **18.2 pp** | same checkpoint-level gain |

The DFN-CLIP gap (17–18 pp) is therefore **comparable to**, not "≫", DFN-CLIP's own measured
fine-tuning gain. The DFN-CLIP @ K=5/10 cells are **decision-relevant, not confirmatory**:
fine-tuning may or may not close the gap. This is a genuine test, not a rubber stamp.

## 1. Question

Under the identical matched adaptation budget already applied to LAION-2B and ResNet-18, does
fine-tuned DFN-CLIP (image encoder + linear head) close its frozen-probe gap to ResNet-18 at
`K ∈ {5, 10}`?

## 2. Scope

DFN-CLIP @ K=5 and DFN-CLIP @ K=10, under the identical matched budget. After this, the
matched-budget fine-tune comparison is **checkpoint-complete**: both CLIP checkpoints have
fine-tune cells at K=5/10, matching the two LAION-2B cells. This directly closes the
asymmetry disclosed at L556 ("DFN-CLIP few-shot fine-tuning ... remain un-run").

## 3. Protocol (identical to the executed LAION-2B cells — non-negotiable)

- **Runner:** `scripts/clip_finetune_kshot.py --ckpt dfn_clip --k {5,10}`.
- **Checkpoint:** `dfn_clip` (ViT-L/14, 428M); `model.visual` fine-tuned + linear head (16).
- **Grid:** `epochs ∈ {30,100,300} × vis_lr ∈ {5e-6, 2e-5, 5e-5}` → 9 configs (asserted in runner).
- **Stage 1:** n=3 seeds (0,1,2); configuration selection on **validation only**.
- **Stage 2:** stage-1 winner re-run from scratch, **n=15 seeds**.
- **Support draw:** `coop_experiment.kshot_subset_idx(y, K, seed+500)`, asserted index-identical
  to `paired_kshot_a4.KShotSubset` at run time (runner aborts on mismatch).
- **Recipe:** head LR 1e-3, weight-decay 0.01, cosine schedule, GradScaler (fp16), batch 32,
  validation every 10 epochs, test evaluated exactly once at the best-val epoch.
- **Output (MUST override `--out`; the runner default collides with the LAION store):**
  - K=5  → `data/p3c_clip_finetune_kshot_dfn.json`
  - K=10 → `data/p3c_clip_finetune_kshot_dfn_k10.json`

## 4. Pre-committed decision rules (fixed before any result is seen)

`R*` = matched-budget ResNet-18 mean (K=5: 64.93%; K=10: 74.64%). `D*` = DFN-CLIP fine-tune
mean at that K. For each `K ∈ {5,10}`, classify by the seed-paired mean difference `D* − R*`
and its bootstrap 95% CI on the 15 shared support draws:

- **Branch R (reversal):** `D* ≥ R*` and CI lower bound > 0. The matched-budget reversal is
  **checkpoint-general**. Action: L30/L64/L264 state "both CLIP checkpoints"; Table 1 fills the
  DFN-CLIP fine-tuned K=5/10 cells; L556 updated to "both checkpoints run; reversal
  checkpoint-general."
- **Branch S (scope):** `D* < R*` (CI upper bound < 0). The reversal is **checkpoint-specific to
  LAION-2B**: DFN-CLIP's frozen-probe gap (17–18 pp) is not closed by its own fine-tuning under
  this budget. Action: L30/L64/L264 scope the few-shot reversal to LAION-2B; Table 1 fills the
  DFN-CLIP cells (showing they trail); L556 updated to "both checkpoints run; the few-shot
  reversal is checkpoint-specific." **This is a real, publishable finding (the reversal is
  checkpoint-conditional), not a failure.**
- **Branch T (tie):** CI spans 0. Report as a **tie** at that K (138-image test set underpowered
  at this cell); no directional claim.

Per-seed values and SD are reported for both cells in every branch, so the reader can re-derive
the classification.

## 5. Multiplicity

The executed family grows from 10 → **12** contrasts (add DFN-CLIP fine-tune K=5 and K=10).
Holm–Bonferroni (FWER) and Benjamini–Hochberg (FDR) are recomputed over all 12; both the 10- and
12-family results are reported. No post-hoc choice of family size.

## 6. Honesty guardrails

- Per-seed values and SD reported for every cell.
- If DFN-CLIP fine-tuning at K=5/10 collapses (e.g. < 50%), that is an
  **optimization-stability finding**, not evidence DFN-CLIP "cannot adapt at low K"; the full
  9-config sweep must be reported. A single collapsed run generalised to "DFN-CLIP fails at low
  K" is a forbidden reading.
- The runner's `best_val_monotonic_nondecreasing` tell-tale (sequential-warm-start bug) is
  asserted against; investigate if True.

## 7. Cost

Measured LAION-2B on the same card (RTX 4070 Ti, 12 GB): K=5 ≈ 3.2 h, K=10 ≈ 8.8 h. DFN-CLIP is
the same 428M architecture, so the same order is expected: **≈ 12 h GPU total.**

## 8. Artifacts

| item | path |
|---|---|
| runner (committed, re-runnable) | `scripts/clip_finetune_kshot.py` |
| DFN-CLIP output K=5 | `data/p3c_clip_finetune_kshot_dfn.json` |
| DFN-CLIP output K=10 | `data/p3c_clip_finetune_kshot_dfn_k10.json` |
| multiplicity (extended to 12) | `scripts/multiplicity_p3b.py` (updated) |

Gate checks in `scripts/check_tex_vs_data.py` are extended **only after** the numbers exist and
are written into the `.tex` — never before. No claim enters the manuscript ahead of its data.

## 9. Definition of done

1. DFN-CLIP @ K=5 and @ K=10 → the two `p3c_..._dfn...json` stores with per-seed values.
2. Branch assigned per §4 **by the pre-committed rule, not by inspection**.
3. Multiplicity recomputed over 12; `multiplicity_p3b.py` updated; gate extended to the
   12-family; `.tex` updated per §4; clean recompile; independently cross-validated.
4. Tracker entry recording the assigned branch and the numbers that justified it.
