# P3b — Pre-registration: few-shot fine-tuned CLIP under a matched adaptation budget

**Status:** PRE-REGISTERED (committed before any GPU time is spent). Nothing below may be
edited after the first run without recording the amendment and its reason in
`EXPERIMENT_TRACKER.md`.

**Date:** 2026-10-16 (amended same day, **before any GPU run** — see §0.1)
**Naming:** `P3b`, not `P3`. `scripts/clip_finetune.py` already labels *itself* "P3"
(the full-data fairness control, L261). This is the **few-shot** counterpart.

---

## 0.1 Amendment log

- **A1 (2026-10-16, pre-run).** Two corrections to the §0 arithmetic, after checking the
  canonical rows directly. `data/clip_fewshot.json` is the **DFN-CLIP** row;
  `data/laion_fewshot_canonical.json` is **LAION-2B**. The paper's headline crossover claim
  is about LAION-2B. Corrected below.
- **A2 (2026-10-16, pre-run).** A seed-pairing concern was raised and **retracted**: the
  `seed+300` draw lives in `scripts/exp_resnet_k5_k10_n15.py`, whose output
  `data/resnet_k5_n15.json` the gate already labels `_provenance: non-authoritative`. The
  authoritative ResNet row is `data/paired_kshot_a4.json`, produced by
  `scripts/paired_kshot_a4.py:69` with `KShotSubset(train_full, K, seed+500)`. Equivalence of
  its local `KShotSubset` to `coop_experiment.kshot_subset_idx` was **verified empirically**
  (identical indices for K ∈ {1,2,5,10} × seeds {0,1,7,14}). The paper's seed-pairing claim
  stands. The P3b runner re-verifies this at run time as an assertion.
- **A3 (2026-10-16, pre-run).** Cost corrected: measured full-data CLIP fine-tune is
  **~1000 s per seed** (976–1524 s). The original "hours, not days" estimate ignored the
  9-config × 15-seed grid and per-epoch val+test evaluation. §5 replaced with a two-stage
  search and sparse evaluation.
- **A4 (2026-10-16, pre-run).** The §6 branch thresholds referenced the *single-config*
  ResNet (65.5% / 75.7%; protocol `lr=3e-4, 30ep`, **no search**). Under a matched search
  that number will likely rise, so thresholds are re-based on the **new** matched-budget
  ResNet. **User decision:** the matched-budget ResNet **replaces** the paper's K-shot ResNet
  values, and the new value is propagated through L249, Table 1 and Figure 1 rather than
  being carried as a second parallel row.
- **A5 (2026-10-16, pre-run).** Scope narrowed by user decision to the **decisive cell
  only**: LAION-2B @ K=5 (plus the matched-budget ResNet @ K=5 needed as its threshold).
  Remaining cells are deferred, not cancelled.

---

## 0. The correction that motivates this experiment

An earlier verbal assessment of this experiment was **wrong twice** and is recorded here so
the error cannot recur:

1. It quoted a canonical CLIP `K=10` of "~62–66%". The true canonical value is
   **56.43%** (`data/clip_fewshot.json`). The 62–66 range was the LP++ variant, not the
   canonical probe.
2. It then compared that 56.43% (which is the **DFN-CLIP** row) against ResNet and concluded
   the gap was ~2× the measured fine-tuning gain at both `K=5` and `K=10`, i.e. "safe,
   confirmatory, run K=10 first".

Both are wrong for the checkpoint the paper's headline is about. The paper's central claim —
the crossover bracketed to `(2,5)` — is about **LAION-2B**, whose row is
`data/laion_fewshot_canonical.json`. The corrected gaps:

| K | LAION-2B frozen probe | ResNet-18 fine-tuned | **LAION gap** | DFN gap |
|---|---|---|---|---|
| 5 | 60.19% | 65.5% | **5.3 pp** | 17.5 pp |
| 10 | 66.47% | 75.7% | **9.3 pp** | 19.3 pp |

Measured CLIP fine-tuning gain **for that same checkpoint at full data: +7.2 pp**
(LAION 87.7% → 94.9%; `data/clip_finetune.json`).

**Therefore the `K=5` LAION gap (5.3 pp) is SMALLER than the measured fine-tuning gain
(+7.2 pp).** If fine-tuning realises anything like its full-data gain, LAION-2B overtakes
ResNet-18 at `K=5` and **the crossover bracket `(2,5)` collapses — there is no crossover at
all**. This experiment is therefore **not** confirmatory. It can invalidate the title claim.

---

## 1. Question

Under an adaptation budget matched *between the two backbones at the same `K`*, does
fine-tuned CLIP still trail fine-tuned ResNet-18 at `K ∈ {5, 10}`?

The paper's present framing (L261, L482, L553) concedes that the primary contrast is an
*unequal-adaptation* comparison (frozen large encoder vs fully fine-tuned small CNN) and
absorbs the objection in prose. Prose is not evidence. This experiment converts
"we did not test it" into "we tested it under a matched budget".

## 2. Pre-committed priority and scope

**Scope (A5): LAION-2B @ K=5, plus matched-budget ResNet-18 @ K=5 as its threshold.**

`K=5` LAION is the single highest-risk cell in the paper: it sits at the lower edge of the
claimed crossover interval and its gap (5.3 pp) is smaller than the measured fine-tuning
gain (+7.2 pp). DFN-CLIP @ K=5/10 (gaps 17.5/19.3 pp ≫ gain) is confirmatory and is
**deferred**. `K=10` is *not* the priority — its gap (9.3 pp) exceeds the gain, making it the
safer of the two LAION cells, not the riskier one. If and only if this cell returns Branch A
or C do the remaining cells become decision-relevant.

## 2.1 Protocol requirements (non-negotiable)

1. **Seed-paired support sets.** Support indices MUST come from
   `coop_experiment.kshot_subset_idx(y, K, seed+500)` — byte-identical to the draws behind
   the canonical probe rows and the ResNet `K`-shot rows (verified per A2). The runner MUST
   assert equality against `paired_kshot_a4.KShotSubset` at run time and abort on mismatch.
   `scripts/exp_resnet_k5_k10_n15.py` (`seed+300`) MUST NOT be used — it is superseded.
2. **Independent seeds.** Each seed MUST restore the pretrained `model.visual` weights
   before training. The sequential-warm-start bug already found and fixed in
   `scripts/clip_finetune.py` MUST be carried over. Symptom to assert against: a
   monotonically non-decreasing `best_val` across seeds indicates the bug is back.
3. **`n=15` seeds** in the final stage, matching the canonical probe and ResNet seed-paired
   design — *not* the 5 seeds of the full-data control. The contrast is seed-paired; the
   seed count must match.
4. **Matched search budget (§5).** A fixed recipe is explicitly *not* admissible.

## 4. The design trap (why §5 exists)

The full-data control's recipe is 30 epochs × batch 32 over 1295 train images ≈ **1200
optimizer steps**, and **measures ~1000 s per seed** (976–1524 s), dominated by per-epoch
validation + test evaluation. Transferred unchanged:

| split | images | steps/epoch | 30-epoch total |
|---|---|---|---|
| full | 1295 | 40 | ~1200 |
| K=10 | 160 | 5 | **150** |
| K=5 | 80 | 2 | **60** |

A 428M encoder given 60 steps at `vis_lr=5e-6` will barely move. Extending the full-data
recipe unchanged would manufacture a **strawman** and license the false conclusion
"fine-tuning CLIP does not work at low K". This must not happen.

## 5. Matched search budget (two-stage)

Both backbones get an **equal-size** grid, selected by best validation accuracy on the same
138-image validation split. Equal size is the fairness requirement; equal grid *shape* is not
required (the encoders differ in depth and native LR tolerance).

- **CLIP**: `epochs ∈ {30, 100, 300}` × `vis_lr ∈ {5e-6, 2e-5, 5e-5}` → 9 configurations.
- **ResNet-18**: an **equally sized** 9-configuration grid over its own space
  (`lr ∈ {3e-4, 1e-3, 3e-3}` × `epochs ∈ {30, 100, 300}`), same best-val selection.

**Stage 1 — search (n=3 seeds: 0,1,2).** All 9 configs. Each config's score is the **mean
best-validation accuracy across its 3 seeds**. Configuration selection uses **validation
only**; test accuracy is recorded but MUST NOT be used to choose.

**Stage 2 — final (n=15 seeds: 0…14).** The single stage-1 winner is re-run from scratch with
all 15 seeds. This is the number reported.

**Evaluation schedule (cost control, A3).** Validation is evaluated every **10** epochs, not
every epoch; **test is evaluated exactly once per run**, at the selected best-val epoch. This
is required to keep a 300-epoch `K=5` run at ~5 minutes instead of ~30. Selection remains
best-val; only the *frequency* of the check changes, and 10-epoch granularity is recorded as
a protocol limitation.

## 6. Pre-committed decision rules

Fixed now, before any result is seen, so the outcome cannot be rationalised after the fact.
`R*` denotes the **matched-budget** ResNet-18 mean at K=5 from §5 (NOT the legacy 65.5%).

- **Branch A — fine-tuned LAION-2B @ K=5 ≥ `R*`.**
  The crossover bracket `(2,5)` is void; CLIP leads at every measured `K` under matched
  adaptation. **Required action: retitle and restructure.** The "regime-dependent ordering"
  framing becomes a statement about the *frozen-probe protocol*, not about the encoders.
  This must NOT be filed as a robustness footnote.
- **Branch B — fine-tuned LAION-2B @ K=5 < `R*`.**
  The regime ordering survives matched adaptation. **Required action:** add the row and one
  sentence; the abstract's "restricted usage mode" concession (L30) is upgraded from
  concession to evidence.
- **Branch C — ambiguous:** the bootstrap 95% CI of the **seed-paired mean difference**
  (CLIP − ResNet, paired on the same 15 support draws) spans 0.
  **Required action:** report as a **tie**, explicitly *not* a resolution, and state that the
  138-image test set is underpowered at this cell (the paper already argues this for the
  full-data cell). No directional claim in either direction.

In every branch the per-seed values for both backbones are reported, so the reader can
re-derive the classification.

## 7. Multiplicity

The existing family is 8 contrasts (`K ∈ {1,2,5,10}` × 2 checkpoints). P3b adds 4
(`K ∈ {5,10}` × 2). **Pre-committed:** Holm–Bonferroni and Benjamini–Hochberg are recomputed
over the expanded family of **12**, and both the old-8 and new-12 results are reported. No
post-hoc choice of family size.

## 8. Honesty guardrails

- Not in scope, and already disclosed at L553: the CLIP image–text objective, LoRA/adapters.
  P3b is encoder + linear head only, matching the full-data control.
- Per-seed values and SD are reported for every cell. The M0 gate already recorded DFN-CLIP
  **overfitting at 10-shot** (51.5%, −6.5 pp), so instability is expected and must be
  reported, never smoothed.
- **If CLIP fine-tuning at K=5 collapses (e.g. < 50%), that is an optimisation-stability
  finding, not evidence that CLIP cannot adapt at low K.** It must be reported as such,
  together with the full sweep grid showing what was tried. A single collapsed run
  generalised to "CLIP fails at low K" is a forbidden reading.

## 9. Artifacts

| item | path |
|---|---|
| CLIP runner (committed, re-runnable) | `scripts/clip_finetune_kshot.py` |
| ResNet matched-budget runner | `scripts/resnet_kshot_matched_budget.py` |
| CLIP output | `data/p3b_clip_finetune_kshot.json` |
| ResNet output | `data/p3b_resnet_kshot_matched.json` |

Gate checks in `scripts/check_tex_vs_data.py` are extended **only after** the numbers exist
and are written into the `.tex` — never before. No claim enters the manuscript ahead of its
data.

## 10. Definition of done (scope A5)

1. Matched-budget ResNet-18 @ K=5 (stage 1 n=3 → stage 2 n=15) → `R*` written to
   `data/p3b_resnet_kshot_matched.json` with per-seed values.
2. Matched-budget LAION-2B @ K=5 (stage 1 n=3 → stage 2 n=15) →
   `data/p3b_clip_finetune_kshot.json` with per-seed values.
3. Seed-paired bootstrap 95% CI of the mean difference computed over the 15 shared draws;
   branch assigned per §6 **by the pre-committed rule, not by inspection**.
4. Per A4: if `R*` differs materially from the legacy 65.5%, propagate the new value through
   L249, Table 1 and Figure 1, re-run the gate to 189/189 + the new checks, and recompile
   clean.
5. Decision log entry in `EXPERIMENT_TRACKER.md`, recording the assigned branch and the
   numbers that justified it.
6. Deferred cells (DFN @ K=5/10, LAION @ K=10) recorded as deferred with reason.