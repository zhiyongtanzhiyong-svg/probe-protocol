# Provenance record — few-shot caliber unification + K=2 completion (2026-10-05)

## What was wrong

Two defects let one nominal method ("the DFN-CLIP probe") carry two accuracy rows.

1. **Support transform.** `scripts/clip_fewshot_probe.py` built the K-shot support
   pool with `TRAIN_TRANSFORM` (RandomHorizontalFlip / RandomRotation /
   ColorJitter / RandomResizedCrop) and encoded it ONCE per seed, because
   sklearn's `LogisticRegression` is a single convex fit with no epoch loop.
   Augmentation therefore bought nothing; it injected one stochastic draw of
   noise. Consequence: the full-data row had seed sd 1.79, contradicting the
   Table 2 caption's claim that "the frozen-probe rows have zero seed variance
   because logistic regression on fixed features is deterministic".
   The fine-tuned ResNet arm keeps `TRAIN_TRANSFORM` because it trains 30
   epochs, where the augmentation is genuine.

2. **Two encodings.** `clip_features_clean.npz` (legacy) differs from
   `canonical_clip_features_dfn.npz` by max |diff| = 0.1185 on L2-normalized
   features and yields full-data C=1 = 76.09% instead of 76.81%. It is read by
   lp_plus_plus, coop_experiment, zpe_experiment, heads_a7, align_a6,
   why_vlms_fail_e1e2, canonical_fig_artifacts, rigorous_analysis,
   exp_full_provenance, verify_c2_probes. The Table 2 caption already disclosed
   the split and scoped head-capacity claims within-artifact; that disclosure
   was sound and is retained. The appendix now also states it explicitly.

**Refuted hypothesis:** the two scripts drew different support indices. They do
not. `data/tpdd/*/labels.json` is class-sorted, so the dict-insertion order used
by the `KShotSubset` classes equals the `sorted()` order used by
`coop_experiment.kshot_subset_idx` and `two_model_e4.kshot_idx`. Verified
identical at K in {1,2,5,10} for seeds {0,1,7}.

## What changed

| Artifact | Status |
|---|---|
| `scripts/clip_fewshot_probe.py` | rewritten: EVAL support, one encoding, index subsetting, gate against canonical cache |
| `scripts/verify_cache_freshness.py` | new — proves the legacy cache is drifted |
| `scripts/paired_kshot_a4_k2.py` | new — the missing K=2 seed-paired ResNet run |
| `scripts/multiplicity_k2.py` | new — 8-contrast family with K=2 |
| `data/clip_fewshot.json` (+ `_fixed`) | regenerated, single caliber |
| `data/fewshot_features_dfn.npz` | new — clean DFN features, verified max|diff| = 0 vs canonical |
| `data/paired_kshot_a4_k2.json` | new |
| `data/multiplicity_k2.json` | new |
| `data/cache_freshness.json` | new |
| `data/clip_fewshot.json.pre_kshot_caliber_fix` | pre-change snapshot |
| `scripts/check_tex_vs_data.py` | +18 single-caliber checks |
| `manuscript/paper/clip_vs_resnet_tobacco.tex` | Table 2 (K=2 column, DFN row), abstract, claims (2), Fig 1 caption, Results, RQ2, Discussion, Conclusion, Appendix A |
| `manuscript/paper/figures/*` | regenerated from the JSON artifacts |

## Corrected numbers

DFN-CLIP probe (canonical, single caliber):

| K | before | after | old SD | new SD |
|---|---|---|---|---|
| 1 | 31.06 | 30.48 | 5.31 | 2.44 |
| 2 | 37.97 | 36.52 | 4.94 | 4.02 |
| 5 | 51.30 | 47.97 | 5.31 | 5.50 |
| 10 | 58.31 | 56.43 | 3.13 | 3.58 |
| full | 75.46 (sd 1.79) | 76.81 (sd 0.0) | 1.79 | 0.00 |

Seed-paired contrasts (family of eight; Holm and BH both keep 8/8):

| K | LAION-2B | DFN-CLIP |
|---|---|---|
| 1 | +7.05 (15/0/0) p=1.5e-4 | **−2.56 (3/3/9) p=0.031** ← was −1.98, p=0.14, non-significant |
| 2 | **+5.46 (12/0/3) p=0.011** | −6.86 (1/0/14) p=2.2e-4 |
| 5 | −7.05 (3/1/11) p=0.001 | −17.49 (0/0/15) p=2.0e-8 |
| 10 | −10.72 (0/0/15) p=1.5e-7 | −19.32 (0/0/15) p=2.7e-11 |

## Substantive consequence

The LAION-2B crossover is now bracketed to the open interval (2, 5) rather than
(1, 5). DFN-CLIP shows no crossover. The K=1 DFN contrast changes status from
non-separable to significant, so "the DFN probe does not differ from the CNN at
K=1" is withdrawn and replaced by "DFN trails at every measured K". This makes
claim (2) simpler, not weaker.

## Not done, and why

The stale `clip_features_clean.npz` is still on disk and still read by ten
scripts. Re-running the LP++ / CoOp / ZPE / heads_a7 arms against the canonical
encoding would change their absolute numbers by roughly 1 pp and was NOT done
here: it is a multi-hour GPU cascade that would also require a fresh LAION
encoding (none is cached) and a rebuild of the 200-view CoOp augmentation store
(`clip_aug_views_*.npz`, 368 MB each, built from the pre-rebuild images). Those
experiments are consistent within themselves and the paper scopes their claims
within-artifact, so the manuscript is internally consistent as it stands. This
is a revision-stage item, not a blocker for the corrected probe caliber.

## Verification

- `python3 scripts/check_tex_vs_data.py` → 155/155 consistent, exit 0
- `latexmk -pdf` → 31 pages, 0 errors, 0 undefined references
- Overfull hboxes > 14pt: 4, all pre-existing (the `\dataavailability`
  placeholder at line 575 and a bibliography DOI at lines 681–685)

---

# Part 2 — Feature-store unification (2026-10-05, continued)

## What the mtime audit and direct measurement showed

34 of 63 `data/*.json|npz` artifacts predate the corpus re-derivation, but
mtime is not evidence: all 1571 images share one mtime (2026-09-29 18:49) yet
caches written 2026-09-27/28 reproduce today's pixels at max |diff| = 0.0000, so
the images were evidently re-copied without changing content. Only direct
measurement was used.

`scripts/audit_feature_stores.py` (new) measured each store against a fresh
canonical extraction:

| Store | mtime | max abs diff | full-data C=1 | verdict |
|---|---|---|---|---|
| `canonical_clip_features_dfn.npz` | 09-28 | 0.000000 | 76.81% | canonical |
| `clip_features_laion_clean.npz` | 09-27 | 0.000000 | 76.09% | current |
| `clip_features_clean.npz` (DFN) | **08-29** | **0.1185** | 76.09% | **drifted** |
| `clip_aug_views_dfn.npz` | 09-27 | n/a (stochastic) | — | current images |
| `clip_aug_views_laion.npz` | 09-27 | n/a | — | current images |

The augmented-view stores were tested rather than assumed: averaging 20 views
per image to suppress augmentation noise, the views sit closer to the canonical
features than to the drifted ones on 1016/1295 images, with a uniformly
positive margin (+0.0149 on the 0.80–0.95 stratum, +0.0185 on the >0.95
stratum). That is the signature of views built from the current images; views
from the old set would show a strongly negative margin.

**Only one store was stale**, and it was the DFN one.

## Two further defects found

**1. `scripts/lp_plus_plus.py` could not execute.** Line 304 asserted
`np.array_equal(y, y_full[:len(y)])` but the array is bound as
`train_labels_full` at line 270. `y_full` is a NameError, so any run raised
immediately. Fixed, plus the loader now normalises three different npz key
conventions. A 2-seed/3-epoch smoke test was run before the full job, per
RESEARCH_PROTOCOL 规程 0.

**2. `scripts/two_model_e4.py` is self-reproducible but not on the canonical
path.** Re-running it reproduces `data/two_model_e4.json` to the digit
(40.1 / 48.8 / 58.4 / 65.0 / 75.4). But its full-data probe is **75.36%** where
`canonical_results.json` and the frozen store both give **76.09%** for the same
LAION checkpoint, same images, same classifier. `scripts/laion_fewshot_canonical.py`
(new) recomputes the LAION few-shot probe from the frozen store.

## Ruled out

- **Support draws.** `two_model_e4.kshot_idx` and
  `coop_experiment.kshot_subset_idx` produce byte-identical indices at
  K in {1,2,5,10} for all 15 seeds. Not the cause.
- **Preprocessing.** The two `EVAL` transforms, the `tensor_to_pil` round trip,
  and the resulting tensors are bit-identical on a test image. Not the cause.
- **Encoder jitter.** Feature extraction is bit-identical across batch sizes
  16/32/64 within a process (`data/encoder_jitter.json`, range 0.00 pp) and
  bit-identical across two separate processes (max |diff| 0.000000). The
  encoder is deterministic. Not the cause.
- An intermediate one-off comparison reported a 0.065 feature difference that
  could not be reproduced by any subsequent run; it is not relied on.

## Resulting numbers

LAION-2B probe, frozen canonical store (supersedes `two_model_e4.json`):
40.14 / 48.79 / 60.19 / 66.47 / 76.09 (was 40.10 / 48.84 / 58.41 / 65.02 / 75.36).

Seed-paired contrasts, family of eight, **8/8 survive Holm and BH**:

| K | LAION-2B | DFN-CLIP |
|---|---|---|
| 1 | +7.10 CI[+5.07,+9.18] 14/0/1 | −2.56 CI[−4.69,−0.63] 3/3/9 |
| 2 | +5.41 CI[+1.79,+8.89] 11/1/3 | −6.86 CI[−9.52,−4.30] 1/0/14 |
| 5 | −5.27 CI[−8.60,−2.03] 3/0/12 | −17.49 CI[−20.48,−14.68] 0/0/15 |
| 10 | −9.28 CI[−10.92,−7.58] 0/0/15 | −19.32 CI[−21.26,−17.44] 0/0/15 |

LP++ re-run on the frozen stores: DFN 32.8 / 43.9 / 60.4 / 71.1 / 83.3 (was
32.8 / 44.5 / 62.5 / 71.8 / 81.2); LAION unchanged at 40.8 / 51.7 / 66.2 /
74.2 / 84.1, which is itself the confirmation that the LAION store was already
current. LP++'s probe reference now equals the canonical row to within 0.01 pp.

No conclusion changed. The crossover bracket (2,5) and the 8/8 multiplicity
result both survive; the $K=1$ DFN contrast remains significant.

## Still open

CoOp was **not** re-run. Its augmented views are current but its clean evaluation
features come from the drifted store, so its probe reference (30.5/49.1/57.7) is
retained and all CoOp claims are scoped within its own artifact. Re-running
CoOp is ~2.2 h of GPU (200 epochs x 15 seeds x 2 checkpoints) and would move the
DFN full-data CoOp-vs-probe margin from +2.2 pp to about +1.5 pp; the direction
would not change. This is a revision-stage item.

Also unresolved: the provenance of `clip_features_laion_clean.npz` (written
2026-09-27) and of `clip_features_clean.npz` (2026-08-29) is not recorded
anywhere in the repository, and neither file is in git. The measured drift is
documented above, but which script produced the drifted store is unknown.

## Verification

- `python3 scripts/check_tex_vs_data.py` → 159/159 consistent, exit 0
- `latexmk -pdf` → 31 pages, 0 errors, 0 undefined references
- Overfull hboxes > 14pt: still 4, all pre-existing and unchanged
