#!/usr/bin/env python3
"""Paired per-image statistics: zero-shot best template vs fixed majority-class predictor.

Reviewer item (13th review, R1-2): the model-vs-majority-baseline comparison acts on the
same test images, so the per-image correctness outcomes are paired. This script computes,
for each checkpoint whose per-image zero-shot predictions are archived in
data/zero_shot_full_20261025.json:

  - per-image agreement structure with the fixed majority-class predictor
    (b = model-right/baseline-wrong, c = model-wrong/baseline-right),
  - exact McNemar test (binomial on the discordant pairs),
  - paired accuracy difference with a 10,000-resample paired bootstrap 95% CI,
  - the observed per-image disagreement rate p_d (input to the power analysis,
    Methods "Power (a-priori)").

Validation gate: recomputed per-template accuracies from the archived predictions and the
canonical test labels must match the archived aggregate accuracies for ALL 42
model-template pairs; any mismatch aborts (the archive would not be self-consistent).

Caveat recorded in the output: the tested "best template" was selected on this same test
set, so the paired statistics are descriptive (post-selection), not confirmatory. They
show that the negative conclusion (no template exceeds the baseline) does not depend on
the marginal-vs-paired choice of test.

Usage:  python3 scripts/paired_zs_baseline.py
Output: data/paired_zs_baseline_20261031.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parent.parent
ZS = ROOT / "data" / "zero_shot_full_20261025.json"
NPZ = ROOT / "data" / "canonical_clip_features_dfn.npz"
OUT = ROOT / "data" / "paired_zs_baseline_20261031.json"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: binomial test of b vs c under P(=)=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    p = 2.0 * binom.cdf(lo, n, 0.5)
    return float(min(1.0, p))


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int =10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = diffs[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def main():
    zs = json.load(open(ZS))
    y = np.load(NPZ)["test_y"].astype(int)
    n = len(y)
    counts = np.bincount(y, minlength=16)
    maj = int(counts.argmax())
    n_maj = int(counts.max())
    print(f"test labels: n={n}, majority class idx={maj} (n={n_maj}, "
          f"{n_maj / n * 100:.2f}%) [paper baseline 13.04% = 18/138]")
    assert abs(n_maj / n * 100 - 13.04) < 0.05, "majority baseline mismatch vs archived JSON"

    correct_maj = (y == maj)

    out = {
        "source": str(ZS.name),
        "labels_source": str(NPZ.name) + " test_y (canonical DFN store; same test split manifest for all checkpoints)",
        "n_test": n,
        "majority_class_idx": maj,
        "majority_baseline_pct": round(n_maj / n * 100, 2),
        "caveat": "the tested best template was selected on this same test set; paired statistics are descriptive (post-selection), not confirmatory",
        "checkpoints": {},
    }

    n_mismatch = 0
    for ck, cv in zs["checkpoints"].items():
        if ck == "siglip_error" or "templates" not in cv:
            continue
        # --- validation gate: recompute every template's accuracy from archived preds ---
        for tname, t in cv["templates"].items():
            preds = np.asarray(t["preds"], dtype=int)
            assert len(preds) == n, f"{ck}/{tname}: preds length {len(preds)} != {n}"
            nc = int((preds == y).sum())
            arch_nc = t.get("n_correct")
            arch_acc = t.get("acc_pct")
            ok = (arch_nc == nc) and (arch_acc is None or abs(arch_acc - nc / n * 100) < 0.05)
            if not ok:
                n_mismatch += 1
                print(f"  MISMATCH {ck}/{tname}: recomputed {nc}/{n} vs archived "
                      f"{arch_nc}/{n} ({arch_acc}%)")
        if n_mismatch:
            print(f"ABORT: {n_mismatch} template accuracies do not match the archive; "
                  "label/prediction pairing is not confirmed.")
            sys.exit(1)

        # --- best template (as archived) ---
        best_t = cv["best_template"]
        preds = np.asarray(cv["templates"][best_t]["preds"], dtype=int)
        correct_m = (preds == y)
        b = int((correct_m & ~correct_maj).sum())   # model right, baseline wrong
        c = int((~correct_m & correct_maj).sum())   # model wrong, baseline right
        p = mcnemar_exact(b, c)
        diffs = (correct_m.astype(int) - correct_maj.astype(int))
        mean_d, lo, hi = paired_bootstrap_ci(diffs)
        acc_m = float(correct_m.mean())
        out["checkpoints"][ck] = {
            "best_template": best_t,
            "best_acc_pct": round(acc_m * 100, 2),
            "paired_vs_majority": {
                "b_model_right_baseline_wrong": b,
                "c_model_wrong_baseline_right": c,
                "discordant_pct": round((b + c) / n * 100, 1),
                "mcnemar_exact_p_two_sided": round(p, 4),
                "paired_diff_pct": round(mean_d * 100, 2),
                "paired_bootstrap_ci95_pct": [round(lo * 100, 2), round(hi * 100, 2)],
                "p_d_observed": round((b + c) / n, 4),
            },
        }
        r = out["checkpoints"][ck]["paired_vs_majority"]
        print(f"{ck}: best={best_t} ({out['checkpoints'][ck]['best_acc_pct']}%)  "
              f"b={b} c={c} McNemar p={r['mcnemar_exact_p_two_sided']}  "
              f"paired diff={r['paired_diff_pct']}pp CI{r['paired_bootstrap_ci95_pct']}  "
              f"p_d={r['p_d_observed']}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")
    print("NOTE: SigLIP has no per-image predictions archived (aggregate-only JSON); "
          "its p=0.0025 remains nominal/descriptive in the text.")


if __name__ == "__main__":
    main()
