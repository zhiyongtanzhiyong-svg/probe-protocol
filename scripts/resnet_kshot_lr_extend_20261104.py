#!/usr/bin/env python3
"""
M2 (Round-18 external review follow-up): ResNet-18 K-shot, EXTENDED low-LR sweep.

Why
---
Limitation (10) of the manuscript states: the ResNet matched-grid stage-1 surface is
steep in learning rate, the selected configuration sits at the grid's lowest LR (3e-4)
with validation accuracy still rising; "an extended sweep below 3e-4 was not run, as it
would break the matched 9-config budget." If ResNet's true optimum lies below 3e-4,
part of the matched-budget fine-tuning margin may reflect grid coverage rather than
adaptation budget. This script runs exactly that extension, clearly labelled
EXPLORATORY (outside the matched 9-config budget):

* New configurations: LR in {3e-5, 1e-4} x epochs {30, 100, 300} = 6 configs, one
  decade below the existing grid floor (3e-4), same epoch grid, same selection rule
  (mean best-VAL over search seeds 0-2 on the 138-image validation split).
* Stage-1 values for the ORIGINAL 9 configs are taken from the saved p3b stage-1
  results (same protocol, same split, deterministic selection rule), so the union
  selection over 15 configs is exact without re-burning the old grid.
* Stage 2: if (and only if) the union winner is a NEW config, it is re-run from
  scratch with n=15 seeds (0..14) -- identical protocol to the matched stage-2.
  If the union winner is an old config, the published matched result stands and no
  stage-2 re-run is performed.

The support draw, transforms, optimizer (AdamW wd=0.01, cosine, grad-clip 1.0),
batch size, VAL_EVERY, and seed offsets are all inherited from
scripts/resnet_kshot_matched_budget.py by direct import, so the new runs are
protocol-identical to the matched runs except for the LR values.

Outputs: data/resnet_kshot_lr_extend_{K}_20261104.json per K (K in 1,2,5,10)
"""
import os
import sys
import json
import time
import importlib.util

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Import the matched-budget script as a module (its main() is guarded).
_spec = importlib.util.spec_from_file_location(
    "rb", os.path.join(REPO, "scripts", "resnet_kshot_matched_budget.py"))
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

# One decade below the matched grid floor, same ratio pattern (3e-5, 1e-4, | 3e-4, 1e-3, 3e-3).
NEW_LR = [3e-5, 1e-4]
NEW_GRID = [{"lr": lr, "epochs": ep} for lr in NEW_LR for ep in rb.EPOCH_GRID]
SEARCH_SEEDS = 3
FINAL_SEEDS = 15

P3B_FILES = {
    1: os.path.join(REPO, "data/p3b_resnet_kshot_matched_k1.json"),
    2: os.path.join(REPO, "data/p3b_resnet_kshot_matched_k2.json"),
    5: os.path.join(REPO, "data/p3b_resnet_kshot_matched.json"),
    10: os.path.join(REPO, "data/p3b_resnet_kshot_matched_k10.json"),
}
P3B_PUBLISHED = {1: 40.4, 2: 52.5, 5: 64.9, 10: 74.6}  # % (stage-2 means)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="K value (default: all of 1,2,5,10)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    Ks = [args.k] if args.k else [1, 2, 5, 10]
    for K in Ks:
        t0 = time.time()
        out = os.path.join(REPO, f"data/resnet_kshot_lr_extend_{K}_20261104.json")
        make_loader = rb.build_loaders(K, args.batch)

        # ---- Stage 1: only the 6 NEW configs, same 3 search seeds. ----
        print(f"\n########## K={K}: extended low-LR stage-1 ({len(NEW_GRID)} new configs x {SEARCH_SEEDS} seeds) ##########", flush=True)
        new_stage1 = []
        for cfg in NEW_GRID:
            runs = [rb.run_one(K, cfg["lr"], cfg["epochs"], s, args.batch, make_loader)
                    for s in range(SEARCH_SEEDS)]
            score = float(np.mean([r["best_val"] for r in runs]))
            new_stage1.append({**cfg, "mean_best_val": round(score, 4),
                               "mean_test": round(float(np.mean([r["test"] for r in runs])), 4),
                               "runs": runs})
            print(f"  NEW cfg lr={cfg['lr']:<6g} ep={cfg['epochs']:<4d} "
                  f"val={score*100:.2f}%  (test {new_stage1[-1]['mean_test']*100:.2f}%, "
                  f"not used for selection)", flush=True)

        # ---- Union selection over the saved old 9 + new 6. ----
        p3b = json.load(open(P3B_FILES[K]))
        old_stage1 = [{**c, "runs": c.get("runs", [])} for c in p3b["stage1_search"]]
        union = old_stage1 + new_stage1
        best = max(union, key=lambda c: c["mean_best_val"])
        is_new = any(abs(c["lr"] - best["lr"]) < 1e-12 and c["epochs"] == best["epochs"]
                     for c in new_stage1)
        print(f"  union winner: lr={best['lr']} ep={best['epochs']} val={best['mean_best_val']*100:.2f}% "
              f"(NEW={is_new})", flush=True)
        print(f"  published matched: lr={p3b['selected']['lr']} ep={p3b['selected']['epochs']} "
              f"val={p3b['selected']['mean_best_val']*100:.2f}% -> stage2 {P3B_PUBLISHED[K]}%", flush=True)

        res = {
            "date": time.strftime("%Y-%m-%d"),
            "gpu": rb.torch.cuda.get_device_name(0) if rb.torch.cuda.is_available() else "cpu",
            "torch": rb.torch.__version__,
            "purpose": ("M2 exploratory follow-up (Round-18 review): extended low-LR sweep below "
                        "the matched grid floor 3e-4; OUTSIDE the matched 9-config budget; "
                        "selection over the 15-config union uses the identical validation-only rule"),
            "K": K,
            "new_grid": NEW_GRID,
            "new_stage1": [{k: v for k, v in c.items() if k != "runs"} for c in new_stage1],
            "union_top5": sorted(union, key=lambda c: -c["mean_best_val"])[:5],
            "union_winner": {"lr": best["lr"], "epochs": best["epochs"],
                             "mean_best_val": best["mean_best_val"], "is_new": is_new},
            "published_matched": {"lr": p3b["selected"]["lr"], "epochs": p3b["selected"]["epochs"],
                                  "stage2_mean_pct": P3B_PUBLISHED[K], "source": os.path.basename(P3B_FILES[K])},
        }

        if is_new:
            print(f"  -> stage-2: new winner re-run with {FINAL_SEEDS} seeds", flush=True)
            final_runs = [rb.run_one(K, best["lr"], best["epochs"], s, args.batch, make_loader)
                          for s in range(FINAL_SEEDS)]
            accs = [r["test"] for r in final_runs]
            res["stage2_new_winner"] = {
                "n_seeds": FINAL_SEEDS,
                "per_seed": final_runs,
                "accs": [round(a, 6) for a in accs],
                "mean": round(float(np.mean(accs)), 6),
                "std": round(float(np.std(accs, ddof=1)), 6),
            }
            res["change_vs_published_pp"] = round((float(np.mean(accs)) - P3B_PUBLISHED[K] / 100) * 100, 2)
            print(f"  stage-2 mean = {np.mean(accs)*100:.2f}% +/- {np.std(accs, ddof=1)*100:.2f}% "
                  f"(change vs published {res['change_vs_published_pp']:+.2f}pp)", flush=True)
        else:
            res["stage2_new_winner"] = None
            res["change_vs_published_pp"] = 0.0
            print(f"  -> union winner is the already-published config; matched result stands.", flush=True)

        res["elapsed_s"] = round(time.time() - t0, 1)
        json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
