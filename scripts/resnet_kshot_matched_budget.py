"""
P3b stage-1/2 -- ResNet-18 K-shot under a MATCHED adaptation budget (TPDD).

Why this exists
---------------
The paper's K-shot ResNet-18 rows (data/paired_kshot_a4.json -> 65.5% at K=5, 75.7% at
K=10) come from a SINGLE configuration: "ResNet-18 ImageNet-pretrained, AdamW lr=3e-4,
cosine, bs=32, 30ep, best-val". No hyper-parameter search was run. P3b compares a
fine-tuned CLIP against ResNet-18 and calls the budgets "matched"; that claim is only
defensible if ResNet-18 also gets a search of the same SIZE. This script supplies it.

User decision (2026-10-16, pre-run): the matched-budget ResNet value REPLACES the legacy
single-config value in the manuscript, and is propagated through L249 / Table 1 / Figure 1.

Protocol (see refine-logs/P3B_KSHOT_PREREGISTRATION.md sections 2.1, 5, 6)
------------------------------------------------------------------------
* Support indices come from coop_experiment.kshot_subset_idx(y, K, seed+500), which was
  verified index-identical to paired_kshot_a4.KShotSubset. Asserted at run time.
* scripts/exp_resnet_k5_k10_n15.py (seed+300) MUST NOT be used -- superseded, and its
  output resnet_k5_n15.json is already labelled non-authoritative by the gate.
* Stage 1: all 9 configs, n=3 seeds (0,1,2). Score = mean best-VAL accuracy. Selection is
  on VALIDATION ONLY; test accuracy is recorded but never used to choose.
* Stage 2: the stage-1 winner re-run from scratch with n=15 seeds (0..14).
* Validation every VAL_EVERY epochs; test evaluated exactly once, at the best-val epoch.

Output: data/p3b_resnet_kshot_matched.json
"""
import os
import sys
import json
import time
import argparse
import importlib.util

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn  # noqa: E402
from model_zoo import ResNetBaseline, set_seed  # noqa: E402
from coop_experiment import kshot_subset_idx  # noqa: E402

NC = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED_OFFSET = 500
VAL_EVERY = 10

# 9 configurations -- deliberately the same COUNT as the CLIP grid, which is the
# fairness requirement. The grid SHAPE differs because a 12M CNN and a 428M ViT do not
# share a sensible LR range; equal size, not equal shape.
LR_GRID = [3e-4, 1e-3, 3e-3]
EPOCH_GRID = [30, 100, 300]


class IndexSubset(Dataset):
    """K-shot subset of a TPDDDataset, selected by explicit indices."""

    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # TPDDDataset.__getitem__ returns a dict (image / label / class_name / kg_*),
        # not a (img, label) tuple. Pass the dict straight through.
        return self.base[int(self.indices[i])]


def verify_draw_matches_canonical(train_ds, K, seeds):
    """Abort unless our draw equals the one behind the published ResNet K-shot rows.

    paired_kshot_a4.py defines its own KShotSubset class. It was checked equal to
    coop_experiment.kshot_subset_idx offline; this re-checks it at run time so the
    seed-pairing claim cannot silently rot if either file changes.
    """
    spec = importlib.util.spec_from_file_location(
        "_pka", os.path.join(REPO, "scripts", "paired_kshot_a4.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    labels = np.asarray(train_ds.label_ids)
    for s in seeds:
        mine = np.sort(kshot_subset_idx(labels, K, s + SEED_OFFSET))
        theirs = np.sort(np.asarray(mod.KShotSubset(train_ds, K, s + SEED_OFFSET).indices))
        assert mine.shape == theirs.shape and bool((mine == theirs).all()), (
            f"K={K} seed={s}: support draw differs from paired_kshot_a4.py. The "
            f"seed-paired comparison would be void. Refusing to run.")
    return True


def build_loaders(K, batch):
    train_full = TPDDDataset("train", transform=TRAIN_TRANSFORM)
    val_full = TPDDDataset("validation", transform=EVAL_TRANSFORM)
    test_full = TPDDDataset("test", transform=EVAL_TRANSFORM)
    verify_draw_matches_canonical(train_full, K, [0, 1, 2, 7, 14])
    labels = np.asarray(train_full.label_ids)

    def loader_for(seed):
        idx = kshot_subset_idx(labels, K, seed + SEED_OFFSET)
        sub = IndexSubset(train_full, idx)
        return (DataLoader(sub, batch_size=min(batch, len(sub)), shuffle=True,
                           num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn),
                DataLoader(val_full, batch_size=64, shuffle=False, num_workers=0, pin_memory=True),
                DataLoader(test_full, batch_size=64, shuffle=False, num_workers=0, pin_memory=True),
                len(sub))

    return loader_for


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = tot = 0
    for batch in loader:
        logits = model(batch["image"].to(DEVICE, non_blocking=True))
        pred = logits.argmax(1)
        lab = batch["label"].to(DEVICE, non_blocking=True)
        correct += int((pred == lab).sum())
        tot += int(lab.numel())
    return correct / max(tot, 1)


def run_one(K, lr, epochs, seed, batch, make_loader):
    set_seed(seed)
    train_dl, val_dl, test_dl, n_support = make_loader(seed)
    model = ResNetBaseline(num_classes=NC).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_val, best_test, best_ep = -1.0, -1.0, -1
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(batch["image"].to(DEVICE, non_blocking=True)),
                        batch["label"].to(DEVICE, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        if (ep + 1) % VAL_EVERY == 0 or ep == epochs - 1:
            val_acc = evaluate(model, val_dl)
            if val_acc > best_val:
                best_val = val_acc
                best_ep = ep + 1
                best_test = evaluate(model, test_dl)
    return {"seed": seed, "lr": lr, "epochs": epochs, "best_val": round(float(best_val), 4),
            "test": round(float(best_test), 4), "best_epoch": best_ep,
            "n_support": n_support, "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--search-seeds", type=int, default=3)
    ap.add_argument("--final-seeds", type=int, default=15)
    ap.add_argument("--out", type=str,
                    default=os.path.join(REPO, "data/p3b_resnet_kshot_matched.json"))
    args = ap.parse_args()

    K = args.k
    make_loader = build_loaders(K, args.batch)
    grid = [{"lr": lr, "epochs": ep} for lr in LR_GRID for ep in EPOCH_GRID]
    assert len(grid) == 9, "grid size is the fairness contract; do not change it silently"

    t0 = time.time()
    print(f"Device={DEVICE} K={K} grid={len(grid)} "
          f"search_seeds={args.search_seeds} final_seeds={args.final_seeds}", flush=True)

    # ---- Stage 1: search. Selection on VALIDATION ONLY. ----
    stage1 = []
    for cfg in grid:
        runs = [run_one(K, cfg["lr"], cfg["epochs"], s, args.batch, make_loader)
                for s in range(args.search_seeds)]
        score = float(np.mean([r["best_val"] for r in runs]))
        stage1.append({**cfg, "mean_best_val": round(score, 4),
                       "mean_test": round(float(np.mean([r["test"] for r in runs])), 4),
                       "runs": runs})
        print(f"  cfg lr={cfg['lr']:<7g} ep={cfg['epochs']:<4d} "
              f"val={score*100:.2f}%  (test {stage1[-1]['mean_test']*100:.2f}%, "
              f"not used for selection)", flush=True)

    best = max(stage1, key=lambda c: c["mean_best_val"])
    print(f"  -> stage-1 winner: lr={best['lr']} epochs={best['epochs']} "
          f"val={best['mean_best_val']*100:.2f}%", flush=True)

    # ---- Stage 2: the winner, re-run from scratch with the full seed count. ----
    final_runs = [run_one(K, best["lr"], best["epochs"], s, args.batch, make_loader)
                  for s in range(args.final_seeds)]
    accs = [r["test"] for r in final_runs]
    res = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "arch": "ResNetBaseline (ResNet-18, ImageNet-pretrained)",
        "K": K,
        "n_classes": NC,
        "seed_offset": SEED_OFFSET,
        "support_draw": "coop_experiment.kshot_subset_idx(y, K, seed+500); asserted "
                        "index-identical to paired_kshot_a4.KShotSubset at run time",
        "val_every_epochs": VAL_EVERY,
        "grid": grid,
        "stage1_search": stage1,
        "selected": {"lr": best["lr"], "epochs": best["epochs"],
                     "mean_best_val": best["mean_best_val"]},
        "stage2_final": {
            "n_seeds": args.final_seeds,
            "per_seed": final_runs,
            "accs": [round(a, 6) for a in accs],
            "mean": round(float(np.mean(accs)), 6),
            "std": round(float(np.std(accs, ddof=1)), 6) if len(accs) > 1 else 0.0,
            "sem": round(float(np.std(accs, ddof=1) / np.sqrt(len(accs))), 6) if len(accs) > 1 else 0.0,
        },
        "legacy_single_config_ref": {
            "value_pct": {1: 33.04, 2: 43.38, 5: 65.46, 10: 75.75}[K],
            "source": ("K=1: data/paired_kshot_a4_both.json; K=2: data/paired_kshot_a4_k2.json; "
                       "K=5/10: data/paired_kshot_a4.json (all: lr=3e-4, 30ep, NO hyper-parameter search)"),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== ResNet-18 K={K} matched budget ===")
    print(f"  stage-2 mean = {res['stage2_final']['mean']*100:.2f}% +/- "
          f"{res['stage2_final']['std']*100:.2f}% (n={args.final_seeds})")
    print(f"  legacy single-config reference = "
          f"{res['legacy_single_config_ref']['value_pct']:.2f}%")
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()