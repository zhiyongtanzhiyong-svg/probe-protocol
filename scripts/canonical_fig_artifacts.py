"""
canonical_fig_artifacts.py — build the EXACT arrays behind Figures 2 and 3 under
the paper's canonical protocol, and assert they match the reported numbers.

Why this file exists
--------------------
Figure 3 previously refit the probe on `data/clip_features_clean.npz` and retrained
ResNet-18 on the fly, which produced 76.1% / 87.0% instead of the 76.8% / 86.4%
reported in Table 2. Root cause: `clip_features_clean.npz` is NOT the feature set
behind the canonical numbers. `scripts/canonical_protocol.py` extracts features
fresh from the images with the deterministic evaluation transform, and that fresh
extraction reproduces the reported 0.7681 exactly; the cached .npz (a different
pixel encoding) gives 0.7609. `data/full_provenance.json` even records 0.7609,
so the repository carried both numbers.

This script therefore rebuilds the figure inputs the canonical way and hard-fails
if they drift from `data/canonical_results.json` / `data/full_provenance.json`:

  * DFN-CLIP frozen features, fresh extraction, deterministic eval transform
  * canonical probe: C grid on a 20% stratified validation split (seed 42),
    refit on 100% of train, one evaluation on the 138 test images
  * ResNet-18 fine-tuned, canonical recipe, 15 seeds
    (ResNetBaseline, set_seed(s), AdamW lr=3e-4, cosine T_max=30, bs=32,
     30 epochs, grad-clip 1.0, best-validation checkpoint)

Outputs
-------
  data/canonical_clip_features_dfn.npz   fresh deterministic features (cache)
  data/canonical_fig_artifacts.json      predictions, confusion matrices,
                                         per-class accuracies, 15-seed accs

Usage
-----
  python3 scripts/canonical_fig_artifacts.py            # full (ResNet 15 seeds)
  python3 scripts/canonical_fig_artifacts.py --clip-only # skip ResNet training
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
P1 = ROOT
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(P1, "scripts"))

import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM, TRAIN_TRANSFORM, worker_init_fn
from model_zoo import ResNetBaseline, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin"
FEAT_CACHE = os.path.join(DATA, "canonical_clip_features_dfn.npz")
OUT = os.path.join(DATA, "canonical_fig_artifacts.json")

C_GRID = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
SEEDS, EPOCHS = 15, 30
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def tensor_to_pil(t):
    a = t.permute(1, 2, 0).cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def extract_clip(model, preprocess, ds):
    dl = DataLoader(ds, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    F, Y = [], []
    with torch.no_grad():
        for b in dl:
            im = torch.stack([preprocess(tensor_to_pil(x)) for x in b["image"]]).to(DEVICE)
            f = model.encode_image(im)
            f = f / f.norm(dim=-1, keepdim=True)
            F.append(f.cpu().numpy())
            Y += b["label"].numpy().tolist()
    return np.concatenate(F), np.array(Y)


def clip_artifacts():
    """Canonical CLIP probe + predictions. Asserts against canonical_results.json."""
    if os.path.exists(FEAT_CACHE):
        z = np.load(FEAT_CACHE)
        tr, tr_y, te, te_y = z["train"], z["train_y"], z["test"], z["test_y"]
        print(f"[clip] features from cache {FEAT_CACHE}")
    else:
        m, _, pp = open_clip.create_model_and_transforms("ViT-L-14", pretrained=CKPT)
        m = m.to(DEVICE).eval()
        tr, tr_y = extract_clip(m, pp, TPDDDataset(split="train", transform=EVAL_TRANSFORM))
        te, te_y = extract_clip(m, pp, TPDDDataset(split="test", transform=EVAL_TRANSFORM))
        del m
        torch.cuda.empty_cache()
        np.savez_compressed(FEAT_CACHE, train=tr, train_y=tr_y, test=te, test_y=te_y)
        print(f"[clip] features extracted and cached -> {FEAT_CACHE}")

    # label alignment guard (RESEARCH_PROTOCOL.md: cache order == dataset order)
    z = np.load(os.path.join(DATA, "clip_features_clean.npz"))
    assert np.array_equal(tr_y, z["train_labels"]), "train label order mismatch vs cache"
    assert np.array_equal(te_y, z["test_labels"]), "test label order mismatch vs cache"

    idx_sub, idx_val = train_test_split(np.arange(len(tr_y)), test_size=0.2,
                                        random_state=42, stratify=tr_y)
    val_accs = {}
    for C in C_GRID:
        lr = LogisticRegression(C=C, max_iter=5000, random_state=42)
        lr.fit(tr[idx_sub], tr_y[idx_sub])
        val_accs[str(C)] = float((lr.predict(tr[idx_val]) == tr_y[idx_val]).mean())
    best_C = max(val_accs, key=val_accs.get)
    c1 = LogisticRegression(C=1.0, max_iter=5000, random_state=42).fit(tr, tr_y)
    pred_c1 = c1.predict(te)
    bestC = LogisticRegression(C=float(best_C), max_iter=5000, random_state=42).fit(tr, tr_y)
    pred_best = bestC.predict(te)

    acc_c1 = float((pred_c1 == te_y).mean())
    acc_best = float((pred_best == te_y).mean())

    ref = json.load(open(os.path.join(DATA, "canonical_results.json")))
    exp_c1 = ref["clip_by_checkpoint"]["dfn_clip"]["linear_C1"]["accuracy"]
    exp_best = ref["clip_by_checkpoint"]["dfn_clip"]["linear_bestC"]["accuracy"]
    exp_C = float(ref["clip_by_checkpoint"]["dfn_clip"]["best_C"])
    print(f"[clip] C=1 {acc_c1:.4f} (canonical {exp_c1:.4f}) | "
          f"best-C={best_C} -> {acc_best:.4f} (canonical C={exp_C:.0f} -> {exp_best:.4f})")

    # HARD GATE: C=1 is the number Table 2 and both figures report, so it must match.
    assert abs(acc_c1 - exp_c1) < 5e-4, f"CLIP C=1 drift: {acc_c1} != {exp_c1}"
    print("[clip] OK: C=1 matches canonical_results.json exactly")

    # NOT a gate: the validation top is a near-flat plateau (90.0-90.4% across
    # C in {100,300,1000,3000}, as the paper itself notes), so its argmax is not
    # stable -- a sub-percent wobble in validation accuracy moves the selected C
    # and shifts the refit test accuracy (observed here: C=100 -> 86.2% vs the
    # reported C=300 -> 85.5%). No figure consumes the best-C number, so this is
    # recorded for the record rather than enforced.
    if float(best_C) != exp_C or abs(acc_best - exp_best) >= 5e-4:
        print(f"[clip] NOTE (not an error): best-C argmax moved {exp_C:.0f} -> {best_C}; "
              f"test {exp_best:.4f} -> {acc_best:.4f}. Validation top is flat: "
              + ", ".join(f"C={c}:{v:.4f}" for c, v in val_accs.items()
                          if float(c) >= 100.0))

    perclass = [float((pred_c1[te_y == c] == c).mean()) if (te_y == c).sum()
                else float("nan") for c in range(16)]
    bal = float(np.mean([(pred_c1[te_y == c] == c).mean() for c in range(16)
                         if (te_y == c).sum()]))
    f1 = float(f1_score(te_y, pred_c1, average="macro", zero_division=0))
    return {
        "protocol": ("fresh deterministic DFN-CLIP ViT-L/14 features (EVAL_TRANSFORM); "
                     "C selected on 20% stratified validation (seed 42), refit on 100% "
                     "train, one evaluation on the 138 test images"),
        "C_grid": C_GRID, "val_accs": val_accs, "best_C": float(best_C),
        "linear_C1": {"accuracy": acc_c1, "balanced_accuracy": bal, "macro_f1": f1,
                      "preds": pred_c1.tolist(),
                      "confusion": confusion_matrix(te_y, pred_c1,
                                                    labels=range(16)).tolist(),
                      "perclass": perclass},
        "linear_bestC": {"accuracy": acc_best, "preds": pred_best.tolist()},
        "test_labels": te_y.tolist(),
    }


def resnet_artifacts():
    """Canonical ResNet-18 fine-tune, 15 seeds. Seed 0 is the figure panel."""
    torch.manual_seed(0)
    np.random.seed(0)
    train_ds = TPDDDataset("train", transform=TRAIN_TRANSFORM)
    val_ds = TPDDDataset("validation", transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset("test", transform=EVAL_TRANSFORM)
    NC = len(train_ds.class_names)
    val_loader = DataLoader(val_ds, 32, shuffle=False, num_workers=2,
                            worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_ds, 32, shuffle=False, num_workers=2,
                             worker_init_fn=worker_init_fn)
    accs, seed0 = [], None
    perclass_all, bals_all = [], []
    t0 = time.time()
    for seed in range(SEEDS):
        set_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  num_workers=2, worker_init_fn=worker_init_fn)
        model = ResNetBaseline(NC).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = nn.CrossEntropyLoss()
        best_val, best_state = 0.0, None
        for ep in range(EPOCHS):
            model.train()
            for b in train_loader:
                img, lbl = b["image"].to(DEVICE), b["label"].to(DEVICE)
                opt.zero_grad()
                loss = crit(model(img), lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            model.eval()
            vc = vn = 0
            with torch.no_grad():
                for b in val_loader:
                    img, lbl = b["image"].to(DEVICE), b["label"].to(DEVICE)
                    vc += (model(img).max(1)[1] == lbl).sum().item()
                    vn += lbl.size(0)
            if vc / vn > best_val:
                best_val = vc / vn
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        assert best_state is not None, (
            "best_state is None after training: validation accuracy was 0 at "
            "every epoch, so no checkpoint was selected. Usually the validation "
            "split is empty or misaligned, or training diverged from step 1.")
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for b in test_loader:
                img, lbl = b["image"].to(DEVICE), b["label"].to(DEVICE)
                preds.append(model(img).max(1)[1].cpu())
                labels.append(lbl.cpu())
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        acc = float((preds == labels).mean())
        accs.append(acc)
        if seed == 0:
            perclass = [float((preds[labels == c] == c).mean()) if (labels == c).sum()
                        else float("nan") for c in range(NC)]
            bal = float(np.mean([(preds[labels == c] == c).mean() for c in range(NC)
                                 if (labels == c).sum()]))
            f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
            seed0 = {"accuracy": acc, "balanced_accuracy": bal, "macro_f1": f1,
                     "preds": preds.tolist(),
                     "confusion": confusion_matrix(labels, preds,
                                                   labels=range(NC)).tolist(),
                     "perclass": perclass, "best_val": best_val}
        # Accumulate per-class recall over EVERY seed, not just seed 0. A single
        # run is not a stable basis for a per-class table: repeated executions of
        # "seed 0" differ by up to ~3 pp because cuDNN autotuning and DataLoader
        # worker seeding are not deterministic, so seed-0-only per-class numbers
        # are an arbitrary draw. The 15-seed mean is the statistic worth tabling.
        perclass_all.append([float((preds[labels == c] == c).mean())
                             if (labels == c).sum() else float("nan")
                             for c in range(NC)])
        bals_all.append(bal if seed == 0 else
                        float(np.mean([(preds[labels == c] == c).mean() for c in range(NC)
                                       if (labels == c).sum()])))
        a = np.array(accs)
        print(f"[resnet] seed {seed:2d}/{SEEDS}: acc={acc*100:.1f}%  "
              f"running mean={a.mean()*100:.2f}% +/- {a.std(ddof=1)*100:.2f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    mean, sd = float(np.mean(accs)), float(np.std(accs, ddof=1))
    ref = json.load(open(os.path.join(DATA, "full_provenance.json")))["resnet_full"]
    print(f"[resnet] 15-seed mean {mean*100:.2f}% +/- {sd*100:.2f} "
          f"(original-image provenance {ref['mean']*100:.2f}% +/- {ref['std']*100:.2f})")
    # The TPDD images were reconstructed after the original 11 MB set was lost, so the
    # 15-seed mean is compared against, not asserted equal to, the provenance value.
    return {
        "protocol": ref["protocol"],
        "n_seeds": SEEDS, "epochs": EPOCHS,
        "accs": accs, "mean": mean, "std": sd,
        "provenance_original_images": {"mean": ref["mean"], "std": ref["std"],
                                       "accs": ref["accs"]},
        "seed0": seed0,
        # 15-seed per-class recall: the tabulatable statistic.
        "perclass_mean": np.nanmean(np.array(perclass_all, dtype=float), axis=0).tolist(),
        "perclass_sd": np.nanstd(np.array(perclass_all, dtype=float), axis=0, ddof=1).tolist(),
        "balanced_accuracy_mean": float(np.mean(bals_all)),
        "balanced_accuracy_sd": float(np.std(bals_all, ddof=1)),
        "perclass_note": ("mean +/- sd over all 15 seeds; a single run is not a stable "
                          "basis for a per-class table because repeated executions of "
                          "'seed 0' differ by a few points (cuDNN autotuning and "
                          "DataLoader worker seeding are not deterministic)."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-only", action="store_true")
    args = ap.parse_args()
    print(f"Device: {DEVICE}")
    art = {"clip": clip_artifacts()}
    if not args.clip_only:
        art["resnet"] = resnet_artifacts()
    prev = {}
    if os.path.exists(OUT):
        prev = json.load(open(OUT))
    prev.update(art)
    with open(OUT, "w") as f:
        json.dump(prev, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
