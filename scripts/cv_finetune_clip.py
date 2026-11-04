"""
5-fold cross-validation of the FINE-TUNED CLIP headline (the symmetric cell the
frozen-probe CV in cv_protocol.py leaves open).

Both reviewers of 2026-10-24 flagged that the headline "fine-tuned CLIP beats
fine-tuned ResNet-18 (95.9% vs 87.7% on TPDD)" rests on the single canonical
train/test split with no fold-level uncertainty, while the frozen-probe result
does carry 5-fold CV. This script closes that asymmetry.

Protocol (mirrors scripts/cv_protocol.py for direct comparability):
  - Same 5-fold stratified partition of the 1295-image training pool
    (StratifiedKFold(n_splits=5, random_state=42)).
  - Per fold: 80/20 stratified split of the fold (StratifiedShuffleSplit(1,
    test_size=0.2, random_state=fold)); 80% trains, 20% is the selection set.
  - Model: CLIP ViT-L/14 image encoder (model.visual) + linear head, fine-tuned
    with the EXACT recipe of the canonical full-data fine-tune cell
    (vis_lr=5e-6, head_lr=1e-3, AdamW wd=0.01, cosine, batch 32, 30 epochs,
    best-validation selection, mixed precision, gradient checkpointing).
  - One seed per fold (seed 0), matching the ResNet-18 CV protocol.
  - Test: held-out 20% of the fold (never seen, never used for selection).
  - Report: per-fold accuracy, mean +/- SD, and a paired t-test against the
    ResNet-18 per-fold accuracies in data/cv_protocol.json.

Output: data/cv_finetune_clip.json
"""
import os, sys, json, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import open_clip
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from scipy import stats

REPO = ROOT
sys.path.insert(0, os.path.join(REPO, "scripts"))
from dataset import TPDDDataset  # noqa
from cv_protocol import set_seed  # noqa (shared seeding)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPTS = {
    "dfn_clip": "checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin",
    "laion2b": "checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin",
}
MODEL = "ViT-L-14"
NC = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5

# CLIP-native normalization (ViT-L/14) for the augmented training pipeline
# (identical to scripts/clip_finetune.py).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_TRAIN_TF = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])

# Canonical fine-tune recipe (from data/clip_finetune.json).
EPOCHS = 30
BATCH = 32
VIS_LR = 5e-6
HEAD_LR = 1e-3


def build_model(ckpt):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=ckpt)
    model = model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.visual.parameters():
        p.requires_grad = True
    if hasattr(model.visual, "set_grad_checkpointing"):
        model.visual.set_grad_checkpointing(True)
    D = model.visual.output_dim
    head = nn.Linear(D, NC).to(DEVICE)
    return model, head, preprocess


def forward(model, head, img):
    return head(model.visual(img))


@torch.no_grad()
def evaluate(model, head, loader):
    model.eval()
    correct = tot = 0
    for batch in loader:
        logits = forward(model, head, batch["image"].to(DEVICE, non_blocking=True))
        pred = logits.argmax(1)
        lab = batch["label"].to(DEVICE, non_blocking=True)
        correct += int((pred == lab).sum())
        tot += int(lab.numel())
    return correct / max(tot, 1)


def train_fold(model, head, vis_params, train_dl, val_dl, test_dl):
    """Fine-tune one fold; return (best_val_acc, best_test_acc)."""
    nn.init.kaiming_uniform_(head.weight, a=5)
    torch.nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW([
        {"params": vis_params, "lr": VIS_LR},
        {"params": [head.weight, head.bias], "lr": HEAD_LR, "weight_decay": 0.0},
    ], weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    best_val, best_test = -1.0, -1.0
    for ep in range(EPOCHS):
        model.train()
        for batch in train_dl:
            img = batch["image"].to(DEVICE, non_blocking=True)
            lab = batch["label"].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(forward(model, head, img), lab)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(vis_params + [head.weight, head.bias], 1.0)
            scaler.step(opt)
            scaler.update()
        sch.step()
        val_acc = evaluate(model, head, val_dl)
        test_acc = evaluate(model, head, test_dl)
        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc
    return best_val, best_test


def run_ckpt(name, ckpt):
    model, head, preprocess = build_model(ckpt)
    vis_params = [p for p in model.visual.parameters() if p.requires_grad]
    # Pretrained snapshot: every fold restarts from common pretrained weights,
    # so folds are independent runs (not sequential warm-starts).
    pretrained_visual = {k: v.detach().cpu().clone() for k, v in model.visual.state_dict().items()}

    train_ds = TPDDDataset("train", transform=CLIP_TRAIN_TF)
    test_ds = TPDDDataset("test", transform=preprocess)
    # Non-augmented view of the SAME train pool, for the selection set.
    sel_base = TPDDDataset("train", transform=preprocess)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    labels = np.array(train_ds.label_ids)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        t0 = time.time()
        # Splits are deterministic via their own random_state; training itself
        # uses a fixed seed 0 in every fold (mirrors the ResNet-18 CV, which
        # trains each fold with set_seed(0)).
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=fold)
        fit_local, sel_local = next(sss.split(np.zeros(len(train_idx)), labels[train_idx]))
        fit_idx = np.array(train_idx)[fit_local]
        sel_idx = np.array(train_idx)[sel_local]
        test_idx = np.array(test_idx)
        set_seed(0)
        model.visual.load_state_dict(pretrained_visual)

        fit_dl = DataLoader(train_ds, batch_size=BATCH,
                            sampler=torch.utils.data.SubsetRandomSampler(list(fit_idx)),
                            num_workers=0, pin_memory=True)
        sel_dl = DataLoader(Subset(sel_base, list(sel_idx)), batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=True)
        # Fold test set = the held-out 20% of the fold, drawn from the TRAIN
        # pool (259 images), viewed with the non-augmented transform.
        fold_test_dl = DataLoader(Subset(sel_base, list(test_idx)), batch_size=64, shuffle=False,
                                  num_workers=0, pin_memory=True)

        best_val, best_test = train_fold(model, head, vis_params, fit_dl, sel_dl, fold_test_dl)
        dt = time.time() - t0
        print(f"  [{name}] fold {fold}: fit={len(fit_idx)} sel={len(sel_idx)} test={len(test_idx)} "
              f"best_val={best_val*100:.2f}% -> fold_test={best_test*100:.2f}% ({dt:.0f}s)", flush=True)
        folds.append({"fold": fold, "n_fit": int(len(fit_idx)), "n_selection": int(len(sel_idx)),
                      "n_test": int(len(test_idx)), "best_val": round(float(best_val), 4),
                      "fold_acc": round(float(best_test), 4), "elapsed_s": round(dt, 1)})
    accs = [f["fold_acc"] for f in folds]
    return {"checkpoint": name, "n_folds": N_FOLDS, "folds": folds,
            "mean": round(float(np.mean(accs)), 4),
            "sd": round(float(np.std(accs, ddof=1)), 4),
            "accs": accs}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=str, default="dfn_clip", help="dfn_clip|laion2b|both")
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "data/cv_finetune_clip.json"))
    args = ap.parse_args()
    which = ["dfn_clip", "laion2b"] if args.which == "both" else [args.which]

    t0 = time.time()
    _cv = json.load(open(os.path.join(REPO, "data/cv_protocol.json")))
    resnet_ref = {"mean": _cv["summary"]["resnet"]["acc_mean"],
                  "sd": _cv["summary"]["resnet"]["acc_sd"],
                  "accs": _cv["summary"]["resnet"]["per_fold"]}
    results = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "open_clip": open_clip.__version__,
        "protocol": (
            "5-fold CV of fine-tuned CLIP (image encoder + linear head), mirroring "
            "cv_protocol.py: StratifiedKFold(5, seed=42) over the 1295-image training pool; "
            "per fold 80% fit / 20% validation-selection (StratifiedShuffleSplit seed=fold); "
            f"recipe identical to the canonical full-data fine-tune cell (30 ep, batch {BATCH}, "
            f"vis_lr {VIS_LR}, head_lr {HEAD_LR}, AdamW wd 0.01, cosine, best-validation selection, "
            "mixed precision, gradient checkpointing); one seed per fold (seed 0); "
            "test = held-out 20% of the fold."
        ),
        "resnet18_ref": resnet_ref,
        "results": {},
    }
    for name in which:
        results["results"][name] = run_ckpt(name, CKPTS[name])

    # Paired t-test vs ResNet-18 per-fold accuracies (same fold partition).
    for name, r in results["results"].items():
        rn = resnet_ref["accs"]
        a = np.array(r["accs"])
        b = np.array(rn[:len(a)])
        d = a - b
        t, p = stats.ttest_1samp(d, 0)
        r["paired_vs_resnet"] = {
            "resnet_accs": b.tolist(),
            "mean_gap_pp": round(float(d.mean()) * 100, 2),
            "t": round(float(t), 3), "df": int(len(a) - 1),
            "p_two_sided": round(float(p), 5),
            "fold_gaps_pp": [round(float(x) * 100, 2) for x in d],
            "note": ("5 folds share ~80% of the training data; the sign is the robust part, "
                     "not the magnitude (same caveat as the frozen-probe CV)."),
        }

    results["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("\n=== SUMMARY ===", flush=True)
    for name, r in results["results"].items():
        pv = r["paired_vs_resnet"]
        print(f"  {name:9s}: fine-tuned CLIP = {r['mean']*100:.2f}% +/- {r['sd']*100:.2f}% over {r['n_folds']} folds", flush=True)
        print(f"             vs ResNet-18 {resnet_ref['mean']*100:.2f}% +/- {resnet_ref['sd']*100:.2f}%", flush=True)
        print(f"             paired: gap {pv['mean_gap_pp']:+.2f}pp, t({pv['df']})={pv['t']}, p_two={pv['p_two_sided']}, "
              f"fold gaps {pv['fold_gaps_pp']}", flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
