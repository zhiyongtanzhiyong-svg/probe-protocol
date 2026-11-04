"""
DINOv2 ViT-L/14 full fine-tune, matched adaptation budget (the missing 2x2 cell).

Closes the matrix cell "image-only backbone x fine-tuned" that the frozen
DINOv2 control (data/dinov2_protocol.json) leaves empty. Recipe mirrors
scripts/clip_finetune_kshot.py exactly so the arm is comparable to the
published CLIP-FT and ResNet-FT arms:

  * support indices: coop_experiment.kshot_subset_idx(y, K, seed+500),
    asserted index-identical to paired_kshot_a4.KShotSubset at run time
    (the seed-paired comparison is void otherwise);
  * grid: 9 configs = 30/100/300 epochs x 5e-6/2e-5/5e-5 backbone lr
    (the same transformer-encoder lr range as the CLIP-FT arm; equal grid
    SIZE is the fairness contract, as in Methods);
  * stage 1: 3 seeds, selection on the 138-image validation split ONLY
    (val every 10 epochs); test evaluated once, at the best-val epoch,
    never used for selection;
  * stage 2: stage-1 winner re-run from scratch, 15 seed-paired seeds;
  * each seed restores the pretrained backbone from the CPU snapshot
    (independent runs; the sequential-warm-start tell is asserted on).

Modes:
  --k K        few-shot (K=1/2/5/10)   -> data/dinov2_kshot_ft_k{K}.json
  --full       full 1295 training      -> data/dinov2_finetune_full.json

Train transform: the CLIP-FT augmented pipeline (Resize 256, RRC 224,
HFlip, Rot 15, ColorJitter) with ImageNet statistics (DINOv2-native).
Val/test: the official DINOv2 eval transform used by dinov2_protocol.py
(Resize 256 bicubic -> CenterCrop 224 -> ImageNet norm), so frozen and
fine-tuned DINOv2 numbers share one evaluation pipeline.
"""
import os
import sys
import json
import time
import argparse
import hashlib
import importlib.util

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from dataset import TPDDDataset, worker_init_fn          # noqa: E402
from coop_experiment import kshot_subset_idx             # noqa: E402
from dinov2_protocol import load_dinov2, DEVICE          # noqa: E402
# PTH overridable via env (the canonical copy lives in this repo's
# checkpoints/dinov2/; other machines keep their own).
PTH = os.environ.get("DINOV2_PTH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints/dinov2/dinov2_vitl14_pretrain.pth")
# Official facebook/dinov2-large LVD-142M base (HuggingFace).
EXPECTED_SHA256 = "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"


def verify_weights(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != EXPECTED_SHA256:
        sys.exit(f"FATAL: weight checksum mismatch for {path}:\n"
                 f"  got      {digest}\n  expected {EXPECTED_SHA256}")
    print(f"[verify] weight sha256 ok ({digest[:12]}...)", flush=True)

NC = 16
SEED_OFFSET = 500
VAL_EVERY = 10
OUT_DIM = 1024

# Same COUNT and same lr range as the CLIP-FT arm (fairness contract, Methods).
LR_GRID = [5e-6, 2e-5, 5e-5]
EPOCH_GRID = [30, 100, 300]
HEAD_LR = 1e-3

IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)

DINO_TRAIN_TF = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.ToTensor(),
    T.Normalize(IN_MEAN, IN_STD),
])
# Official DINOv2 eval transform (identical to dinov2_protocol.py).
DINO_EVAL_TF = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IN_MEAN, IN_STD),
])


class IndexSubset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]


def verify_draw_matches_canonical(train_ds, K, seeds):
    spec = importlib.util.spec_from_file_location(
        "_pka", os.path.join(REPO, "scripts", "paired_kshot_a4.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    labels = np.asarray(train_ds.label_ids)
    for s in seeds:
        mine = np.sort(kshot_subset_idx(labels, K, s + SEED_OFFSET))
        theirs = np.sort(np.asarray(mod.KShotSubset(train_ds, K, s + SEED_OFFSET).indices))
        assert mine.shape == theirs.shape and bool((mine == theirs).all()), (
            f"K={K} seed={s}: support draw differs from paired_kshot_a4.py; "
            f"the seed-paired comparison would be void. Refusing to run.")
    return True


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward(model, head, img):
    return head(model(img))


@torch.no_grad()
def evaluate(model, head, loader):
    model.eval()
    correct = tot = 0
    for batch in loader:
        pred = forward(model, head, batch["image"].to(DEVICE, non_blocking=True)).argmax(1)
        lab = batch["label"].to(DEVICE, non_blocking=True)
        correct += int((pred == lab).sum())
        tot += int(lab.numel())
    return correct / max(tot, 1)


def run_one(model, head, pretrained, train_dl, val_dl, test_dl, lr, epochs, seed):
    set_seed(seed)
    model.load_state_dict(pretrained)          # independent run from common init
    nn.init.kaiming_uniform_(head.weight, a=5)
    nn.init.zeros_(head.bias)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": params, "lr": lr},
        {"params": [head.weight, head.bias], "lr": HEAD_LR, "weight_decay": 0.0},
    ], weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    allp = params + [head.weight, head.bias]
    best_val, best_test, best_ep = -1.0, -1.0, -1
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for batch in train_dl:
            img = batch["image"].to(DEVICE, non_blocking=True)
            lab = batch["label"].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(forward(model, head, img), lab)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            scaler.step(opt)
            scaler.update()
        sch.step()
        if (ep + 1) % VAL_EVERY == 0 or ep == epochs - 1:
            val_acc = evaluate(model, head, val_dl)
            if val_acc > best_val:
                best_val, best_ep = val_acc, ep + 1
                best_test = evaluate(model, head, test_dl)
    return {"seed": seed, "lr": lr, "epochs": epochs,
            "best_val": round(float(best_val), 4), "test": round(float(best_test), 4),
            "best_epoch": best_ep, "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--search-seeds", type=int, default=3)
    ap.add_argument("--final-seeds", type=int, default=15)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    assert args.full != (args.k is not None), "use exactly one of --k K or --full"
    K = args.k or "full"
    if args.out is None:
        fn = f"dinov2_finetune_full.json" if args.full else f"dinov2_kshot_ft_k{K}.json"
        args.out = os.path.join(REPO, "data", fn)
    grid = [{"lr": lr, "epochs": ep} for lr in LR_GRID for ep in EPOCH_GRID]
    assert len(grid) == 9, "grid size is the fairness contract; do not change it silently"

    t0 = time.time()
    print(f"Device={DEVICE} K={K} grid=9 search_seeds={args.search_seeds} "
          f"final_seeds={args.final_seeds}", flush=True)

    train_full = TPDDDataset("train", transform=DINO_TRAIN_TF)
    val_full = TPDDDataset("validation", transform=DINO_EVAL_TF)
    test_full = TPDDDataset("test", transform=DINO_EVAL_TF)
    labels = np.asarray(train_full.label_ids)
    val_dl = DataLoader(val_full, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_full, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    verify_weights(PTH)
    model = load_dinov2(PTH)
    for p in model.parameters():
        p.requires_grad = True
    head = nn.Linear(OUT_DIM, NC).to(DEVICE)
    pretrained = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def train_loader(seed):
        if args.full:
            idx = np.arange(len(train_full))
        else:
            idx = kshot_subset_idx(labels, args.k, seed + SEED_OFFSET)
        sub = IndexSubset(train_full, idx)
        return DataLoader(sub, batch_size=min(args.batch, len(sub)), shuffle=True,
                          num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn), len(sub)

    if not args.full:
        verify_draw_matches_canonical(train_full, args.k, [0, 1, 2, 7, 14])
    n_support = len(train_full) if args.full else int(len(kshot_subset_idx(labels, args.k, SEED_OFFSET)))

    def dump(obj):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(obj, open(args.out, "w"), indent=2, ensure_ascii=False)

    # ---- Stage 1: search on VALIDATION ONLY. ----
    stage1 = []
    for cfg in grid:
        runs = []
        for s in range(args.search_seeds):
            tl, ns = train_loader(s)
            runs.append(run_one(model, head, pretrained, tl, val_dl, test_dl,
                                cfg["lr"], cfg["epochs"], s))
        score = float(np.mean([r["best_val"] for r in runs]))
        stage1.append({**cfg, "mean_best_val": round(score, 4),
                       "mean_test": round(float(np.mean([r["test"] for r in runs])), 4),
                       "runs": runs})
        print(f"  cfg lr={cfg['lr']:<8g} ep={cfg['epochs']:<4d} val={score*100:.2f}% "
              f"(test {stage1[-1]['mean_test']*100:.2f}%, not used for selection) "
              f"[{runs[-1]['elapsed_s']}s/run]", flush=True)
        dump({  # incremental: a killed run keeps its stage-1 progress
            "arch": "DINOv2 ViT-L/14 (LVD-142M base pretrain) full fine-tune + linear head 16",
            "K": K, "n_support": n_support, "stage1_search": stage1, "partial": True})

    best = max(stage1, key=lambda c: c["mean_best_val"])
    print(f"  -> stage-1 winner: lr={best['lr']} epochs={best['epochs']} "
          f"val={best['mean_best_val']*100:.2f}%", flush=True)

    # ---- Stage 2: winner re-run from scratch, full seed count. ----
    final_runs = []
    for s in range(args.final_seeds):
        tl, ns = train_loader(s)
        r = run_one(model, head, pretrained, tl, val_dl, test_dl,
                    best["lr"], best["epochs"], s)
        final_runs.append(r)
        print(f"  final seed {s:2d}: val={r['best_val']*100:.2f}% test={r['test']*100:.2f}% "
              f"({r['elapsed_s']}s)", flush=True)
        dump({
            "arch": "DINOv2 ViT-L/14 (LVD-142M base pretrain) full fine-tune + linear head 16",
            "K": K, "n_support": n_support, "stage1_search": stage1,
            "selected": best, "stage2_final": {
                "n_seeds": len(final_runs), "per_seed": final_runs,
                "accs": [x["test"] for x in final_runs],
                "mean": round(float(np.mean([x["test"] for x in final_runs])), 6),
                "std": round(float(np.std([x["test"] for x in final_runs], ddof=1)), 6),
            }, "partial": len(final_runs) < args.final_seeds})

    bvals = [r["best_val"] for r in final_runs]
    res = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "arch": "DINOv2 ViT-L/14 (LVD-142M base pretrain) full fine-tune + linear head 16",
        "checkpoint": PTH,
        "K": K, "n_classes": NC, "seed_offset": SEED_OFFSET,
        "support_draw": ("coop_experiment.kshot_subset_idx(y, K, seed+500); asserted "
                         "index-identical to paired_kshot_a4.KShotSubset at run time")
        if not args.full else "full 1295-image training split (no support draw)",
        "n_support": n_support,
        "val_every_epochs": VAL_EVERY,
        "head_lr": HEAD_LR,
        "train_transform": "CLIP-FT augmented pipeline, ImageNet stats (DINOv2-native)",
        "eval_transform": "official DINOv2 (Resize256 bicubic -> CenterCrop224 -> ImageNet norm)",
        "grid": grid,
        "stage1_search": stage1,
        "selected": best,
        "stage2_final": {
            "n_seeds": args.final_seeds,
            "per_seed": final_runs,
            "accs": [r["test"] for r in final_runs],
            "mean": round(float(np.mean([r["test"] for r in final_runs])), 6),
            "std": round(float(np.std([r["test"] for r in final_runs], ddof=1)), 6),
            "sem": round(float(np.std([r["test"] for x in final_runs], ddof=1)
                               / np.sqrt(len(final_runs))), 6),
        },
        "best_val_monotonic_nondecreasing": all(
            bvals[i] <= bvals[i + 1] for i in range(len(bvals) - 1)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    dump(res)
    print(f"\n=== DINOv2 K={K} matched budget ===")
    print(f"  stage-2 mean = {res['stage2_final']['mean']*100:.2f}% +/- "
          f"{res['stage2_final']['std']*100:.2f}% (n={args.final_seeds})")
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
