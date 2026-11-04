"""
P3b stage-1/2 -- few-shot FINE-TUNED CLIP at K=5 on TPDD, matched adaptation budget.

Why this exists
---------------
The paper's primary contrast is a FROZEN CLIP probe vs a FINE-TUNED ResNet-18. The paper
concedes (L261, L482, L553) that this is an unequal-adaptation comparison and absorbs the
objection in prose. P3b turns prose into evidence: fine-tune the CLIP encoder under a budget
matched to ResNet-18's and see whether the K=5 ordering survives.

Why K=5 is the decisive cell
-----------------------------
LAION-2B frozen probe at K=5 = 60.19%; ResNet-18 at K=5 = 65.46% -> gap 5.3 pp. The measured
fine-tuning gain for that SAME checkpoint at full data is +7.2 pp (87.7% -> 94.9%). The gap
is SMALLER than the gain, so if fine-tuning realises anything like its full-data benefit,
LAION-2B overtakes ResNet-18 at K=5 and the paper's crossover bracket (2,5) is void.
This experiment can invalidate the title claim. It is not confirmatory.

Protocol: refine-logs/P3B_KSHOT_PREREGISTRATION.md sections 2.1, 5, 6, 7, 8.

* Support indices from coop_experiment.kshot_subset_idx(y, K, seed+500), asserted
  index-identical to paired_kshot_a4.KShotSubset at RUN TIME (the seed-paired comparison is
  void otherwise).
* Stage 1: 9 configs (epochs x vis_lr), n=3 seeds, selection on VALIDATION ONLY.
* Stage 2: stage-1 winner re-run from scratch, n=15 seeds.
* Validation every VAL_EVERY epochs; test evaluated exactly once, at the best-val epoch.
* Each seed restores the pretrained model.visual weights -- the sequential-warm-start bug
  from the full-data control is carried over deliberately, and the tell-tale (a
  monotonically non-decreasing best_val across seeds) is asserted against.

Output: data/p3b_clip_finetune_kshot.json
"""
import os
import sys
import json
import time
import argparse
import importlib.util

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import open_clip

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from dataset import TPDDDataset, worker_init_fn  # noqa: E402
from coop_experiment import kshot_subset_idx  # noqa: E402

CKPTS = {
    "dfn_clip": "checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin",
    "laion2b": "checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin",
}
MODEL = "ViT-L-14"
NC = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED_OFFSET = 500
VAL_EVERY = 10

# Same COUNT as the ResNet grid (9) -- equal size is the fairness contract.
VIS_LR_GRID = [5e-6, 2e-5, 5e-5]
EPOCH_GRID = [30, 100, 300]
HEAD_LR = 1e-3

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# Same augmented train pipeline as the full-data control (clip_finetune.py), for
# consistency with the published full-data number.
CLIP_TRAIN_TF = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])


class IndexSubset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # TPDDDataset returns a dict (image / label / class_name / kg_*).
        return self.base[int(self.indices[i])]


def verify_draw_matches_canonical(train_ds, K, seeds):
    """Abort unless our draw equals the one behind the published ResNet K-shot rows."""
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


def build_model(ckpt):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=ckpt)
    model = model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.visual.parameters():
        p.requires_grad = True
    if hasattr(model.visual, "set_grad_checkpointing"):
        model.visual.set_grad_checkpointing(True)  # 428M ViT on a 12GB card
    head = nn.Linear(model.visual.output_dim, NC).to(DEVICE)
    return model, head, preprocess


def forward(model, head, img):
    return head(model.visual(img))


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


def run_one(model, head, pretrained_visual, train_dl, val_dl, test_dl,
            vis_lr, epochs, seed):
    set_seed(seed)
    model.visual.load_state_dict(pretrained_visual)  # independent run from common init
    nn.init.kaiming_uniform_(head.weight, a=5)
    nn.init.zeros_(head.bias)
    vis_params = [p for p in model.visual.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": vis_params, "lr": vis_lr},
        {"params": [head.weight, head.bias], "lr": HEAD_LR, "weight_decay": 0.0},
    ], weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
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
            torch.nn.utils.clip_grad_norm_(vis_params + [head.weight, head.bias], 1.0)
            scaler.step(opt)
            scaler.update()
        sch.step()  # once per EPOCH, matching the ResNet control
        if (ep + 1) % VAL_EVERY == 0 or ep == epochs - 1:
            val_acc = evaluate(model, head, val_dl)
            if val_acc > best_val:
                best_val, best_ep = val_acc, ep + 1
                best_test = evaluate(model, head, test_dl)
    return {"seed": seed, "vis_lr": vis_lr, "epochs": epochs,
            "best_val": round(float(best_val), 4), "test": round(float(best_test), 4),
            "best_epoch": best_ep, "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ckpt", type=str, default="laion2b")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--search-seeds", type=int, default=3)
    ap.add_argument("--final-seeds", type=int, default=15)
    ap.add_argument("--out", type=str,
                    default=os.path.join(REPO, "data/p3b_clip_finetune_kshot.json"))
    args = ap.parse_args()

    K = args.k
    grid = [{"vis_lr": lr, "epochs": ep} for lr in VIS_LR_GRID for ep in EPOCH_GRID]
    assert len(grid) == 9, "grid size is the fairness contract; do not change it silently"

    t0 = time.time()
    print(f"Device={DEVICE} K={K} ckpt={args.ckpt} grid={len(grid)} "
          f"search_seeds={args.search_seeds} final_seeds={args.final_seeds}", flush=True)

    train_full = TPDDDataset("train", transform=CLIP_TRAIN_TF)
    verify_draw_matches_canonical(train_full, K, [0, 1, 2, 7, 14])
    labels = np.asarray(train_full.label_ids)

    model, head, preprocess = build_model(CKPTS[args.ckpt])
    val_full = TPDDDataset("validation", transform=preprocess)
    test_full = TPDDDataset("test", transform=preprocess)
    val_dl = DataLoader(val_full, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_full, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    def train_loader(seed):
        idx = kshot_subset_idx(labels, K, seed + SEED_OFFSET)
        sub = IndexSubset(train_full, idx)
        return DataLoader(sub, batch_size=min(args.batch, len(sub)), shuffle=True,
                          num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn), len(sub)

    # Snapshot pretrained visual weights ONCE on CPU (see clip_finetune.py: without this
    # each seed warm-starts from the previous seed and "n seeds" is one sequential run).
    pretrained_visual = {k: v.detach().cpu().clone()
                         for k, v in model.visual.state_dict().items()}

    # ---- Stage 1: search on VALIDATION ONLY. ----
    stage1 = []
    for cfg in grid:
        runs = []
        for s in range(args.search_seeds):
            tl, ns = train_loader(s)
            runs.append(run_one(model, head, pretrained_visual, tl, val_dl, test_dl,
                                cfg["vis_lr"], cfg["epochs"], s))
        score = float(np.mean([r["best_val"] for r in runs]))
        stage1.append({**cfg, "mean_best_val": round(score, 4),
                       "mean_test": round(float(np.mean([r["test"] for r in runs])), 4),
                       "runs": runs})
        print(f"  cfg vis_lr={cfg['vis_lr']:<8g} ep={cfg['epochs']:<4d} val={score*100:.2f}%  "
              f"(test {stage1[-1]['mean_test']*100:.2f}%, not used for selection) "
              f"[{runs[-1]['elapsed_s']}s/run]", flush=True)

    best = max(stage1, key=lambda c: c["mean_best_val"])
    print(f"  -> stage-1 winner: vis_lr={best['vis_lr']} epochs={best['epochs']} "
          f"val={best['mean_best_val']*100:.2f}%", flush=True)

    # ---- Stage 2: winner re-run from scratch, full seed count. ----
    final_runs = []
    for s in range(args.final_seeds):
        tl, ns = train_loader(s)
        r = run_one(model, head, pretrained_visual, tl, val_dl, test_dl,
                    best["vis_lr"], best["epochs"], s)
        final_runs.append(r)
        print(f"  final seed {s:2d}: val={r['best_val']*100:.2f}% test={r['test']*100:.2f}% "
              f"({r['elapsed_s']}s)", flush=True)

    bvals = [r["best_val"] for r in final_runs]
    # Tell-tale for the sequential-warm-start bug: best_val that never goes DOWN across
    # seeds is what a single sequential 450-epoch run looks like. Independent seeds make
    # this unlikely; True is a flag to investigate, not a proof the bug is present.
    best_val_monotonic = all(bvals[i] <= bvals[i + 1] for i in range(len(bvals) - 1))
    n_support = len(kshot_subset_idx(labels, K, SEED_OFFSET))
    res = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "open_clip": open_clip.__version__,
        "arch": f"CLIP {MODEL} image-encoder fine-tune (model.visual + linear head {NC})",
        "checkpoint": args.ckpt,
        "K": K, "n_classes": NC, "seed_offset": SEED_OFFSET,
        "support_draw": "coop_experiment.kshot_subset_idx(y, K, seed+500); asserted "
                        "index-identical to paired_kshot_a4.KShotSubset at run time",
        "n_support": int(n_support),
        "val_every_epochs": VAL_EVERY,
        "head_lr": HEAD_LR,
        "grid": grid,
        "stage1_search": stage1,
        "selected": best and {"vis_lr": best["vis_lr"], "epochs": best["epochs"],
                              "mean_best_val": best["mean_best_val"]},
        "stage2_final": {
            "n_seeds": args.final_seeds,
            "per_seed": final_runs,
            "accs": [r["test"] for r in final_runs],
            "mean": round(float(np.mean([r["test"] for r in final_runs])), 6),
            "std": round(float(np.std([r["test"] for r in final_runs], ddof=1)), 6),
            "sem": round(float(np.std([r["test"] for r in final_runs], ddof=1)
                               / np.sqrt(len(final_runs))), 6),
        },
        "best_val_monotonic_nondecreasing": best_val_monotonic,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== CLIP {args.ckpt} K={K} matched budget ===")
    print(f"  stage-2 mean = {res['stage2_final']['mean']*100:.2f}% +/- "
          f"{res['stage2_final']['std']*100:.2f}% (n={args.final_seeds})")
    print(f"  best_val monotonic non-decreasing across seeds = {best_val_monotonic} "
          f"(True is the warm-start-bug tell; a coincidence here is not proof of absence)")
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()