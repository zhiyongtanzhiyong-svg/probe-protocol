"""
P4: scale control -- ResNet-50 (25.6M) full-data fine-tune, matched to ResNet-18.

Answers the "428M vs 12M (~35x) scale asymmetry, no size-disclosed CNN baseline" concern:
is the fine-tuned-CNN result (ResNet-18 86.4%) scale-robust, or a ResNet-18 curiosity?
Run ResNet-50 (~2x ResNet-18, still far below CLIP's 428M) under the identical protocol
(ImageNet-pretrained, AdamW lr=3e-4, cosine, batch 32, 30 epochs, best-val, N seeds,
full 1295 train / 138 test, ImageNet-native transforms). Compare to ResNet-18 (86.4%)
and fine-tuned CLIP (94.8-96.7%).

Output: data/resnet50_full.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

REPO = "/home/lizhuo/Work/TobAI/Paper2-CLIPvsResNet"
PAPER1 = "/home/lizhuo/Work/TobAI/Paper2-CLIPvsResNet/scripts"
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, PAPER1)
from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from model_zoo import ResNetBaseline, set_seed

NC = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def perclass_and_overall(model, loader, device, nc):
    model.eval(); corr = np.zeros(nc); cnt = np.zeros(nc); ok = n = 0
    with torch.no_grad():
        for batch in loader:
            img, lbl = batch["image"].to(device), batch["label"].to(device)
            pred = model(img).argmax(1)
            for c in range(nc):
                mask = (lbl == c).cpu().numpy()
                if mask.sum():
                    corr[c] += int((pred[mask] == c).sum()); cnt[c] += int(mask.sum())
            ok += int((pred == lbl).sum()); n += int(lbl.numel())
    pc = [None if cnt[c] == 0 else float(corr[c] / cnt[c]) for c in range(nc)]
    return pc, (ok / n if n else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default="resnet50")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "data/resnet50_full.json"))
    args = ap.parse_args()

    train_full = TPDDDataset("train", transform=TRAIN_TRANSFORM)
    val_full = TPDDDataset("validation", transform=EVAL_TRANSFORM)
    test_full = TPDDDataset("test", transform=EVAL_TRANSFORM)
    train_loader = DataLoader(train_full, batch_size=args.batch, shuffle=True,
                              num_workers=2, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_full, batch_size=64, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_full, batch_size=64, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    print(f"backbone={args.backbone} train={len(train_full)} val={len(val_full)} test={len(test_full)} "
          f"seeds={args.seeds} epochs={args.epochs}", flush=True)

    out = {"date": datetime.today().isoformat(), "backbone": args.backbone,
           "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
           "n_seeds": args.seeds, "epochs": args.epochs, "accs": []}
    accs = []
    t0 = time.time()
    for seed in range(args.seeds):
        set_seed(seed)
        model = ResNetBaseline(NC, backbone=args.backbone).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss()
        best_val, best_state = 0.0, None
        ts = time.time()
        for ep in range(args.epochs):
            model.train()
            for batch in train_loader:
                img, lbl = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                opt.zero_grad(); crit(model(img), lbl).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            model.eval(); vc = vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    img, lbl = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                    vc += int((model(img).max(1)[1] == lbl).sum()); vn += lbl.size(0)
            if vc / vn > best_val:
                best_val = vc / vn
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        _, ta = perclass_and_overall(model, test_loader, DEVICE, NC)
        accs.append(ta); a = np.array(accs)
        print(f"  seed {seed}: best-val={best_val*100:.2f}% -> test={ta*100:.2f}% "
              f"mean={a.mean()*100:.1f}+/-{a.std(ddof=1)*100:.1f} [{time.time()-t0:.0f}s]", flush=True)
    a = np.array(accs)
    out["mean"] = float(a.mean()); out["std"] = float(a.std(ddof=1))
    out["sem"] = float(a.std(ddof=1) / np.sqrt(len(accs)))
    out["min"] = float(a.min()); out["max"] = float(a.max())
    out["elapsed_s"] = float(time.time() - t0)
    out["protocol"] = (f"{args.backbone} ImageNet-pretrained, AdamW lr=3e-4, cosine, bs={args.batch}, "
                       f"{args.epochs}ep, best-val, full TPDD train (1295)/138 test, {args.seeds} seeds")
    out["refs"] = {"resnet18_finetuned": 0.8643, "clip_finetuned_dfn": 0.9667,
                   "clip_finetuned_laion": 0.9478, "clip_frozen_probe_dfn": 0.8551,
                   "clip_frozen_probe_laion": 0.8623}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== {args.backbone}: {a.mean()*100:.2f}% +/- {a.std(ddof=1)*100:.2f}% (n={len(accs)}) "
          f"vs ResNet-18 86.43% vs fine-tuned CLIP 94.78-96.67% ===", flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
