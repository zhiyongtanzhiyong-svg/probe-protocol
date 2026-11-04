"""
ConvNeXt fine-tuning on TPDD — modern CNN baseline.

Replaces the 2015 ResNet-18 with a 2022 ConvNeXt (convnext_tiny, ~28M params)
as the "fine-tuned CNN" control. Same training protocol as the ResNet-18
canonical run (AdamW, cosine, batch 32, 30 epochs, best-validation selection,
AMP, 15 seeds) so the comparison is apples-to-apples.

Uses torchvision's built-in ConvNeXt (no download needed).

Output: data/convnext_full.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as tv_models

REPO = ROOT
sys.path.insert(0, os.path.join(REPO, "scripts"))
from dataset import TPDDDataset  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NC = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "convnext_tiny"

# ImageNet normalization (native for ConvNeXt)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRAIN_TF = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TF = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(wid):
    np.random.seed((int(time.time()) + wid) % (2**32))


def build_model():
    weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = tv_models.convnext_tiny(weights=weights)
    # classifier is nn.Sequential(LayerNorm2d, Flatten, Linear(768, 1000))
    in_dim = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_dim, NC)
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ConvNeXt] {MODEL_NAME}: {n_params:,} params", flush=True)
    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = tot = 0
    for batch in loader:
        logits = model(batch["image"].to(DEVICE, non_blocking=True))
        pred = logits.argmax(1)
        lab = batch["label"].to(DEVICE, non_blocking=True)
        correct += int((pred == lab).sum()); tot += int(lab.numel())
    return correct / max(tot, 1)


def run(seeds, epochs, batch, lr, wd):
    train_ds = TPDDDataset("train", transform=TRAIN_TF)
    val_ds = TPDDDataset("validation", transform=EVAL_TF)
    test_ds = TPDDDataset("test", transform=EVAL_TF)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0,
                          pin_memory=True, worker_init_fn=worker_init_fn)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[ConvNeXt] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"seeds={seeds} epochs={epochs} batch={batch} lr={lr} wd={wd}", flush=True)

    # Snapshot pretrained weights for seed reset
    model = build_model()
    pretrained = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out = {"model": MODEL_NAME, "n_seeds": seeds, "epochs": epochs, "batch": batch,
           "lr": lr, "wd": wd, "seeds": []}
    for seed in range(seeds):
        set_seed(seed)
        model.load_state_dict(pretrained)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        best_val, best_test = -1.0, -1.0
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            for batch in train_dl:
                img = batch["image"].to(DEVICE, non_blocking=True)
                lab = batch["label"].to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    loss = crit(model(img), lab)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            sch.step()
            val_acc = evaluate(model, val_dl)
            test_acc = evaluate(model, test_dl)
            if val_acc > best_val:
                best_val, best_test = val_acc, test_acc
            if (ep in (0, epochs // 2, epochs - 1)):
                print(f"  seed {seed} ep {ep+1}/{epochs}: val={val_acc*100:.1f}% "
                      f"test={test_acc*100:.1f}%", flush=True)
        dt = time.time() - t0
        print(f"  seed {seed}: best-val={best_val*100:.2f}% -> test={best_test*100:.2f}% "
              f"({dt:.0f}s)", flush=True)
        out["seeds"].append({"seed": seed, "best_val": round(float(best_val), 4),
                             "test": round(float(best_test), 4), "elapsed_s": round(dt, 1)})
    accs = [s["test"] for s in out["seeds"]]
    out["mean"] = round(float(np.mean(accs)), 4)
    out["std"] = round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0
    out["sem"] = round(float(np.std(accs, ddof=1) / np.sqrt(len(accs))), 4) if len(accs) > 1 else 0.0
    out["protocol"] = (
        f"ConvNeXt ({MODEL_NAME}, torchvision) fine-tuned end-to-end, ImageNet normalization, "
        f"RandomResizedCrop/flip/rotate/color-jitter for train, AdamW (lr {lr}, wd {wd}), "
        f"cosine, batch {batch}, {epochs} ep, best-validation selection, "
        f"full TPDD train (1295) / 138 test, mixed precision, {seeds} seeds."
    )
    return out


def _resnet_ref():
    import json as _json
    from pathlib import Path as _Path
    art = _json.loads((_Path(__file__).resolve().parent.parent
                       / "data" / "canonical_fig_artifacts.json").read_text())
    r = art["resnet"]
    return {"accuracy": r["mean"], "n_seeds": r["n_seeds"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "data/convnext_full.json"))
    args = ap.parse_args()

    t0 = time.time()
    results = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
        "resnet18_finetuned_ref": _resnet_ref(),
        "results": run(args.seeds, args.epochs, args.batch, args.lr, args.wd),
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    r = results["results"]
    ref = results["resnet18_finetuned_ref"]
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  ConvNeXt-V2: {r['mean']*100:.2f}% +/- {r['std']*100:.2f}% (n={r['n_seeds']}) "
          f"vs ResNet-18: {ref['accuracy']*100:.2f}% (n={ref['n_seeds']})", flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
