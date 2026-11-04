"""
P3: fairness control -- fine-tuned CLIP (image branch) on TPDD, full data.

The paper compares FROZEN CLIP (probe) vs FINE-TUNED ResNet-18, and probes frozen
ResNet-18, but never fine-tunes CLIP. Reviewers flag this as "asymmetric effort."
This closes the symmetric cell: fine-tune both CLIP checkpoints' image encoder + a
linear head under a matched budget, and report fine-tuned CLIP accuracy.

Matched budget (mirrors exp_full_provenance.py ResNet-18: AdamW, cosine, batch 32,
30 epochs, best-validation selection, full 1295 train / 138 test, 15 seeds). CLIP
uses its native preprocessing (open_clip `preprocess` for val/test, CLIP-normalized
augmented pipeline for train); ResNet used ImageNet-native transforms -- each model
gets its native preprocessing, the fair standard.

Recipe: fine-tune `model.visual` (image ViT-L/14) + a linear head; differential LR
(vision 5e-6, head 1e-3) to avoid catastrophic forgetting on a small set; AdamW,
cosine, batch 32, 30 epochs, best-val selection, mixed precision, N seeds.

Output: data/clip_finetune.json
"""
import os, sys, json, time, argparse
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import open_clip

REPO = ROOT
PAPER1 = ROOT + "/scripts"
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, PAPER1)
from dataset import TPDDDataset  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPTS = {
    "dfn_clip": "checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin",
    "laion2b": "checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin",
}
MODEL = "ViT-L-14"
NC = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP-native normalization (ViT-L/14) for the augmented training pipeline.
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


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(wid):
    np.random.seed((int(time.time()) + wid) % (2**32))


def build_model(ckpt, grad_ckpt=True):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=ckpt)
    model = model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.visual.parameters():
        p.requires_grad = True
    if grad_ckpt and hasattr(model.visual, "set_grad_checkpointing"):
        model.visual.set_grad_checkpointing(True)  # cut activation memory for 428M ViT on 12GB
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
        correct += int((pred == lab).sum()); tot += int(lab.numel())
    return correct / max(tot, 1)


def run_ckpt(name, ckpt, seeds, epochs, batch, vis_lr, head_lr):
    model, head, preprocess = build_model(ckpt)
    train_ds = TPDDDataset("train", transform=CLIP_TRAIN_TF)
    val_ds = TPDDDataset("validation", transform=preprocess)
    test_ds = TPDDDataset("test", transform=preprocess)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0,
                          pin_memory=True, worker_init_fn=worker_init_fn)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[{name}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"seeds={seeds} epochs={epochs} batch={batch} vis_lr={vis_lr} head_lr={head_lr}", flush=True)

    vis_params = [p for p in model.visual.parameters() if p.requires_grad]
    # Snapshot the PRETRAINED visual weights once. Without this the encoder is
    # never re-initialised between seeds: seed k silently continued from seed
    # k-1's fine-tuned weights, so the "n=5 seeds" were one sequential 150-epoch
    # run and the reported std was warm-start drift, not seed variance. The
    # tell-tale was a monotonically non-decreasing best_val across seeds.
    # Snapshot on CPU: the encoder is fp32 ViT-L/14 (~1.7 GB) and this card
    # already holds weights + gradients + AdamW state. load_state_dict copies
    # CPU->GPU for us, so there is no reason to spend VRAM on the snapshot.
    pretrained_visual = {k: v.detach().cpu().clone()
                         for k, v in model.visual.state_dict().items()}
    out = {"checkpoint": name, "n_seeds": seeds, "epochs": epochs, "batch": batch,
           "vis_lr": vis_lr, "head_lr": head_lr, "seeds": []}
    for seed in range(seeds):
        set_seed(seed)
        # Restore the encoder to its pretrained weights: each seed is an
        # independent run from a common initialisation.
        model.visual.load_state_dict(pretrained_visual)
        nn.init.kaiming_uniform_(head.weight, a=5)
        torch.nn.init.zeros_(head.bias)
        opt = torch.optim.AdamW([
            {"params": vis_params, "lr": vis_lr},
            {"params": [head.weight, head.bias], "lr": head_lr, "weight_decay": 0.0},
        ], weight_decay=0.01)
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
                    loss = crit(forward(model, head, img), lab)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(vis_params + [head.weight, head.bias], 1.0)
                scaler.step(opt); scaler.update()
            # Step the cosine schedule ONCE PER EPOCH, matching the ResNet
            # control in canonical_fig_artifacts.resnet_artifacts. Stepping it
            # per batch made T_max=30 elapse in ~1 epoch (1230 steps), after
            # which the cosine formula wraps and the LR oscillates ~50x/second.
            sch.step()
            val_acc = evaluate(model, head, val_dl)
            test_acc = evaluate(model, head, test_dl)
            if val_acc > best_val:
                best_val, best_test = val_acc, test_acc
            if seed % 10 == 0 and (ep in (0, epochs // 2, epochs - 1)):
                print(f"  [{name}] seed {seed} ep {ep+1}/{epochs}: val={val_acc*100:.1f}% "
                      f"test={test_acc*100:.1f}%", flush=True)
        dt = time.time() - t0
        print(f"  [{name}] seed {seed}: best-val={best_val*100:.2f}% -> test={best_test*100:.2f}% "
              f"({dt:.0f}s)", flush=True)
        out["seeds"].append({"seed": seed, "best_val": round(float(best_val), 4),
                             "test": round(float(best_test), 4), "elapsed_s": round(dt, 1)})
    accs = [s["test"] for s in out["seeds"]]
    out["mean"] = round(float(np.mean(accs)), 4)
    out["std"] = round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0
    out["sem"] = round(float(np.std(accs, ddof=1) / np.sqrt(len(accs))), 4) if len(accs) > 1 else 0.0
    out["protocol"] = (
        f"CLIP {MODEL} image-encoder fine-tune (model.visual + linear head {NC} classes), "
        f"native CLIP preprocessing (open_clip preprocess for val/test; CLIP-normalized "
        f"RandomResizedCrop/flip/rotate/color-jitter for train), AdamW (vision {vis_lr}, "
        f"head {head_lr}, wd 0.01), cosine, batch {batch}, {epochs} ep, best-validation "
        f"selection, full TPDD train (1295) / 138 test, mixed precision, {seeds} seeds. "
        f"Matched budget to ResNet-18 (AdamW, cosine, batch 32, 30 ep, best-val) but with fewer seeds; see n_seeds here vs resnet_finetuned_ref.n_seeds."
    )
    return out


def _resnet_ref():
    """Authoritative fine-tuned ResNet-18 reference, read from the artifacts."""
    import json as _json
    from pathlib import Path as _Path
    art = _json.loads((_Path(__file__).resolve().parent.parent
                       / "data" / "canonical_fig_artifacts.json").read_text())
    r = art["resnet"]
    return {"accuracy": r["mean"],
            "n_seeds": r["n_seeds"],
            "protocol": r["protocol"]}


def _resnet_sd():
    """Seed-to-seed SD of the canonical fine-tuned ResNet-18 reference."""
    import json as _json
    from pathlib import Path as _Path
    art = _json.loads((_Path(__file__).resolve().parent.parent
                       / "data" / "canonical_fig_artifacts.json").read_text())
    return art["resnet"]["std"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--vis-lr", type=float, default=5e-6)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--which", type=str, default="both", help="both|dfn_clip|laion2b")
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "data/clip_finetune.json"))
    args = ap.parse_args()

    which = ["dfn_clip", "laion2b"] if args.which == "both" else [args.which]
    t0 = time.time()
    results = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "open_clip": open_clip.__version__,
        # Reference ResNet number is read from the canonical artifacts rather than
        # hard-coded: a constant here silently went stale when the ResNet protocol
        # was re-run, which would mis-state the margin this experiment exists to
        # measure.
        "resnet_finetuned_ref": _resnet_ref(),
        "results": {},
    }
    for name in which:
        results["results"][name] = run_ckpt(name, CKPTS[name], args.seeds, args.epochs,
                                            args.batch, args.vis_lr, args.head_lr)
    results["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== SUMMARY ===", flush=True)
    for name in which:
        r = results["results"][name]
        print(f"  {name:9s}: fine-tuned CLIP = {r['mean']*100:.2f}% +/- {r['std']*100:.2f}% "
              f"(n={r['n_seeds']})  vs  ResNet-18 "
              f"{results['resnet_finetuned_ref']['accuracy'] * 100:.2f}% +/- "
              f"{_resnet_sd() * 100:.1f}% (n={results['resnet_finetuned_ref']['n_seeds']})", flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
