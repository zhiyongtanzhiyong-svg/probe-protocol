"""
LoRA (Low-Rank Adaptation) fine-tuning of CLIP ViT-L/14 on TPDD.

Applies LoRA adapters (r=16, alpha=32) to the out_proj of each attention layer
in the visual transformer. All original weights are frozen; only the LoRA A/B
matrices and the linear classification head are trained.

This tests whether parameter-efficient fine-tuning (training <1% of parameters)
closes the gap to full fine-tuning, directly addressing the "resource-constrained"
practical setting that is the paper's deployment framing.

Protocol: matched to clip_finetune.py (AdamW, cosine, batch 32, 30 epochs,
best-validation selection, AMP, 15 seeds).

Output: data/clip_lora_finetune.json
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
sys.path.insert(0, os.path.join(REPO, "scripts"))
from dataset import TPDDDataset  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPTS = {
    "dfn_clip": "checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin",
    "laion2b": "checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin",
}
MODEL = "ViT-L-14"
NC = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_R = 16
LORA_ALPHA = 32

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


def apply_lora(model_visual, r=LORA_R, alpha=LORA_ALPHA):
    """Apply LoRA to out_proj of every nn.MultiheadAttention in the visual transformer.

    Returns the list of trainable LoRA parameters.
    """
    scale = alpha / r
    lora_params = []

    for name, module in model_visual.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            in_dim = module.in_proj_weight.shape[1]
            out_dim = in_dim

            A = nn.Parameter(torch.empty(r, in_dim, device=next(model_visual.parameters()).device))
            B = nn.Parameter(torch.zeros(out_dim, r, device=next(model_visual.parameters()).device))
            nn.init.kaiming_uniform_(A, a=5**0.5)

            module.lora_A = A
            module.lora_B = B
            module.lora_scale = scale

            def make_hook(A, B, scale):
                def hook(mod, inp, output):
                    x = inp[0]
                    lora_out = (x @ A.T) @ B.T * scale
                    if isinstance(output, tuple):
                        return (output[0] + lora_out, output[1])
                    return output + lora_out
                return hook

            module._lora_hook_handle = module.register_forward_hook(make_hook(A, B, scale))
            lora_params.extend([A, B])

    return lora_params


def build_model(ckpt):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=ckpt)
    model = model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    lora_params = apply_lora(model.visual)

    D = model.visual.output_dim
    head = nn.Linear(D, NC).to(DEVICE)

    trainable = lora_params + [head.weight, head.bias]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)", flush=True)

    return model, head, preprocess, lora_params


@torch.no_grad()
def evaluate(model, head, loader):
    model.eval()
    correct = tot = 0
    for batch in loader:
        logits = head(model.visual(batch["image"].to(DEVICE, non_blocking=True)))
        pred = logits.argmax(1)
        lab = batch["label"].to(DEVICE, non_blocking=True)
        correct += int((pred == lab).sum()); tot += int(lab.numel())
    return correct / max(tot, 1)


def run_ckpt(name, ckpt, seeds, epochs, batch, lora_lr, head_lr):
    model, head, preprocess, lora_params = build_model(ckpt)
    train_ds = TPDDDataset("train", transform=CLIP_TRAIN_TF)
    val_ds = TPDDDataset("validation", transform=preprocess)
    test_ds = TPDDDataset("test", transform=preprocess)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0,
                          pin_memory=True, worker_init_fn=worker_init_fn)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[{name}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"seeds={seeds} epochs={epochs} batch={batch} lora_lr={lora_lr} head_lr={head_lr}", flush=True)

    # Snapshot LoRA init for seed reset
    lora_init = {p.name: p.detach().cpu().clone() for p in lora_params}
    # Build a flat list for reset
    lora_flat = [(name2, p) for name2, p in model.named_parameters() if p.requires_grad and name2.startswith("visual")]

    out = {"checkpoint": name, "n_seeds": seeds, "epochs": epochs, "batch": batch,
           "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
           "lora_lr": lora_lr, "head_lr": head_lr,
           "n_trainable_params": sum(p.numel() for p in lora_params) + head.weight.numel() + head.bias.numel(),
           "seeds": []}
    for seed in range(seeds):
        set_seed(seed)
        # Reset LoRA to init
        for p in lora_params:
            p.data.zero_()
        # Re-init A matrices
        for mod in model.visual.modules():
            if hasattr(mod, 'lora_A'):
                nn.init.kaiming_uniform_(mod.lora_A, a=5**0.5)
                mod.lora_B.data.zero_()
        nn.init.kaiming_uniform_(head.weight, a=5)
        torch.nn.init.zeros_(head.bias)

        opt = torch.optim.AdamW([
            {"params": lora_params, "lr": lora_lr},
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
                    loss = crit(head(model.visual(img)), lab)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(lora_params + [head.weight, head.bias], 1.0)
                scaler.step(opt); scaler.update()
            sch.step()
            val_acc = evaluate(model, head, val_dl)
            test_acc = evaluate(model, head, test_dl)
            if val_acc > best_val:
                best_val, best_test = val_acc, test_acc
            if (ep in (0, epochs // 2, epochs - 1)):
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
        f"CLIP {MODEL} LoRA fine-tune (r={LORA_R}, alpha={LORA_ALPHA}, out_proj adapters on "
        f"24 attention layers + linear head {NC} classes). Native CLIP preprocessing. "
        f"AdamW (LoRA {lora_lr}, head {head_lr}, wd 0.01), cosine, batch {batch}, {epochs} ep, "
        f"best-validation selection, full TPDD train (1295) / 138 test, mixed precision, {seeds} seeds."
    )
    return out


def _resnet_ref():
    import json as _json
    from pathlib import Path as _Path
    art = _json.loads((_Path(__file__).resolve().parent.parent
                       / "data" / "canonical_fig_artifacts.json").read_text())
    r = art["resnet"]
    return {"accuracy": r["mean"], "n_seeds": r["n_seeds"], "protocol": r["protocol"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--which", type=str, default="both", help="both|dfn_clip|laion2b")
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "data/clip_lora_finetune.json"))
    args = ap.parse_args()

    which = ["dfn_clip", "laion2b"] if args.which == "both" else [args.which]
    t0 = time.time()
    results = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "open_clip": open_clip.__version__,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "resnet_finetuned_ref": _resnet_ref(),
        "clip_full_finetune_ref": {},
        "results": {},
    }
    # Load full fine-tune reference for comparison
    ft_path = os.path.join(REPO, "data/clip_finetune.json")
    if os.path.exists(ft_path):
        with open(ft_path) as f:
            ft = json.load(f)
        for k, v in ft.get("results", {}).items():
            results["clip_full_finetune_ref"][k] = {"mean": v["mean"], "std": v["std"]}

    for name in which:
        results["results"][name] = run_ckpt(name, CKPTS[name], args.seeds, args.epochs,
                                            args.batch, args.lora_lr, args.head_lr)
    results["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== SUMMARY ===", flush=True)
    for name in which:
        r = results["results"][name]
        ref = results["clip_full_finetune_ref"].get(name, {})
        ref_str = f"  (full-FT: {ref.get('mean', 0)*100:.1f}%)" if ref else ""
        print(f"  {name:9s}: LoRA = {r['mean']*100:.2f}% +/- {r['std']*100:.2f}% "
              f"(n={r['n_seeds']}){ref_str}", flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
