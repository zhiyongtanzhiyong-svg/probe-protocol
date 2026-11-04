"""
P3 extension: harden the full-data fine-tuned CLIP cell from n=5 to n=15 seeds.

Reuses the exact recipe and building blocks of scripts/clip_finetune.py (same
build_model / forward / evaluate / set_seed / worker_init_fn / CLIP_TRAIN_TF,
same per-seed pretrained-weight restore + Kaiming head init + AdamW differential
LR + cosine + best-val selection), so seeds 5-14 are directly comparable to the
existing seeds 0-4 and the combined set is a clean 15-seed estimate.

Crash-safe: each finished seed is appended to a per-checkpoint JSONL file and
fsync'd immediately, so a mid-run crash does not lose completed seeds. Re-running
skips seeds already present in the JSONL.

On completion, merges the original store (seeds 0-4) with the extension
(seeds 5-14) into data/clip_finetune_n15.json with recomputed mean/std/sem.

Launch (single GPU, ~6h):
  nohup python3 scripts/clip_finetune_extend.py --seed-start 5 --n 10 \
      > /tmp/ft_extend.log 2>&1 &
"""
import os, sys, json, time, argparse
import numpy as np

REPO = ROOT
sys.path.insert(0, os.path.join(REPO, "scripts"))
import torch
from torch.utils.data import DataLoader
import clip_finetune as cf  # noqa: E402  (reuse all building blocks verbatim)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_seed(name, seed, epochs, batch, vis_lr, head_lr, model, head,
             vis_params, pretrained_visual, train_dl, val_dl, test_dl):
    """Run one seed, matching clip_finetune.run_ckpt's per-seed body exactly."""
    cf.set_seed(seed)
    model.visual.load_state_dict(pretrained_visual)
    torch.nn.init.kaiming_uniform_(head.weight, a=5)
    torch.nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW([
        {"params": vis_params, "lr": vis_lr},
        {"params": [head.weight, head.bias], "lr": head_lr, "weight_decay": 0.0},
    ], weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    best_val, best_test = -1.0, -1.0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for b in train_dl:
            img = b["image"].to(cf.DEVICE, non_blocking=True)
            lab = b["label"].to(cf.DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(cf.forward(model, head, img), lab)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(vis_params + [head.weight, head.bias], 1.0)
            scaler.step(opt)
            scaler.update()
        sch.step()
        val_acc = cf.evaluate(model, head, val_dl)
        test_acc = cf.evaluate(model, head, test_dl)
        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc
        if ep in (0, epochs // 2, epochs - 1):
            print(f"  [{name}] ext-seed {seed} ep {ep+1}/{epochs}: "
                  f"val={val_acc*100:.1f}% test={test_acc*100:.1f}%", flush=True)
    dt = time.time() - t0
    rec = {"seed": seed, "best_val": round(float(best_val), 4),
           "test": round(float(best_test), 4), "elapsed_s": round(dt, 1)}
    print(f"  [{name}] ext-seed {seed}: best-val={best_val*100:.2f}% -> "
          f"test={best_test*100:.2f}% ({dt:.0f}s)", flush=True)
    return rec


def _jsonl_append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _jsonl_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["seed"])
    return done


def _jsonl_load(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def _clean_protocol(name, epochs, batch, vis_lr, head_lr, n_seeds):
    return (
        f"CLIP ViT-L-14 image-encoder fine-tune ({name}; model.visual + linear head 16 "
        f"classes), native CLIP preprocessing (open_clip preprocess for val/test; "
        f"CLIP-normalized RandomResizedCrop/flip/rotate/color-jitter for train), AdamW "
        f"(vision {vis_lr}, head {head_lr}, wd 0.01), cosine, batch {batch}, {epochs} ep, "
        f"best-validation selection, full TPDD train (1295) / 138 test, mixed precision, "
        f"{n_seeds} independent seeds from a common pretrained initialisation. Matched "
        f"budget to ResNet-18 (AdamW, cosine, batch 32, 30 ep, best-val)."
    )


def extend(which, args, data_dir):
    from dataset import TPDDDataset  # noqa: local import keeps module import light
    for name in which:
        ckpt = cf.CKPTS[name]
        model, head, preprocess = cf.build_model(ckpt)
        pretrained_visual = {k: v.detach().cpu().clone()
                             for k, v in model.visual.state_dict().items()}
        vis_params = [p for p in model.visual.parameters() if p.requires_grad]
        train_dl = DataLoader(TPDDDataset("train", transform=cf.CLIP_TRAIN_TF),
                              batch_size=args.batch, shuffle=True, num_workers=0,
                              pin_memory=True, worker_init_fn=cf.worker_init_fn)
        val_dl = DataLoader(TPDDDataset("validation", transform=preprocess),
                            batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        test_dl = DataLoader(TPDDDataset("test", transform=preprocess),
                             batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        jsonl_path = os.path.join(data_dir, f"clip_finetune_ext_{name}.jsonl")
        done = _jsonl_done(jsonl_path)
        print(f"[{name}] extending seeds {args.seed_list} (already done: {sorted(done)})",
              flush=True)
        for seed in args.seed_list:
            if seed in done:
                print(f"  [{name}] ext-seed {seed}: already in JSONL, skipping", flush=True)
                continue
            rec = run_seed(name, seed, args.epochs, args.batch, args.vis_lr,
                           args.head_lr, model, head, vis_params, pretrained_visual,
                           train_dl, val_dl, test_dl)
            _jsonl_append(jsonl_path, rec)
        del model, head, pretrained_visual
        torch.cuda.empty_cache()


def merge(which, args, data_dir):
    src = os.path.join(data_dir, "clip_finetune.json")
    with open(src) as f:
        orig = json.load(f)
    out = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "open_clip": __import__("open_clip").__version__,
        "resnet_finetuned_ref": orig["resnet_finetuned_ref"],
        "n_seeds_target": 15,
        "note": ("Full-data fine-tuned CLIP cell extended from n=5 (seeds 0-4) to n=15 "
                 "(seeds 5-14) to tighten the CI on the decisive CLIP-vs-ResNet cell. "
                 "Recipe identical to the original store; every seed is an independent "
                 "run from a common pretrained initialisation."),
        "results": {},
    }
    for name in which:
        base = orig["results"][name]
        base_seeds = list(base["seeds"])
        ext_seeds = _jsonl_load(os.path.join(data_dir, f"clip_finetune_ext_{name}.jsonl"))
        merged = {s["seed"]: s for s in base_seeds}
        for s in ext_seeds:
            merged[s["seed"]] = s
        seeds = [merged[k] for k in sorted(merged)]
        accs = [s["test"] for s in seeds]
        n = len(accs)
        out["results"][name] = {
            "checkpoint": name,
            "n_seeds": n,
            "epochs": base.get("epochs", args.epochs),
            "batch": base.get("batch", args.batch),
            "vis_lr": base.get("vis_lr", args.vis_lr),
            "head_lr": base.get("head_lr", args.head_lr),
            "seeds": seeds,
            "mean": round(float(np.mean(accs)), 4),
            "std": round(float(np.std(accs, ddof=1)), 4) if n > 1 else 0.0,
            "sem": round(float(np.std(accs, ddof=1) / np.sqrt(n)), 4) if n > 1 else 0.0,
            "protocol": _clean_protocol(name, base.get("epochs", args.epochs),
                                        base.get("batch", args.batch),
                                        base.get("vis_lr", args.vis_lr),
                                        base.get("head_lr", args.head_lr), n),
        }
    dst = os.path.join(data_dir, "clip_finetune_n15.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n=== MERGE SUMMARY ===", flush=True)
    ref = out["resnet_finetuned_ref"]
    for name in which:
        r = out["results"][name]
        print(f"  {name:9s}: n={r['n_seeds']}  mean={r['mean']*100:.2f}%  "
              f"std={r['std']*100:.2f}%  sem={r['sem']*100:.2f}%   "
              f"(ResNet-18 ref {ref['accuracy']*100:.2f}%, n={ref['n_seeds']})",
              flush=True)
    print(f"saved {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=5)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--vis-lr", type=float, default=5e-6)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--which", type=str, default="both", help="both|dfn_clip|laion2b")
    ap.add_argument("--data-dir", type=str, default=os.path.join(REPO, "data"))
    ap.add_argument("--merge", type=int, default=1, help="1 = merge into n15 json at end")
    args = ap.parse_args()
    args.seed_list = list(range(args.seed_start, args.seed_start + args.n))
    which = ["dfn_clip", "laion2b"] if args.which == "both" else [args.which]

    t0 = time.time()
    extend(which, args, args.data_dir)
    if args.merge:
        merge(which, args, args.data_dir)
    print(f"\ndone in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
