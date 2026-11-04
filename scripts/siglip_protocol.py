"""
SigLIP protocol on TPDD — modern CLIP alternative (sigmoid loss).

Tests SigLIP (SoViT-400M-14, ~878M params) under three regimes:
zero-shot, few-shot linear probe, and full-data fine-tuning.

Uses transformers library with local checkpoint (downloaded from ModelScope).

Output: data/siglip_protocol.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from sklearn.linear_model import LogisticRegression

REPO = ROOT
sys.path.insert(0, os.path.join(REPO, "scripts"))
from dataset import TPDDDataset  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "checkpoints/siglip"
NC = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384
BATCH = 8

TEMPLATES = [
    ("EN-Symptom", "a photo of a leaf with {label} symptoms"),
    ("KG-Symptom", "a photo of a leaf showing {label} disease symptoms on tobacco"),
    ("EN-Photo", "a photo of a {label} affected tobacco leaf"),
    ("KG-Combined", "a photo of a tobacco leaf with {label} symptoms and conditions"),
    ("KG-Condition", "a photo of a leaf in the condition of {label}"),
    ("CN-Symptom", "一张患有{label}症状的叶片照片"),
    ("EN-Closeup", "a closeup photo of a leaf with {label}"),
    ("CN-Photo", "一张{label}烟草叶片的照片"),
    ("CN-Label", "{label}"),
    ("CN-Disease", "一种叫做{label}的疾病"),
    ("CN-Closeup", "一张{label}的特写照片"),
    ("EN-Disease", "a photo of a {label} disease"),
    ("EN-Expert", "an expert photo of a leaf diagnosed with {label}"),
    ("EN-Label", "{label}"),
]


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_names():
    import json as _json
    from pathlib import Path as _Path
    labels = _json.loads((_Path(REPO) / "data" / "tpdd" / "test" / "labels.json").read_text())
    if isinstance(labels, list):
        return sorted(set(item["label"] for item in labels))
    return sorted(set(labels.values()))


def get_image_features_batched(model, images, batch_size=BATCH):
    """Extract image features in batches to avoid OOM."""
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            chunk = images[i:i+batch_size].to(DEVICE)
            feats = model.get_image_features(pixel_values=chunk)
            all_feats.append(feats.cpu())
            del chunk, feats
    del images
    torch.cuda.empty_cache()
    return torch.cat(all_feats)


def zero_shot(model, image_proc, tokenizer, class_names):
    model.eval()
    # Text features
    text_features = []
    for _, tmpl in TEMPLATES:
        texts = [tmpl.format(label=c) for c in class_names]
        inputs = tokenizer(texts, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            tf = model.get_text_features(**inputs)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        text_features.append(tf.cpu())
        del inputs, tf
    text_features = torch.stack(text_features)  # (n_templates, n_classes, dim)

    # Image features
    test_ds = TPDDDataset("test", transform=image_proc)
    test_imgs = torch.stack([s["image"] for s in test_ds])
    test_labels = np.array([s["label"] for s in test_ds])

    img_feats = get_image_features_batched(model, test_imgs)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    results = []
    for i, (name, _) in enumerate(TEMPLATES):
        sim = img_feats @ text_features[i].T
        pred = sim.argmax(dim=1).cpu().numpy()
        acc = float((pred == test_labels).mean())
        results.append({"template": name, "acc": round(acc, 4)})
    from collections import Counter
    majority = Counter(test_labels).most_common(1)[0][1] / len(test_labels)
    results.append({"template": "Majority-class baseline", "acc": round(majority, 4)})
    return results


def few_shot_probe(model, image_proc, class_names, K):
    model.eval()
    train_ds = TPDDDataset("train", transform=image_proc)
    test_ds = TPDDDataset("test", transform=image_proc)

    from collections import defaultdict
    by_class = defaultdict(list)
    for i, sample in enumerate(train_ds):
        by_class[int(sample["label"])].append(i)

    rng = np.random.RandomState(42)
    support_idx = []
    for c in range(NC):
        idx = by_class[c]
        support_idx.extend(rng.choice(idx, size=min(K, len(idx)), replace=False).tolist())
    support_idx = np.array(support_idx)

    support_imgs = torch.stack([train_ds[int(i)]["image"] for i in support_idx])
    support_labels = np.array([int(train_ds[int(i)]["label"]) for i in support_idx])

    sup_feats = get_image_features_batched(model, support_imgs)
    sup_feats = sup_feats / sup_feats.norm(dim=-1, keepdim=True)
    sup_feats = sup_feats.numpy()

    test_imgs = torch.stack([s["image"] for s in test_ds])
    test_labels = np.array([s["label"] for s in test_ds])

    test_feats = get_image_features_batched(model, test_imgs)
    test_feats = test_feats / test_feats.norm(dim=-1, keepdim=True)
    test_feats = test_feats.numpy()

    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(sup_feats, support_labels)
    pred = clf.predict(test_feats)
    return float((pred == test_labels).mean())


def fine_tune(model, image_proc, seeds, epochs, batch, vis_lr, head_lr):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.vision_model.parameters():
        p.requires_grad = True
    D = model.config.vision_config.hidden_size
    head = nn.Linear(D, NC).to(DEVICE)

    train_ds = TPDDDataset("train", transform=image_proc)
    val_ds = TPDDDataset("validation", transform=image_proc)
    test_ds = TPDDDataset("test", transform=image_proc)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    vis_params = [p for p in model.vision_model.parameters() if p.requires_grad]
    pretrained = {k: v.detach().cpu().clone() for k, v in model.vision_model.state_dict().items()}

    out = {"seeds": []}
    for seed in range(seeds):
        set_seed(seed)
        model.vision_model.load_state_dict(pretrained)
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
            for batch_data in train_dl:
                img = batch_data["image"].to(DEVICE, non_blocking=True)
                lab = torch.tensor(batch_data["label"], dtype=torch.long, device=DEVICE)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    feat = model.get_image_features(pixel_values=img)
                    loss = crit(head(feat), lab)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(vis_params + [head.weight, head.bias], 1.0)
                scaler.step(opt); scaler.update()
            sch.step()
            model.eval()
            with torch.no_grad():
                val_c = val_t = 0
                for b in val_dl:
                    feat = model.get_image_features(pixel_values=b["image"].to(DEVICE))
                    pred = head(feat).argmax(1)
                    lab = torch.tensor(b["label"], dtype=torch.long, device=DEVICE)
                    val_c += int((pred == lab).sum())
                    val_t += lab.numel()
                test_c = test_t = 0
                for b in test_dl:
                    feat = model.get_image_features(pixel_values=b["image"].to(DEVICE))
                    pred = head(feat).argmax(1)
                    lab = torch.tensor(b["label"], dtype=torch.long, device=DEVICE)
                    test_c += int((pred == lab).sum())
                    test_t += lab.numel()
            val_acc = val_c / max(val_t, 1)
            test_acc = test_c / max(test_t, 1)
            if val_acc > best_val:
                best_val, best_test = val_acc, test_acc
            if (ep in (0, epochs // 2, epochs - 1)):
                print(f"  seed {seed} ep {ep+1}/{epochs}: val={val_acc*100:.1f}% test={test_acc*100:.1f}%", flush=True)
        dt = time.time() - t0
        print(f"  seed {seed}: best-val={best_val*100:.2f}% -> test={best_test*100:.2f}% ({dt:.0f}s)", flush=True)
        out["seeds"].append({"seed": seed, "best_val": round(float(best_val), 4),
                             "test": round(float(best_test), 4), "elapsed_s": round(dt, 1)})
    accs = [s["test"] for s in out["seeds"]]
    out["mean"] = round(float(np.mean(accs)), 4)
    out["std"] = round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0
    out["sem"] = round(float(np.std(accs, ddof=1) / np.sqrt(len(accs))), 4) if len(accs) > 1 else 0.0
    out["protocol"] = (
        f"SigLIP (SoViT-400M-14, sigmoid loss, 878M params) fine-tuned "
        f"(vision_model + linear head {NC} classes), "
        f"AdamW (vision {vis_lr}, head {head_lr}, wd 0.01), cosine, batch {batch}, {epochs} ep, "
        f"best-val selection, full TPDD train (1295) / 138 test, AMP, {seeds} seeds."
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vis-lr", type=float, default=5e-6)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "data/siglip_protocol.json"))
    args = ap.parse_args()

    print(f"Loading SigLIP from {MODEL_PATH}", flush=True)
    model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Use standard torchvision transforms (SigLIP normalizes with 0.5/0.5)
    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)
    image_proc = T.Compose([
        T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}", flush=True)

    class_names = get_class_names()
    print(f"  classes: {len(class_names)}", flush=True)

    t0 = time.time()
    results = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "model": "siglip-so400m-patch14-384",
        "n_params": n_params,
    }

    print("\n=== Zero-shot ===", flush=True)
    results["zero_shot"] = zero_shot(model, image_proc, tokenizer, class_names)
    best_zs = max(r["acc"] for r in results["zero_shot"] if "baseline" not in r["template"])
    print(f"  best zero-shot: {best_zs*100:.1f}%", flush=True)

    print("\n=== Few-shot probe ===", flush=True)
    results["fewshot_probe"] = {}
    for K in [1, 5, 10]:
        acc = few_shot_probe(model, image_proc, class_names, K)
        results["fewshot_probe"][f"K={K}"] = round(acc, 4)
        print(f"  K={K}: {acc*100:.1f}%", flush=True)

    if not args.skip_finetune:
        print("\n=== Full fine-tune ===", flush=True)
        results["finetune"] = fine_tune(model, image_proc, args.seeds, args.epochs,
                                        args.batch, args.vis_lr, args.head_lr)
        print(f"  fine-tuned: {results['finetune']['mean']*100:.2f}% +/- {results['finetune']['std']*100:.2f}%", flush=True)

    results["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  SigLIP (SoViT-400M, 878M params):", flush=True)
    print(f"    zero-shot best: {best_zs*100:.1f}%", flush=True)
    for k, v in results["fewshot_probe"].items():
        print(f"    {k} probe: {v*100:.1f}%", flush=True)
    if "finetune" in results:
        print(f"    fine-tuned: {results['finetune']['mean']*100:.2f}% +/- {results['finetune']['std']*100:.2f}%", flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
