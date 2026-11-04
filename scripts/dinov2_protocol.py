"""
P2: DINOv2 ViT-L/14 frozen-feature probe (representation-generality control).

DINOv2 is a NON-CLIP, image-only self-supervised ViT-L/14 — the cleanest test that
the frozen-feature result is not an artifact of CLIP's text alignment.

Reuses the paper's canonical probe protocol (scripts/canonical_protocol.py:
run_clip_protocol = C-grid {0.01..3000} + best-C + MLP(256,64) + kNN(1)) and the
K=1/5/10 few-shot probe protocol (scripts/clip_fewshot_probe.py), with the SAME
train/test split and SAME test set. Only the feature source changes (DINOv2 ViT-L/14).

Outputs data/dinov2_protocol.json with:
  - full_data: top-1 / balanced-acc / macro-F1 (C=1, best-C, MLP, kNN)  [vs CLIP 85.5%, ResNet-18 86.4%]
  - kshot: K=1/5/10 (+full) 15-seed mean/std  [vs CLIP & ResNet K-curves]
  - geometry (train/test) for the representational comparison.
"""
import os
import sys, os, json, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import sklearn.metrics as M

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, ROOT + "/scripts")
from dataset import TPDDDataset, EVAL_TRANSFORM, TRAIN_TRANSFORM, worker_init_fn
from canonical_protocol import run_clip_protocol, compute_geometry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PTH = "checkpoints/dinov2/dinov2_vitl14_pretrain.pth"
OUT = os.path.join(REPO, "data", "dinov2_protocol.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DINOV2_EVAL = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def tensor_to_pil(img_tensor):
    from PIL import Image
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np.clip(np_img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
    return Image.fromarray((np_img * 255).astype(np.uint8))


# ---- Self-contained DINOv2 ViT-L/14 (matches the native facebook .pth state dict) ----
class _PatchEmbed(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        return self.proj(x)

class _LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * self.gamma

class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class _Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, dim * 4)
        self.ls2 = _LayerScale(dim)
    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

class DINOv2ViTL14(nn.Module):
    def __init__(self, dim=1024, depth=24, heads=16, patch=14, grid=37):
        super().__init__()
        self.patch = patch
        self.grid = grid                                            # native pos_embed grid (518/14 = 37)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))         # MAE token; unused at inference, absorbs the ckpt key
        self.patch_embed = _PatchEmbed(dim, patch)
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
    def _interp_pos(self, n_patches):
        cls = self.pos_embed[:, :1]
        pos = self.pos_embed[:, 1:]
        dim = pos.shape[2]
        old_n = self.grid
        new_n = int(round(n_patches ** 0.5))
        if new_n != old_n:
            pos = F.interpolate(pos.transpose(1, 2).reshape(1, dim, old_n, old_n),
                               size=(new_n, new_n), mode="bicubic", align_corners=False)
            pos = pos.reshape(1, dim, new_n * new_n).transpose(1, 2)
        return torch.cat([cls, pos], dim=1)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)          # B, N, dim
        n = x.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._interp_pos(n)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return x[:, 0]


def load_dinov2(ckpt_path):
    model = DINOv2ViTL14()
    sd = torch.load(ckpt_path, map_location="cpu")
    # some mirrors wrap the state dict
    if isinstance(sd, dict) and "state_dict" in sd and "blocks.0.attn.qkv" not in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DINOv2 loaded: {n_params/1e6:.1f}M params | missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", missing[:5])
    if unexpected:
        print("  unexpected sample:", unexpected[:5])
    # The previous check only looked at keys starting with "blocks", so a failure
    # to load patch_embed.proj, cls_token, pos_embed or norm would pass the
    # assert and leave the model running on zero-initialised parameters while
    # still producing a plausible-looking accuracy. Require an exact load.
    assert not missing, f"DINOv2 checkpoint incomplete, missing: {missing[:8]}"
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:8]}"
    model.eval().to(DEVICE)
    return model


def extract(model, preprocess, dataset, device):
    feats, labs = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, 16, shuffle=False, num_workers=0):
            imgs = batch["image"]
            x = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).to(device)
            f = model(x)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            feats.append(f.cpu().numpy())
            labs.extend(batch["label"].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


class KShotSubset(Dataset):
    def __init__(self, full_ds, K, seed):
        self.full_ds = full_ds
        rng = np.random.RandomState(seed)
        by_c = {i: [] for i in range(len(full_ds.class_names))}
        for idx, lbl in enumerate(full_ds.label_ids):
            by_c[lbl].append(idx)
        idxs = []
        for c, il in by_c.items():
            idxs.extend(rng.choice(il, size=min(K, len(il)), replace=False))
        self.indices = sorted(idxs)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        return self.full_ds[self.indices[i]]


def kshot_curve(model, preprocess, train_full, test_feats, test_labels, Ks=(1, 5, 10, -1), n_seeds=15):
    from sklearn.linear_model import LogisticRegression
    out = {}
    for K in Ks:
        accs = []
        for seed in range(n_seeds):
            sub = train_full if K == -1 else KShotSubset(train_full, K, seed + 500)
            tr_f, tr_l = extract(model, preprocess, sub, DEVICE)
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(tr_f, tr_l)
            accs.append(float((clf.predict(test_feats) == test_labels).mean()))
        accs = np.array(accs)
        label = f"K={K}" if K > 0 else "full"
        # At K=-1 the support set is the whole train split and the features come
        # from the deterministic EVAL transform, so every seed necessarily
        # produces the SAME number. Reporting n_seeds=15 with std 0.0 implies a
        # seed-variance measurement that was never made; report the effective
        # count instead, as coop_official.py already does.
        n_eff = 1 if K == -1 else n_seeds
        out[label] = {"mean": round(float(accs.mean()), 4), "std": round(float(accs.std(ddof=1)), 4),
                      "n_seeds": n_eff, "n_seeds_requested": n_seeds,
                      "deterministic_full_run": K == -1,
                      "accs": accs.tolist()}
        print(f"  DINOv2 probe {label:5s}: {accs.mean()*100:.1f}% +/- {accs.std(ddof=1)*100:.1f}")
    return out


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}")
    assert os.path.exists(PTH), f"missing {PTH}; run scripts/download_dinov2.py first"
    model = load_dinov2(PTH)

    train_ds = TPDDDataset(split="train", transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset(split="test", transform=EVAL_TRANSFORM)
    n_classes = len(train_ds.class_names)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {n_classes}")

    print("\n[1/3] DINOv2 frozen features (train + test)...")
    tr, tr_y = extract(model, DINOV2_EVAL, train_ds, DEVICE)
    te, te_y = extract(model, DINOV2_EVAL, test_ds, DEVICE)
    print(f"  train {tr.shape} test {te.shape}")
    del model
    torch.cuda.empty_cache()

    print("\n[2/3] Canonical probe protocol (C-grid + best-C + MLP + kNN)...")
    proto = run_clip_protocol(tr, tr_y, te, te_y)
    full = {
        "linear_C1": {k: v for k, v in proto["linear_C1"].items() if k != "preds"},
        "linear_bestC": {k: v for k, v in proto["linear_bestC"].items() if k != "preds"},
        "mlp": {k: v for k, v in proto["mlp"].items() if k != "preds"},
        "knn1": {k: v for k, v in proto["knn1"].items() if k != "preds"},
        "best_C": proto["best_C"],
        "val_accs": proto["val_accs"],
    }
    print(f"  DINOv2 C=1 {proto['linear_C1']['accuracy']*100:.1f}% | best-C(C={proto['best_C']}) "
          f"{proto['linear_bestC']['accuracy']*100:.1f}% | MLP {proto['mlp']['accuracy']*100:.1f}% | kNN(1) {proto['knn1']['accuracy']*100:.1f}%")
    print(f"  [ref] CLIP best-C 85.5% | ResNet-18 fine-tuned 86.4%")

    print("\n[3/3] K-shot probe curve (15 seeds)...")
    # reload model for K-shot (we freed it above); cheap enough to just re-extract via a fresh load
    model = load_dinov2(PTH)
    kshot = kshot_curve(model, DINOV2_EVAL, train_ds, te, te_y)
    del model
    torch.cuda.empty_cache()

    geom = {
        "dinov2_train": compute_geometry(tr, tr_y, n_classes),
        "dinov2_test": compute_geometry(te, te_y, n_classes),
    }
    R = {
        "model": "DINOv2 ViT-L/14 (facebook/dinov2-vitl14 base pretrain, LVD-142M), 1024-dim CLS, frozen",
        "protocol": "same canonical protocol as canonical_protocol.py (C-grid, best-C on 20% val seed42, MLP, kNN) + clip_fewshot_probe.py K-curve (15 seeds, C=1.0); same train/test split & test set; DINOv2 official eval transform (Resize256 bicubic -> CenterCrop224 -> ImageNet norm).",
        "n_train": int(len(tr_y)), "n_test": int(len(te_y)), "n_classes": int(n_classes),
        "full_data": full,
        "kshot": kshot,
        "geometry": geom,
        "refs": canonical_refs(),
        "note": "DINOv2 has NO text branch; this isolates the visual representation from CLIP's text alignment.",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(R, open(OUT, "w"), indent=2)
    print(f"\nDone in {R['elapsed_sec']}s -> {OUT}")



def canonical_refs():
    """Cross-backbone reference numbers, read from canonical_fig_artifacts.json.

    Hard-coding these meant every re-run of the referenced experiment left a
    stale number in the shipped artifact, and three of these artifacts are cited
    by the manuscript. Read them instead.
    """
    import json as _json
    from pathlib import Path as _P
    art = _json.loads((_P(__file__).resolve().parent.parent
                       / "data" / "canonical_fig_artifacts.json").read_text())
    return {"clip_dfn_bestC_full": art["clip"]["linear_bestC"]["accuracy"],
            "resnet18_finetuned_full": art["resnet"]["mean"]}


if __name__ == "__main__":
    main()
