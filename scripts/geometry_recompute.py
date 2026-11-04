"""
Consistent feature-geometry comparison: frozen DFN-CLIP vs frozen ImageNet ResNet-18.

WHY THIS SCRIPT EXISTS
----------------------
Round-4 reviewers (all four) flagged that Section 3.5 mixed two different
computation runs: the CLIP ratio (0.59) came from one run (why_vlms_fail) while
the ResNet ratios (0.51 train / 0.88 test) came from another (r4), and the CLIP
test-split ratio was missing. A fair comparison requires CLIP and ResNet to be
measured with the SAME metric on the SAME split.

This script extracts BOTH frozen feature sets from the SAME images (same
preprocessing pipeline as scripts/r4_experiments.py) and reports, for each
model and for BOTH the train and test splits:
  - inter/intra centroid ratio  (mean pairwise centroid dist / mean within-class dist)
  - Fisher discriminant ratio   (tr(S_B)/tr(S_W))
  - silhouette score            (Euclidean)
so that every number in the paper comes from one reproducible source.

The metric `compute_geometry` is copied VERBATIM from scripts/r4_experiments.py
so these numbers are directly comparable to data/r4_results.json.

OUTPUT
------
  data/geometry_consistent.json   (full metrics, train + test, both models)
  + a paper-ready summary table on stdout

RUN
----
  python scripts/geometry_recompute.py
(Requires the same env as r4_experiments.py: torch, torchvision, open_clip,
 and the shared dataset/model_zoo under ..)
"""
import os
import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

sys.path.insert(0, ROOT + '/scripts')
try:
    from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
except Exception as e:
    raise SystemExit(
        f"Could not import TPDDDataset from .: {e}\n"
        "Make sure the sibling project is present, or adjust the sys.path line above."
    )

import open_clip

CLIP_CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'geometry_consistent.json')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_geometry(X, y, n_classes):
    """Copied verbatim from scripts/r4_experiments.py (identical method)."""
    from scipy.spatial.distance import cdist
    from sklearn.metrics import silhouette_score
    centroids = [X[y == c].mean(axis=0) for c in range(n_classes) if (y == c).sum() > 0]
    centroids = np.array(centroids)
    inter_dist = cdist(centroids, centroids, metric='euclidean')
    np.fill_diagonal(inter_dist, 0)
    mean_inter = inter_dist[~np.eye(len(centroids), dtype=bool)].mean()
    intra_dists = []
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 1:
            D = cdist(X[mask], X[mask], metric='euclidean')
            np.fill_diagonal(D, 0)
            n_c = X[mask].shape[0]
            intra_dists.append(D[~np.eye(n_c, dtype=bool)].mean())
    mean_intra = np.mean(intra_dists)
    ratio = mean_inter / (mean_intra + 1e-8)
    S_W = np.zeros((X.shape[1], X.shape[1]))
    global_mean = X.mean(axis=0)
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 1:
            Xc = X[mask]; mu_c = Xc.mean(axis=0)
            S_W += (Xc - mu_c).T @ (Xc - mu_c)
    S_B = np.zeros((X.shape[1], X.shape[1]))
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 0:
            mu_c = X[mask].mean(axis=0); n_c = mask.sum()
            S_B += n_c * np.outer(mu_c - global_mean, mu_c - global_mean)
    fisher_ratio = np.trace(S_B) / (np.trace(S_W) + 1e-8)
    n_samples = min(len(X), 2000)
    idx = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
    # random_state matters: sklearn draws the subsample with
    # check_random_state(None).permutation(n)[:sample_size], so the TRAIN split
    # (n=1295 -> 1000 of 1295, unstratified) gave 0.0158-0.0210 across reruns of
    # identical code, a 33% swing in the reported value. The test split (n=138)
    # is unaffected because sample_size >= n makes the draw a no-op.
    sil = silhouette_score(X[idx], y[idx], metric='euclidean',
                           sample_size=min(n_samples, 1000), random_state=42)
    return {
        'mean_inter_centroid_dist': round(float(mean_inter), 4),
        'mean_intra_class_dist': round(float(mean_intra), 4),
        'inter_intra_ratio': round(float(ratio), 4),
        'fisher_ratio': round(float(fisher_ratio), 2),
        'silhouette_score': round(float(sil), 4),
        'n_classes_with_samples': int(sum(1 for c in range(n_classes) if (y == c).sum() > 0)),
    }


def tensor_to_pil(img_tensor):
    from PIL import Image
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    return Image.fromarray((np_img * 255).astype(np.uint8))


def extract_clip(model, preprocess, dataset, device):
    feats, labs = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn):
            imgs = batch['image']
            clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).to(device)
            f = model.encode_image(clip_imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


def extract_resnet_frozen(resnet, dataset, device):
    feats, labs = [], []
    resnet.eval()
    original_fc = resnet.fc
    resnet.fc = nn.Identity()
    with torch.no_grad():
        for batch in DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn):
            f = resnet(batch['image'].to(device))
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    resnet.fc = original_fc
    return np.concatenate(feats), np.array(labs)


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}")
    train_ds = TPDDDataset(split='train', transform=TRAIN_TRANSFORM)
    test_ds = TPDDDataset(split='test', transform=EVAL_TRANSFORM)
    n_classes = len(train_ds.class_names)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {n_classes}")

    print("\n[1/3] Frozen DFN-CLIP features...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CLIP_CKPT)
    clip_model = clip_model.to(DEVICE).eval()
    clip_tr, clip_tr_y = extract_clip(clip_model, preprocess, train_ds, DEVICE)
    clip_te, clip_te_y = extract_clip(clip_model, preprocess, test_ds, DEVICE)
    del clip_model

    print("[2/3] Frozen ImageNet ResNet-18 features...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE).eval()
    rn_tr, rn_tr_y = extract_resnet_frozen(resnet, train_ds, DEVICE)
    rn_te, rn_te_y = extract_resnet_frozen(resnet, test_ds, DEVICE)
    del resnet

    print("[3/3] Geometry (train + test, both models, identical metric)...")
    res = {
        'clip_dfn': {'train': compute_geometry(clip_tr, clip_tr_y, n_classes),
                     'test':  compute_geometry(clip_te, clip_te_y, n_classes)},
        'resnet18_frozen_imagenet': {'train': compute_geometry(rn_tr, rn_tr_y, n_classes),
                                     'test':  compute_geometry(rn_te, rn_te_y, n_classes)},
    }
    res['elapsed_sec'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 64)
    print(f"{'split':<8}{'model':<24}{'inter/intra':>12}{'Fisher':>10}{'silhouette':>12}")
    print("-" * 64)
    for split in ['train', 'test']:
        for name, label in [('clip_dfn', 'CLIP (DFN)'), ('resnet18_frozen_imagenet', 'ResNet-18 frozen')]:
            g = res[name][split]
            print(f"{split:<8}{label:<24}{g['inter_intra_ratio']:>12.3f}{g['fisher_ratio']:>10.2f}{g['silhouette_score']:>12.3f}")
    print("=" * 64)
    print(f"\nSaved -> {OUT}")
    print("Use these numbers in Section 3.5 (both models, both splits, one source).")


if __name__ == '__main__':
    main()
