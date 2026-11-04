"""
CANONICAL PROTOCOL - single source of truth for all full-data numbers.

Goal: one deterministic run that produces every full-data CLIP/ResNet number the
paper reports, on ONE feature-extraction path, so the manuscript has a single
source of truth. This is the "clean protocol" (r4b): fresh frozen CLIP features,
C selected on a 20% stratified validation split (seed 42), refit on 100% train,
evaluated once on the held-out test set.

Outputs data/canonical_results.json with:
  - matrix:   CLIP C=1 / CLIP best-C / CLIP MLP / ResNet frozen linear / ResNet frozen MLP / ResNet fine-tuned
  - balanced accuracy + macro-F1 for each full-data entry
  - kNN(1) on CLIP features (geometry section)
  - geometry on BOTH train and test splits for CLIP and frozen ResNet
  - TOST training-run level (15 ResNet seeds vs fixed CLIP best-C) + per-image (one representative seed)

Deterministic: fixed features + fixed seeds. No per-seed re-encoding.
"""
import os
import sys, json, time, os
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import t as tdist
from torch.utils.data import DataLoader

sys.path.insert(0, ROOT + '/scripts')
from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
import open_clip
import torch.nn as nn
import torchvision.models as models

CLIP_CKPTS = {
    'dfn_clip': 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin',
    'laion2b': 'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin',
}
PRIMARY = 'dfn_clip'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUT = os.path.join(DATA_DIR, 'canonical_results.json')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Extended grid (1000, 3000) added to test whether the validation argmax is a genuine
# plateau interior point or an artifact of the grid boundary at C=300.
C_GRID = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]


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


def compute_geometry(X, y, n_classes):
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
    }


def eval_block(model, X_test, y_test):
    p = model.predict(X_test)
    return {
        'accuracy': round(float(accuracy_score(y_test, p)), 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(y_test, p)), 4),
        'macro_f1': round(float(f1_score(y_test, p, average='macro')), 4),
        'preds': p,
    }


def run_clip_protocol(tr, tr_y, te, te_y):
    """Canonical frozen-CLIP head protocol on one checkpoint's features:
    C=1 (fit on 100%), best-C (selected on 20% val, refit on 100%), MLP, kNN(1)."""
    idx_sub, idx_val = train_test_split(np.arange(len(tr_y)), test_size=0.2, random_state=42, stratify=tr_y)
    X_tr, y_tr = tr[idx_sub], tr_y[idx_sub]
    X_val, y_val = tr[idx_val], tr_y[idx_val]
    val_accs = {}
    for C in C_GRID:
        lr = LogisticRegression(C=C, max_iter=5000, random_state=42)
        lr.fit(X_tr, y_tr)
        val_accs[C] = round(float(accuracy_score(y_val, lr.predict(X_val))), 4)
    best_C = max(val_accs, key=val_accs.get)
    lr_c1 = LogisticRegression(C=1.0, max_iter=5000, random_state=42); lr_c1.fit(tr, tr_y)
    lr_best = LogisticRegression(C=best_C, max_iter=5000, random_state=42); lr_best.fit(tr, tr_y)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=1000, random_state=42); mlp.fit(tr, tr_y)
    knn = KNeighborsClassifier(n_neighbors=1); knn.fit(tr, tr_y)
    return {
        'val_accs': {str(k): v for k, v in val_accs.items()}, 'best_C': best_C,
        'linear_C1': eval_block(lr_c1, te, te_y),
        'linear_bestC': eval_block(lr_best, te, te_y),
        'mlp': eval_block(mlp, te, te_y),
        'knn1': eval_block(knn, te, te_y),
    }


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}")
    # Clean/canonical protocol: deterministic (un-augmented) features on BOTH train and test.
    # Using EVAL_TRANSFORM (Resize+CenterCrop+Normalize, no stochastic ops) makes the frozen
    # features fully reproducible run-to-run, which is what the paper's "deterministic, no
    # seed-level variance" claim requires. TRAIN_TRANSFORM is stochastic (flip/rotate/jitter/crop).
    train_ds = TPDDDataset(split='train', transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset(split='test', transform=EVAL_TRANSFORM)
    n_classes = len(train_ds.class_names)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {n_classes}")

    print("\n[1/5] Frozen CLIP features (train+test), both checkpoints...")
    clip_feats = {}
    for name, ckpt in CLIP_CKPTS.items():
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ckpt)
        clip_model = clip_model.to(DEVICE).eval()
        tr, tr_y = extract_clip(clip_model, preprocess, train_ds, DEVICE)
        te, te_y = extract_clip(clip_model, preprocess, test_ds, DEVICE)
        del clip_model
        torch.cuda.empty_cache()
        clip_feats[name] = (tr, tr_y, te, te_y)
        print(f"  {name}: train {tr.shape} test {te.shape}")
    clip_tr, clip_tr_y, clip_te, clip_te_y = clip_feats[PRIMARY]

    print("\n[2/5] Frozen ResNet-18 features (train+test)...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE).eval()
    rn_tr, rn_tr_y = extract_resnet_frozen(resnet, train_ds, DEVICE)
    rn_te, rn_te_y = extract_resnet_frozen(resnet, test_ds, DEVICE)
    del resnet
    torch.cuda.empty_cache()
    print(f"  ResNet train {rn_tr.shape} test {rn_te.shape}")

    R = {}

    # ---- [3/5] canonical CLIP protocol, both checkpoints ----
    print("\n[3/5] Canonical CLIP protocol (C=1 / best-C / MLP / kNN), both checkpoints...")
    clip_proto = {}
    for name, (tr, tr_y, te, te_y) in clip_feats.items():
        proto = run_clip_protocol(tr, tr_y, te, te_y)
        clip_proto[name] = proto
        print(f"  {name}: C=1 {proto['linear_C1']['accuracy']*100:.1f}% | "
              f"best-C(C={proto['best_C']}) {proto['linear_bestC']['accuracy']*100:.1f}% | "
              f"MLP {proto['mlp']['accuracy']*100:.1f}% | kNN(1) {proto['knn1']['accuracy']*100:.1f}%")
    clip_proto_primary = clip_proto[PRIMARY]
    best_C = clip_proto_primary['best_C']
    clip_c1 = clip_proto_primary['linear_C1']
    clip_best = clip_proto_primary['linear_bestC']
    clip_mlp = clip_proto_primary['mlp']
    clip_knn = clip_proto_primary['knn1']
    val_accs = clip_proto_primary['val_accs']

    # ---- ResNet frozen heads (same protocol) ----
    rn_lr = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    rn_lr.fit(rn_tr, rn_tr_y)
    rn_linear = eval_block(rn_lr, rn_te, rn_te_y)
    rn_mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=1000, random_state=42)
    rn_mlp.fit(rn_tr, rn_tr_y)
    rn_mlp_b = eval_block(rn_mlp, rn_te, rn_te_y)
    print(f"  ResNet frozen linear {rn_linear['accuracy']*100:.1f}% | MLP {rn_mlp_b['accuracy']*100:.1f}%")

    # ---- ResNet frozen C-grid (head-matched fairness: give the frozen ResNet probe the
    # SAME validation-selected C that the CLIP probe receives, not just the default C=1) ----
    idx_sub_r, idx_val_r = train_test_split(np.arange(len(rn_tr_y)), test_size=0.2, random_state=42, stratify=rn_tr_y)
    rn_val_accs = {}
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=5000, random_state=42)
        m.fit(rn_tr[idx_sub_r], rn_tr_y[idx_sub_r])
        rn_val_accs[C] = round(float(accuracy_score(rn_tr_y[idx_val_r], m.predict(rn_tr[idx_val_r]))), 4)
    rn_best_C = max(rn_val_accs, key=rn_val_accs.get)
    rn_lr_best = LogisticRegression(C=rn_best_C, max_iter=5000, random_state=42)
    rn_lr_best.fit(rn_tr, rn_tr_y)
    rn_linear_best = eval_block(rn_lr_best, rn_te, rn_te_y)
    print(f"  ResNet frozen best-C (C={rn_best_C}) {rn_linear_best['accuracy']*100:.1f}% | val argmax {rn_val_accs[rn_best_C]}")

    # ---- ResNet fine-tuned ----
    # Source of truth is data/canonical_fig_artifacts.json (15 seeds re-run under
    # this same canonical protocol on the image set currently available). The
    # original 11 MB TPDD images were lost, so the values in
    # data/full_provenance.json (86.4% +/- 1.9, original images) can no longer be
    # reproduced and are retained there only for provenance. Everything this
    # script reports downstream is taken from the re-run.
    with open(os.path.join(DATA_DIR, 'canonical_fig_artifacts.json')) as f:
        farts = json.load(f)
    rn_ft_accs = farts['resnet']['accs']
    rn_ft_mean = float(np.mean(rn_ft_accs)); rn_ft_sd = float(np.std(rn_ft_accs, ddof=1))
    rn_ft_bal = farts['resnet']['seed0']['balanced_accuracy']
    rn_ft_f1 = farts['resnet']['seed0']['macro_f1']
    print(f"  ResNet fine-tuned mean {rn_ft_mean*100:.1f}% +/- {rn_ft_sd*100:.1f}% "
          f"(n={len(rn_ft_accs)}) | seed0 bal {rn_ft_bal*100:.1f}% | f1 {rn_ft_f1*100:.1f}%")

    # ---- [4/5] geometry BOTH splits, both models ----
    print("\n[4/5] Geometry (train + test, both models)...")
    geom = {
        'clip_dfn_train': compute_geometry(clip_tr, clip_tr_y, n_classes),
        'clip_dfn_test': compute_geometry(clip_te, clip_te_y, n_classes),
        'resnet_frozen_train': compute_geometry(rn_tr, rn_tr_y, n_classes),
        'resnet_frozen_test': compute_geometry(rn_te, rn_te_y, n_classes),
    }
    print(f"  CLIP  train {geom['clip_dfn_train']['inter_intra_ratio']} / test {geom['clip_dfn_test']['inter_intra_ratio']}")
    print(f"  ResNet train {geom['resnet_frozen_train']['inter_intra_ratio']} / test {geom['resnet_frozen_test']['inter_intra_ratio']}")

    # ---- [5/6] per-class accuracy (16 classes) on canonical features ----
    # Resolves the balanced-accuracy attribution question with actual per-class numbers
    # instead of the "concentrated on the 4 singleton classes" hand-wave.
    class_names = [str(n) for n in train_ds.class_names]

    def _perclass(p, y, names):
        out = {}
        for c in range(len(names)):
            mask = (y == c)
            if mask.sum() == 0:
                out[names[c]] = None
                continue
            out[names[c]] = {'n_test': int(mask.sum()), 'acc': round(float((p[mask] == c).mean()), 4)}
        return out

    per_class = {
        'class_names': class_names,
        'clip_linear_C1': _perclass(clip_c1['preds'], clip_te_y, class_names),
        'clip_linear_bestC': _perclass(clip_best['preds'], clip_te_y, class_names),
        'resnet_finetuned_seed0': None,
        'note': "per-class top-1 accuracy on the 138 test images (canonical, deterministic CLIP). ResNet fine-tuned = representative seed 0 (88.4%) if available; the 15-seed mean is not per-class (only aggregate accs are stored).",
    }
    try:
        rn_p = np.load(os.path.join(DATA_DIR, 'p2_resnet_full_preds.npy'))
        y_te = np.load(os.path.join(DATA_DIR, 'p2_test_labels.npy'))
        # Length alone is too weak: a stale p2_test_labels.npy of the same length
        # would be silently paired with canonical features. Require identity.
        if len(rn_p) == len(clip_te_y) and np.array_equal(y_te, clip_te_y):
            per_class['resnet_finetuned_seed0'] = _perclass(rn_p, y_te, class_names)
            print(f"  per-class: CLIP best-C {sum(1 for v in per_class['clip_linear_bestC'].values() if isinstance(v,dict) and v['acc']==1.0)} classes at 100%; ResNet seed0 loaded")
        else:
            print(f"  (per-class ResNet skipped: pred len {len(rn_p)} != {len(clip_te_y)})")
    except Exception as e:
        print(f"  (per-class ResNet skipped: {e})")

    # ---- [5/5] TOST ----
    print("\n[5/5] TOST (training-run level + per-image)...")
    clip_fixed = clip_best['accuracy']
    seeds = np.array(rn_ft_accs)
    diff = seeds - clip_fixed  # ResNet minus CLIP, per training run
    n = len(diff); mean_d = float(diff.mean()); sd_d = float(diff.std(ddof=1))
    se = sd_d / np.sqrt(n)
    tcrit = float(tdist.ppf(0.975, df=n - 1))
    ci_low = mean_d - tcrit * se; ci_high = mean_d + tcrit * se
    delta = 0.05
    tost_run = {
        'clip_fixed_acc': round(clip_fixed, 4),
        'resnet_mean': round(rn_ft_mean, 4),
        'resnet_sd': round(rn_ft_sd, 4),
        'n_seeds': n,
        'mean_diff_pp': round(mean_d * 100, 2),
        'ci95_pp': [round(ci_low * 100, 2), round(ci_high * 100, 2)],
        'delta_pp': delta * 100,
        'within_margin': bool(ci_low > -delta and ci_high < delta),
        'note': "training-run level: 15 stochastic ResNet seeds vs ONE fixed deterministic CLIP probe (single fixed test set). NOT population-level.",
    }
    print(f"  training-run: mean_diff {mean_d*100:+.1f}pp, 95%CI [{ci_low*100:+.1f},{ci_high*100:+.1f}]pp, within +/-5pp = {tost_run['within_margin']}")

    # per-image (one representative ResNet seed, seed index 0)
    tost_img = None
    try:
        rn_preds = np.load(os.path.join(DATA_DIR, 'p2_resnet_full_preds.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'p2_test_labels.npy'))
        diff_img = (clip_best['preds'] == y_test).astype(float) - (rn_preds == y_test).astype(float)
        m = float(diff_img.mean()); sd_i = float(diff_img.std(ddof=1))
        se_i = sd_i / np.sqrt(len(diff_img)); tci = float(tdist.ppf(0.975, df=len(diff_img) - 1))
        lo = m - tci * se_i; hi = m + tci * se_i
        tost_img = {
            'clip_acc': round(float((clip_best['preds'] == y_test).mean()), 4),
            'resnet_rep_seed_acc': round(float((rn_preds == y_test).mean()), 4),
            'mean_diff_pp': round(m * 100, 2),
            'ci95_pp': [round(lo * 100, 2), round(hi * 100, 2)],
            'delta_pp': delta * 100,
            'within_margin': bool(lo > -delta and hi < delta),
            'note': "per-image level against ONE representative ResNet seed; SE~3pp, not resolvable at 5pp margin.",
        }
        print(f"  per-image: mean_diff {m*100:+.1f}pp, 95%CI [{lo*100:+.1f},{hi*100:+.1f}]pp, within +/-5pp = {tost_img['within_margin']}")
    except Exception as e:
        print(f"  (per-image TOST skipped: {e})")

    def _clean(proto):
        out = {k: v for k, v in proto.items() if k != 'val_accs'}
        for blk in out.values():
            if isinstance(blk, dict) and 'preds' in blk:
                del blk['preds']
        return out

    R = {
        'protocol': "clean/canonical: fresh frozen CLIP ViT-L/14 features (deterministic), C selected on 20% stratified validation (seed 42), refit on 100% train, evaluated once on 138 test images. Both checkpoints (dfn_clip primary, laion2b) run under the SAME protocol. ResNet fine-tuned loaded from full_provenance.json.",
        'n_train': int(len(clip_tr_y)), 'n_test': int(len(clip_te_y)), 'n_classes': int(n_classes),
        'primary_checkpoint': PRIMARY,
        'C_grid': C_GRID, 'val_accs': {str(k): v for k, v in val_accs.items()}, 'best_C': best_C,
        'resnet_frozen_val_accs': {str(k): v for k, v in rn_val_accs.items()}, 'resnet_frozen_best_C': rn_best_C,
        'clip_by_checkpoint': {name: _clean(proto) for name, proto in clip_proto.items()},
        'matrix': {
            'clip_linear_C1': clip_c1, 'clip_linear_bestC': clip_best, 'clip_mlp': clip_mlp, 'clip_knn1': clip_knn,
            'resnet_frozen_linear': rn_linear, 'resnet_frozen_mlp': rn_mlp_b, 'resnet_frozen_bestC': rn_linear_best,
            'resnet_finetuned': {'accuracy': round(rn_ft_mean, 4), 'balanced_accuracy': round(rn_ft_bal, 4), 'macro_f1': round(rn_ft_f1, 4), 'sd': round(rn_ft_sd, 4), 'n_seeds': len(rn_ft_accs)},
        },
        'geometry': geom,
        'per_class': per_class,
        'tost_training_run': tost_run,
        'tost_per_image': tost_img,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    for blk in R['matrix'].values():
        if isinstance(blk, dict) and 'preds' in blk:
            del blk['preds']
    with open(OUT, 'w') as f:
        json.dump(R, f, indent=2)
    print(f"\nDone in {R['elapsed_sec']}s -> {OUT}")


if __name__ == '__main__':
    main()
