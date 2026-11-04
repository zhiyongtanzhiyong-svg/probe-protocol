"""
R4 Experiments (4th round reviewer response):
1. TOST equivalence test: CLIP-C30 vs fine-tuned ResNet-18 (delta=5pp)
2. Feature geometry comparison: CLIP vs frozen ResNet-18 (inter/intra, Fisher, silhouette)
3. Exact binomial test: 14 zero-shot prompts vs majority baseline
Produces data/r4_results.json
"""
import os
import sys, json, torch, numpy as np, os, time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import pairwise_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cdist
from scipy.stats import binomtest, norm
from sklearn.metrics import silhouette_score

sys.path.insert(0, ROOT + '/scripts')

import open_clip
from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from model_zoo import ResNetBaseline, set_seed
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIP_CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS = {}


def tensor_to_pil(img_tensor):
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def extract_clip_features(model, preprocess, dataset, device):
    loader = DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    feats, labs = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image']
            clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).to(device)
            f = model.encode_image(clip_imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


def extract_resnet_features_frozen(resnet, transform, dataset, device, feat_dim=512):
    """Extract features from frozen ResNet-18 (before FC layer)."""
    loader = DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    feats, labs = [], []
    resnet.eval()
    # Remove the final FC layer to get features
    original_fc = resnet.fc
    resnet.fc = nn.Identity()
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            f = resnet(imgs)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    resnet.fc = original_fc
    return np.concatenate(feats), np.array(labs)


def compute_geometry(X, y, n_classes):
    """Compute inter/intra ratio, Fisher ratio, silhouette score."""
    # Inter-class centroid distance (mean pairwise distance between class centroids)
    centroids = []
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 0:
            centroids.append(X[mask].mean(axis=0))
    centroids = np.array(centroids)
    inter_dist = cdist(centroids, centroids, metric='euclidean')
    np.fill_diagonal(inter_dist, 0)
    mean_inter = inter_dist[~np.eye(len(centroids), dtype=bool)].mean()

    # Intra-class distance (mean within-class distance)
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

    # Fisher discriminant ratio (between-class / within-class scatter)
    # Using the trace ratio: tr(SB) / tr(SW)
    S_W = np.zeros((X.shape[1], X.shape[1]))
    global_mean = X.mean(axis=0)
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 1:
            Xc = X[mask]
            mu_c = Xc.mean(axis=0)
            S_W += (Xc - mu_c).T @ (Xc - mu_c)
    S_B = np.zeros((X.shape[1], X.shape[1]))
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 0:
            mu_c = X[mask].mean(axis=0)
            n_c = mask.sum()
            S_B += n_c * np.outer(mu_c - global_mean, mu_c - global_mean)
    fisher_ratio = np.trace(S_B) / (np.trace(S_W) + 1e-8)

    # Silhouette score (subsample for speed)
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


def tost_equivalence(pred_a, pred_b, labels, delta=0.05, n_boot=10000, seed=42):
    """
    Two One-Sided Tests (TOST) for equivalence.
    H0: |mean(A) - mean(B)| >= delta
    H1: |mean(A) - mean(B)| < delta
    Using paired bootstrap on per-sample correctness difference.
    """
    correct_a = (pred_a == labels).astype(float)
    correct_b = (pred_b == labels).astype(float)
    diff = correct_a - correct_b  # per-sample difference

    acc_a = correct_a.mean()
    acc_b = correct_b.mean()
    point_est = acc_a - acc_b

    # Paired bootstrap CI for the difference
    rng = np.random.RandomState(seed)
    n = len(labels)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_diffs.append(diff[idx].mean())
    boot_diffs = np.array(boot_diffs)
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)

    # TOST: equivalent if CI is entirely within [-delta, delta]
    equivalent = (ci_low > -delta) and (ci_high < delta)

    return {
        'acc_a': round(float(acc_a), 4),
        'acc_b': round(float(acc_b), 4),
        'point_estimate_diff_pp': round(float(point_est * 100), 2),
        'delta_pp': delta * 100,
        'boot_95ci_low_pp': round(float(ci_low * 100), 2),
        'boot_95ci_high_pp': round(float(ci_high * 100), 2),
        'tost_equivalent_at_delta': bool(equivalent),
        'interpretation': (
            f"CLIP-C30 and ResNet-18 are practically equivalent within {delta*100:.0f}pp"
            if equivalent else
            f"Cannot establish equivalence within {delta*100:.0f}pp (CI [{ci_low*100:.1f}, {ci_high*100:.1f}]pp)"
        ),
    }


def main():
    print(f"Device: {DEVICE}")
    t0 = time.time()

    # Load datasets
    print("Loading TPDD dataset...")
    train_ds = TPDDDataset(split='train', transform=TRAIN_TRANSFORM)
    test_ds = TPDDDataset(split='test', transform=EVAL_TRANSFORM)
    n_classes = len(train_ds.class_names)
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {n_classes}")

    # ============================================================
    # 1. Extract CLIP features (frozen)
    # ============================================================
    print("\n[1/4] Extracting frozen CLIP features...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained=CLIP_CKPT
    )
    clip_model = clip_model.to(DEVICE).eval()

    clip_feats_train, clip_labs_train = extract_clip_features(clip_model, preprocess, train_ds, DEVICE)
    clip_feats_test, clip_labs_test = extract_clip_features(clip_model, preprocess, test_ds, DEVICE)
    print(f"  CLIP features: train {clip_feats_train.shape}, test {clip_feats_test.shape}")

    # ============================================================
    # 2. Extract frozen ResNet-18 features (ImageNet pretrained)
    # ============================================================
    print("\n[2/4] Extracting frozen ResNet-18 features (ImageNet pretrained)...")
    resnet_frozen = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE).eval()

    # ResNet uses ImageNet normalization directly (TPDD pixels are already ImageNet-normalized)
    rn_feats_train, rn_labs_train = extract_resnet_features_frozen(resnet_frozen, None, train_ds, DEVICE)
    rn_feats_test, rn_labs_test = extract_resnet_features_frozen(resnet_frozen, None, test_ds, DEVICE)
    print(f"  ResNet features: train {rn_feats_train.shape}, test {rn_feats_test.shape}")

    # ============================================================
    # 3. TOST equivalence: CLIP-C30 vs fine-tuned ResNet
    # ============================================================
    print("\n[3/4] TOST equivalence test...")
    # Train C=30 logistic regression on CLIP features
    lr_c30 = LogisticRegression(C=30.0, max_iter=5000, random_state=42)
    lr_c30.fit(clip_feats_train, clip_labs_train)
    clip_c30_preds = lr_c30.predict(clip_feats_test)
    clip_c30_acc = accuracy_score(clip_labs_test, clip_c30_preds)
    print(f"  CLIP-C30 accuracy: {clip_c30_acc*100:.1f}%")

    # Load fine-tuned ResNet predictions (seed 1, deterministic)
    rn_full_preds = np.load(ROOT + '/data/p2_resnet_full_preds.npy')
    test_labels = np.load(ROOT + '/data/p2_test_labels.npy')
    rn_full_acc = accuracy_score(test_labels, rn_full_preds)
    print(f"  ResNet-18 (fine-tuned, seed-1) accuracy: {rn_full_acc*100:.1f}%")

    tost = tost_equivalence(clip_c30_preds, rn_full_preds, test_labels, delta=0.05, n_boot=10000)
    RESULTS['tost_equivalence'] = tost
    print(f"  TOST: {tost['interpretation']}")
    print(f"  CI: [{tost['boot_95ci_low_pp']}, {tost['boot_95ci_high_pp']}]pp")

    # ============================================================
    # 4. Feature geometry comparison
    # ============================================================
    print("\n[4/4] Feature geometry comparison...")
    clip_geom = compute_geometry(clip_feats_test, clip_labs_test, n_classes)
    rn_frozen_geom = compute_geometry(rn_feats_test, rn_labs_test, n_classes)
    print(f"  CLIP:       ratio={clip_geom['inter_intra_ratio']}, Fisher={clip_geom['fisher_ratio']}, sil={clip_geom['silhouette_score']}")
    print(f"  ResNet-fz:  ratio={rn_frozen_geom['inter_intra_ratio']}, Fisher={rn_frozen_geom['fisher_ratio']}, sil={rn_frozen_geom['silhouette_score']}")

    RESULTS['feature_geometry'] = {
        'clip_dfn': clip_geom,
        'resnet18_frozen_imagenet': rn_frozen_geom,
    }

    # Also compute on train split for consistency check
    clip_geom_train = compute_geometry(clip_feats_train, clip_labs_train, n_classes)
    rn_frozen_geom_train = compute_geometry(rn_feats_train, rn_labs_train, n_classes)
    RESULTS['feature_geometry']['clip_dfn_train'] = clip_geom_train
    RESULTS['feature_geometry']['resnet18_frozen_imagenet_train'] = rn_frozen_geom_train

    # ============================================================
    # 5. Exact binomial test: zero-shot prompts vs majority baseline
    # ============================================================
    print("\n[5/5] Exact binomial test for zero-shot prompts...")
    # Load zero-shot results
    zs_path = ROOT + '/data/zero_shot_clean.json'
    with open(zs_path) as f:
        zs = json.load(f)

    n_test = 138
    n_majority = 18  # wildfire: 18/138
    p_baseline = n_majority / n_test

    # Get per-template accuracies
    templates = zs.get('templates', zs)
    if isinstance(templates, dict):
        # Try to extract per-template accuracy
        binomial_results = {}
        for name, info in templates.items():
            if isinstance(info, dict) and 'accuracy' in info:
                n_correct = int(round(info['accuracy'] * n_test))
                bt = binomtest(n_correct, n_test, p_baseline, alternative='two-sided')
                binomial_results[name] = {
                    'n_correct': n_correct,
                    'accuracy': round(info['accuracy'], 4),
                    'p_value': round(float(bt.pvalue), 4),
                    'significant_at_005': bool(bt.pvalue < 0.05),
                }
            elif isinstance(info, (int, float)):
                n_correct = int(round(info * n_test))
                bt = binomtest(n_correct, n_test, p_baseline, alternative='two-sided')
                binomial_results[name] = {
                    'n_correct': n_correct,
                    'accuracy': round(float(info), 4),
                    'p_value': round(float(bt.pvalue), 4),
                    'significant_at_005': bool(bt.pvalue < 0.05),
                }
        RESULTS['binomial_tests'] = {
            'n_test': n_test,
            'n_majority': n_majority,
            'p_baseline': round(p_baseline, 4),
            'templates': binomial_results,
        }
    else:
        RESULTS['binomial_tests'] = {'note': 'Could not parse zero_shot_clean.json structure'}

    # ============================================================
    # Save
    # ============================================================
    RESULTS['elapsed_sec'] = round(time.time() - t0, 1)
    out_path = ROOT + '/data/r4_results.json'
    with open(out_path, 'w') as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nDone in {RESULTS['elapsed_sec']}s. Results saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"TOST: {tost['interpretation']}")
    print(f"  CLIP-C30: {tost['acc_a']*100:.1f}%, ResNet: {tost['acc_b']*100:.1f}%")
    print(f"  Boot CI for diff: [{tost['boot_95ci_low_pp']}, {tost['boot_95ci_high_pp']}]pp (delta={tost['delta_pp']}pp)")
    print(f"\nFeature Geometry (test set):")
    print(f"  {'Metric':<30} {'CLIP':>10} {'ResNet-fz':>10}")
    for key in ['mean_inter_centroid_dist', 'mean_intra_class_dist', 'inter_intra_ratio', 'fisher_ratio', 'silhouette_score']:
        print(f"  {key:<30} {clip_geom[key]:>10} {rn_frozen_geom[key]:>10}")
    print(f"\nBinomial tests (top 5):")
    if 'templates' in RESULTS.get('binomial_tests', {}):
        sorted_bt = sorted(RESULTS['binomial_tests']['templates'].items(), key=lambda x: -x[1]['accuracy'])
        for name, info in sorted_bt[:5]:
            sig = "YES" if info['significant_at_005'] else "no"
            print(f"  {name:<30} acc={info['accuracy']*100:.1f}%, p={info['p_value']:.4f} (sig: {sig})")


if __name__ == '__main__':
    main()
