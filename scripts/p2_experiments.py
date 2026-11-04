"""
P2 Experiments: McNemar, Balanced Accuracy, C-grid, Per-sample predictions.
Run once, produces data/p2_results.json with all metrics.
"""
import os
import sys, json, torch, numpy as np, time, os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, accuracy_score
)
from scipy.stats import binomtest
sys.path.insert(0, ROOT + '/scripts')

import open_clip
from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from model_zoo import ResNetBaseline, set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIP_CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def tensor_to_pil(img_tensor):
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)


class KShotSubset(Dataset):
    def __init__(self, full_ds, K, seed):
        self.transform = full_ds.transform
        rng = np.random.RandomState(seed)
        indices_by_class = {i: [] for i in range(len(full_ds.class_names))}
        for idx, lbl in enumerate(full_ds.label_ids):
            indices_by_class[lbl].append(idx)
        idxs = []
        for c, idx_list in indices_by_class.items():
            idxs.extend(rng.choice(idx_list, size=min(K, len(idx_list)), replace=False))
        self.full_ds = full_ds
        self.indices = sorted(idxs)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.full_ds[self.indices[idx]]


def clip_get_features(model, preprocess, dataset, device):
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


def resnet_predict(model, dataset, device):
    loader = DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    preds, labs = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch['image'].to(device), batch['label'].numpy()
            logits = model(imgs)
            p = logits.argmax(1).cpu().numpy()
            preds.extend(p.tolist())
            labs.extend(lbls.tolist())
    return np.array(preds), np.array(labs)


def train_resnet(train_ds, val_ds, num_classes, epochs=30, seed=0, device=None):
    set_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    model = ResNetBaseline(num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_val, best_state = 0, None
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            img, lbl = batch['image'].to(device), batch['label'].to(device)
            opt.zero_grad()
            loss = crit(model(img), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                img, lbl = batch['image'].to(device), batch['label'].to(device)
                p = model(img).argmax(1)
                correct += (p == lbl).sum().item()
                n += lbl.size(0)
        val_acc = correct / n
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None, (
        "best_state is None after training: validation accuracy was 0 at "
        "every epoch, so no checkpoint was selected. Usually the validation "
        "split is empty or misaligned, or training diverged from step 1.")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


def mcnemar_test(pred_a, pred_b, labels):
    """McNemar test on per-sample predictions.
    pred_a, pred_b, labels: arrays of length N.
    Returns: (n_disc, n_01, n_10, p_value)
    n_01: A wrong, B right (B is better)
    n_10: A right, B wrong (A is better)
    """
    a_correct = (pred_a == labels)
    b_correct = (pred_b == labels)
    n_10 = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    n_01 = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    n_disc = n_01 + n_10
    if n_disc == 0:
        return 0, 0, 0, 1.0
    # Exact binomial test (more appropriate for small n)
    bt = binomtest(min(n_01, n_10), n_disc, 0.5)
    return n_disc, n_01, n_10, float(bt.pvalue)


def compute_metrics(preds, labels, class_names):
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_f1': float(macro_f1),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    t0 = time.time()

    # Load datasets
    train_full = TPDDDataset('train', transform=TRAIN_TRANSFORM)
    val_full = TPDDDataset('validation', transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)
    N_CLASSES = len(train_full.class_names)
    class_names = train_full.class_names
    test_labels = np.array(test_ds.label_ids)
    print(f'Classes: {N_CLASSES}, test set: {len(test_ds)} images')

    # ===================== CLIP PROBE =====================
    print('\n=== CLIP Probe ===')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained=CLIP_CKPT)
    model = model.eval().to(device)

    test_feats, _ = clip_get_features(model, preprocess, test_ds, device)
    train_feats_full, train_labels_full = clip_get_features(model, preprocess, train_full, device)

    # K=1, seed=500 (matches original protocol)
    k1_subset = KShotSubset(train_full, 1, seed=500)
    k1_feats, k1_labels = clip_get_features(model, preprocess, k1_subset, device)

    # Train probes
    clf_k1 = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
    clf_k1.fit(k1_feats, k1_labels)
    clip_k1_preds = clf_k1.predict(test_feats)

    clf_full = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
    clf_full.fit(train_feats_full, train_labels_full)
    clip_full_preds = clf_full.predict(test_feats)

    del model
    torch.cuda.empty_cache()

    # ===================== RESNET-18 =====================
    print('\n=== ResNet-18 ===')
    # K=1, seed=300 (matches original protocol)
    k1_resnet_ds = KShotSubset(train_full, 1, seed=300)
    rn_k1 = train_resnet(k1_resnet_ds, val_full, N_CLASSES, epochs=30, seed=0, device=device)
    rn_k1_preds, _ = resnet_predict(rn_k1, test_ds, device)

    rn_full = train_resnet(train_full, val_full, N_CLASSES, epochs=30, seed=0, device=device)
    rn_full_preds, _ = resnet_predict(rn_full, test_ds, device)

    del rn_k1, rn_full
    torch.cuda.empty_cache()

    # ===================== C-GRID =====================
    print('\n=== C-grid (full data) ===')
    c_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    c_results = []
    for c in c_values:
        clf_c = LogisticRegression(max_iter=2000, C=c, random_state=0)
        clf_c.fit(train_feats_full, train_labels_full)
        preds_c = clf_c.predict(test_feats)
        c_results.append({
            'C': c,
            'accuracy': float(accuracy_score(test_labels, preds_c)),
        })
    best_c = max(c_results, key=lambda x: x['accuracy'])
    print(f'  Best C: {best_c["C"]} -> {best_c["accuracy"]*100:.1f}%')

    # ===================== METRICS =====================
    print('\n=== Metrics ===')
    results = {
        'n_test': int(len(test_ds)),
        'n_classes': N_CLASSES,
        'class_names': class_names,
        'class_distribution_test': {
            name: int(np.sum(test_labels == i))
            for i, name in enumerate(class_names)
        },
        'majority_class': class_names[int(np.bincount(test_labels).argmax())],
        'majority_baseline': float(np.bincount(test_labels).max() / len(test_ds)),
        'uniform_baseline': float(1.0 / N_CLASSES),
    }

    for name, preds in [
        ('clip_k1', clip_k1_preds),
        ('clip_full', clip_full_preds),
        ('resnet_k1', rn_k1_preds),
        ('resnet_full', rn_full_preds),
    ]:
        m = compute_metrics(preds, test_labels, class_names)
        m['n_correct'] = int(np.sum(preds == test_labels))
        m['n_total'] = int(len(test_labels))
        results[name] = m
        print(f'  {name:12s}: acc={m["accuracy"]*100:.1f}%  bal_acc={m["balanced_accuracy"]*100:.1f}%  macroF1={m["macro_f1"]*100:.1f}%')

    # ===================== McNemar =====================
    print('\n=== McNemar (K=1: CLIP vs ResNet) ===')
    n_disc, n_01, n_10, p_mc = mcnemar_test(clip_k1_preds, rn_k1_preds, test_labels)
    results['mcnemar_k1'] = {
        'n_discordant': n_disc,
        'n_clip_right_resnet_wrong': n_10,
        'n_resnet_right_clip_wrong': n_01,
        'p_value': p_mc,
        'significant_at_0.05': p_mc < 0.05,
    }
    print(f'  n_disc={n_disc}, n_clip_only={n_10}, n_resnet_only={n_01}, p={p_mc:.4f}')

    # McNemar full data
    n_disc_f, n_01_f, n_10_f, p_mc_f = mcnemar_test(clip_full_preds, rn_full_preds, test_labels)
    results['mcnemar_full'] = {
        'n_discordant': n_disc_f,
        'n_clip_right_resnet_wrong': n_10_f,
        'n_resnet_right_clip_wrong': n_01_f,
        'p_value': p_mc_f,
        'significant_at_0.05': p_mc_f < 0.05,
    }
    print(f'  Full: n_disc={n_disc_f}, n_clip_only={n_10_f}, n_resnet_only={n_01_f}, p={p_mc_f:.4f}')

    # Paired bootstrap on per-sample accuracy
    print('\n=== Paired bootstrap (K=1) ===')
    B = 10000
    rng = np.random.RandomState(42)
    n = len(test_labels)
    boot_diffs = np.zeros(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        d = (clip_k1_preds[idx] == test_labels[idx]).mean() - (rn_k1_preds[idx] == test_labels[idx]).mean()
        boot_diffs[b] = d
    diff_mean = float(boot_diffs.mean())
    diff_ci = np.percentile(boot_diffs, [2.5, 97.5])
    results['paired_bootstrap_k1'] = {
        'mean_diff': diff_mean,
        'ci_95': [float(diff_ci[0]), float(diff_ci[1])],
        'n_boot': B,
    }
    print(f'  diff={diff_mean*100:.2f}pp  95%CI=[{diff_ci[0]*100:.2f}, {diff_ci[1]*100:.2f}]')

    # ===================== C-GRID =====================
    results['c_grid'] = c_results
    results['c_grid_best'] = best_c

    # ===================== Save =====================
    elapsed = time.time() - t0
    results['elapsed_s'] = float(elapsed)
    results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'p2_results.json')
    json.dump(results, open(out_path, 'w'), indent=2)
    print(f'\nSaved to {out_path}')
    print(f'Elapsed: {elapsed:.0f}s')

    # Per-sample predictions (for future use)
    np.save(os.path.join(out_dir, 'p2_clip_k1_preds.npy'), clip_k1_preds)
    np.save(os.path.join(out_dir, 'p2_clip_full_preds.npy'), clip_full_preds)
    np.save(os.path.join(out_dir, 'p2_resnet_k1_preds.npy'), rn_k1_preds)
    np.save(os.path.join(out_dir, 'p2_resnet_full_preds.npy'), rn_full_preds)
    np.save(os.path.join(out_dir, 'p2_test_labels.npy'), test_labels)
    print('Per-sample predictions saved as .npy files')


if __name__ == '__main__':
    main()
