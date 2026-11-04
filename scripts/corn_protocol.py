"""
PLANTVILLAGE (CORN, 4 classes) - 2ND-INDEPENDENT-CROP CONSISTENCY PROTOCOL.

Answers the "single-crop / single-dataset" limitation: does the TPDD pattern
  (1) zero-shot CLIP at/near baseline,
  (2) regime-dependent CLIP-vs-ResNet ordering,
  (3) full-data near-parity with a small, stable fine-tuned-CNN edge under CV,
also hold on a DIFFERENT crop? Mirrors the TPDD canonical protocol exactly so the
two datasets are directly comparable. Reuses shared CLIP extraction + ResNet recipe
+ C-grid from canonical_protocol.py / cv_protocol.py.

Data: data/plantvillage_corn/Corn___*/  (4 classes, 3852 imgs, no augmentation).
Resumable: every phase/seed/fold flushes to data/corn_results.json.
CLIP features cached to data/corn_clip_features.npz.
"""
import os
import sys, os, json, time, glob
import numpy as np, torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
PV_DIR = os.path.join(DATA_DIR, 'plantvillage_corn')
OUT = os.path.join(DATA_DIR, 'corn_results.json')
FEAT_NPZ = os.path.join(DATA_DIR, 'corn_clip_features.npz')

sys.path.insert(0, SCRIPTS)
sys.path.insert(0, ROOT + '/scripts')
from dataset import TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from model_zoo import ResNetBaseline, set_seed
from canonical_protocol import extract_clip, run_clip_protocol, C_GRID, CLIP_CKPTS, PRIMARY, tensor_to_pil
import open_clip
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SEEDS = 15
RESNET_EPOCHS = 30
BATCH = 32
SPLIT_SEED = 42

CLEAN_NAMES = {
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora (gray) leaf spot on a corn leaf',
    'Corn___Common_rust': 'common rust on a corn leaf',
    'Corn___healthy': 'a healthy corn leaf',
    'Corn___Northern_Leaf_Blight': 'northern leaf blight on a corn leaf',
}
ZERO_SHOT_TEMPLATES = [
    'a photo of a {x}', '{x}', 'a close-up photo of a {x}',
    'a corn leaf with {x}', 'a plant leaf showing {x}',
]


class CornDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = [str(p) for p in paths]
        self.labels = list(labels)
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {'image': img, 'label': self.labels[i]}


def _loader(ds, shuffle=False, workers=4):
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=workers,
                      worker_init_fn=(worker_init_fn if workers > 0 else None))


def train_resnet(train_ds, val_ds, num_classes, seed=0, epochs=RESNET_EPOCHS):
    set_seed(seed)
    model = ResNetBaseline(num_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_val, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        for batch in _loader(train_ds, shuffle=True):
            img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
            opt.zero_grad(); loss = crit(model(img), lbl); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval(); correct = n = 0
        with torch.no_grad():
            for batch in _loader(val_ds):
                img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
                p = model(img).argmax(1); correct += (p == lbl).sum().item(); n += lbl.size(0)
        val_acc = correct / n
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None, (
        "best_state is None after training: validation accuracy was 0 at "
        "every epoch, so no checkpoint was selected. Usually the validation "
        "split is empty or misaligned, or training diverged from step 1.")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model.eval()


def eval_model(model, ds):
    model.eval(); correct = n = 0
    with torch.no_grad():
        for batch in _loader(ds):
            img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
            p = model(img).argmax(1); correct += (p == lbl).sum().item(); n += lbl.size(0)
    return correct / n


def _save(R):
    with open(OUT, 'w') as f:
        json.dump(R, f, indent=2)


def _load():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {}


def build_split():
    class_dirs = sorted(glob.glob(os.path.join(PV_DIR, 'Corn___*')))
    class_dirs = [d for d in class_dirs if os.path.isdir(d) and CLEAN_NAMES.get(os.path.basename(d))]
    name_to_idx = {os.path.basename(d): i for i, d in enumerate(class_dirs)}
    paths, labels = [], []
    for d in class_dirs:
        cn = os.path.basename(d)
        for p in sorted(glob.glob(os.path.join(d, '*.*'))):
            if p.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(p); labels.append(name_to_idx[cn])
    paths = np.array(paths); labels = np.array(labels)
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(paths))
    paths, labels = paths[perm], labels[perm]
    tr_l = []; va_l = []; te_l = []
    for c in range(len(name_to_idx)):
        idx = np.where(labels == c)[0]
        n_tr = int(round(len(idx) * 0.82)); n_va = int(round(len(idx) * 0.09))
        tr_l += list(idx[:n_tr]); va_l += list(idx[n_tr:n_tr + n_va]); te_l += list(idx[n_tr + n_va:])
    return name_to_idx, paths, labels, np.array(tr_l), np.array(va_l), np.array(te_l)


def load_clip():
    m, _, pre = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CLIP_CKPTS[PRIMARY])
    m = m.half()            # cast to fp16 on CPU first (~800MB), then move
    return m.to(DEVICE).eval(), pre


def extract_half(model, preprocess, dataset, device, batch=8):
    """fp16 CLIP image encoding, small batch, no worker forks (memory-safe)."""
    feats, labs = [], []
    with torch.no_grad():
        for b in DataLoader(dataset, batch, shuffle=False, num_workers=0):
            imgs = b['image']
            clip_imgs = torch.stack([preprocess(tensor_to_pil(x)) for x in imgs]).to(device).half()
            f = model.encode_image(clip_imgs)
            f = (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()
            feats.append(f); labs.extend(b['label'].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


def get_feats(R, name_to_idx, paths, labels, tr_i, va_i, te_i):
    if os.path.exists(FEAT_NPZ):
        z = np.load(FEAT_NPZ)
        # Fingerprint the cache against the split it must correspond to. Without
        # it, adding or removing one file under plantvillage_corn/ changes
        # te_i/tr_i while the cached features stay, and the two are paired with
        # no error and no shape change.
        want_tr = [paths[i] for i in tr_i]
        want_te = [paths[i] for i in te_i]
        if ("n_train" in z and "n_test" in z
                and int(z["n_train"]) == len(want_tr)
                and int(z["n_test"]) == len(want_te)
                and list(z["train_paths"]) == want_tr
                and list(z["test_paths"]) == want_te):
            return z['Xtr'], z['ytr'], z['Xte'], z['yte']
        print("  [cache] plantvillage feature cache does not match the current "
              "file listing; recomputing.", flush=True)
    m, pre = load_clip()
    Xtr, ytr = extract_half(m, pre, CornDataset(paths[tr_i], labels[tr_i], EVAL_TRANSFORM), DEVICE)
    Xte, yte = extract_half(m, pre, CornDataset(paths[te_i], labels[te_i], EVAL_TRANSFORM), DEVICE)
    del m; torch.cuda.empty_cache()
    np.savez(FEAT_NPZ, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
             n_train=len(tr_i), n_test=len(te_i),
             train_paths=np.array([paths[i] for i in tr_i]),
             test_paths=np.array([paths[i] for i in te_i]))
    return Xtr, ytr, Xte, yte


def phase0_split_neardup(R):
    name_to_idx, paths, labels, tr_i, va_i, te_i = build_split()
    n_classes = len(name_to_idx)
    R['data'] = {
        'n_images': int(len(paths)), 'n_classes': int(n_classes),
        'class_names': [CLEAN_NAMES[k] for k in sorted(name_to_idx, key=name_to_idx.get)],
        'class_counts': {CLEAN_NAMES[k]: int((labels == i).sum()) for k, i in name_to_idx.items()},
        'split': {'train': int(len(tr_i)), 'val': int(len(va_i)), 'test': int(len(te_i))},
    }
    Xtr, ytr, Xte, yte = get_feats(R, name_to_idx, paths, labels, tr_i, va_i, te_i)
    Xtr_n = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    Xte_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    chunk = 512; maxcos = np.zeros(len(Xte_n))
    for s in range(0, len(Xte_n), chunk):
        maxcos[s:s + chunk] = (Xte_n[s:s + chunk] @ Xtr_n.T).max(axis=1)
    R['near_dup'] = {
        'max_cosine_test_to_train': round(float(maxcos.max()), 4),
        'frac_test_maxcos_gt_0.99': round(float((maxcos > 0.99).mean()), 4),
        'frac_test_maxcos_gt_0.97': round(float((maxcos > 0.97).mean()), 4),
        'note': 'split-integrity screen mirroring TPDD; >0.99 flagged as true near-duplicate',
    }
    _save(R)
    print(f"[phase0] split={R['data']['split']} near-dup max={R['near_dup']['max_cosine_test_to_train']} "
          f"frac>0.99={R['near_dup']['frac_test_maxcos_gt_0.99']}", flush=True)
    return name_to_idx, paths, labels, tr_i, va_i, te_i, Xtr, ytr, Xte, yte


def phase1_zeroshot(R, name_to_idx, paths, labels, tr_i, va_i, te_i, Xte, yte):
    m, _pre = load_clip()
    text_tokens = []
    for t in ZERO_SHOT_TEMPLATES:
        text_tokens.append([t.format(x=CLEAN_NAMES[k]) for k in sorted(name_to_idx, key=name_to_idx.get)])
    with torch.no_grad():
        text_feats = []
        for toks in text_tokens:
            tt = open_clip.tokenize(toks).to(DEVICE)
            tf = m.encode_text(tt); tf = (tf / tf.norm(dim=-1, keepdim=True)).float()
            text_feats.append(tf.cpu().numpy())
    del m; torch.cuda.empty_cache()
    Xte_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    res = {}
    for ti, tmpl in enumerate(ZERO_SHOT_TEMPLATES):
        logits = Xte_n @ text_feats[ti].T
        p = logits.argmax(1)
        res[tmpl] = {'accuracy': round(float(accuracy_score(yte, p)), 4),
                     'balanced_accuracy': round(float(balanced_accuracy_score(yte, p)), 4)}
    from collections import Counter
    maj = Counter(yte.tolist()).most_common(1)[0][0]
    res['__majority_baseline'] = {'accuracy': round(float((yte == maj).mean()), 4),
                                  'class': CLEAN_NAMES[sorted(name_to_idx, key=name_to_idx.get)[maj]]}
    res['__uniform_baseline'] = {'accuracy': round(1.0 / len(name_to_idx), 4)}
    R['zero_shot'] = res
    _save(R)
    print(f"[phase1] zero-shot: " + ", ".join(f"{k}={v['accuracy']*100:.1f}%" for k, v in res.items()), flush=True)


def _clip_probe(X_fit, y_fit, X_te, y_te):
    lr = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    lr.fit(X_fit, y_fit); p = lr.predict(X_te)
    return {'accuracy': round(float(accuracy_score(y_te, p)), 4),
            'balanced_accuracy': round(float(balanced_accuracy_score(y_te, p)), 4),
            'macro_f1': round(float(f1_score(y_te, p, average='macro')), 4)}


def phase2_fewshot(R, name_to_idx, paths, labels, tr_i, va_i, te_i, Xtr, ytr, Xte, yte):
    n_classes = len(name_to_idx)
    pool_paths = paths[tr_i]   # train-pool paths, aligned with ytr / Xtr (sel indexes the pool)
    test_ds = CornDataset(paths[te_i], labels[te_i], EVAL_TRANSFORM)
    fs = {}
    for K in [1, 5, 10]:
        clip_accs = []; resnet_accs = []
        for s in range(N_SEEDS):
            seed = 1000 + K * 100 + s
            rng = np.random.RandomState(seed)
            sel = []
            for c in range(n_classes):
                idx = np.where(ytr == c)[0]
                sel += list(rng.choice(idx, size=min(K, len(idx)), replace=False))
            sel = np.array(sorted(sel))
            clip_accs.append(_clip_probe(Xtr[sel], ytr[sel], Xte, yte)['accuracy'])
            sup_ds = CornDataset(pool_paths[sel], ytr[sel], TRAIN_TRANSFORM)
            val_ds = CornDataset(paths[va_i], labels[va_i], EVAL_TRANSFORM)
            model = train_resnet(sup_ds, val_ds, n_classes, seed=seed)
            resnet_accs.append(round(float(eval_model(model, test_ds)), 4))
            del model; torch.cuda.empty_cache()
        fs[f'K={K}'] = {
            'n_support': int(K * n_classes),
            'clip_probe_C1': {'mean': round(float(np.mean(clip_accs)), 4),
                              'sd': round(float(np.std(clip_accs, ddof=1)), 4), 'per_seed': clip_accs},
            'resnet_finetuned': {'mean': round(float(np.mean(resnet_accs)), 4),
                                 'sd': round(float(np.std(resnet_accs, ddof=1)), 4), 'per_seed': resnet_accs},
            'gap_clip_minus_resnet': round(float(np.mean(clip_accs) - np.mean(resnet_accs)), 4),
        }
        print(f"[phase2] K={K}: CLIP {np.mean(clip_accs)*100:.1f}% vs ResNet {np.mean(resnet_accs)*100:.1f}% "
              f"(gap {fs[f'K={K}']['gap_clip_minus_resnet']*100:+.1f}pp)", flush=True)
        R['few_shot'] = fs; _save(R)
    _save(R)


def phase3_fulldata(R, name_to_idx, paths, labels, tr_i, va_i, te_i, Xtr, ytr, Xte, yte):
    n_classes = len(name_to_idx)
    test_ds = CornDataset(paths[te_i], labels[te_i], EVAL_TRANSFORM)
    val_ds = CornDataset(paths[va_i], labels[va_i], EVAL_TRANSFORM)
    train_ds = CornDataset(paths[tr_i], labels[tr_i], TRAIN_TRANSFORM)
    proto = run_clip_protocol(Xtr, ytr, Xte, yte)
    clip_full = {
        'clip_C1': round(proto['linear_C1']['accuracy'], 4),
        'clip_bestC': round(proto['linear_bestC']['accuracy'], 4), 'best_C': float(proto['best_C']),
        'clip_MLP': round(proto['mlp']['accuracy'], 4),
        'clip_knn1': round(proto['knn1']['accuracy'], 4),
    }
    print(f"[phase3] CLIP full-data: C1={clip_full['clip_C1']*100:.1f}% "
          f"bestC(C={clip_full['best_C']})={clip_full['clip_bestC']*100:.1f}% MLP={clip_full['clip_MLP']*100:.1f}%", flush=True)
    resnet_accs = []
    for s in range(N_SEEDS):
        seed = 42 + s
        model = train_resnet(train_ds, val_ds, n_classes, seed=seed)
        acc = round(float(eval_model(model, test_ds)), 4)
        resnet_accs.append(acc)
        del model; torch.cuda.empty_cache()
        print(f"[phase3] ResNet seed {s}: {acc*100:.1f}%", flush=True)
        R['full_data'] = {
            'clip': clip_full,
            'resnet': {'mean': round(float(np.mean(resnet_accs)), 4),
                       'sd': round(float(np.std(resnet_accs, ddof=1)), 4), 'per_seed': resnet_accs},
            'gap_clip_minus_resnet': round(float(clip_full['clip_bestC'] - np.mean(resnet_accs)), 4),
            'n_seeds_done': len(resnet_accs),
        }
        _save(R)
    R['full_data']['gap_clip_minus_resnet'] = round(float(clip_full['clip_bestC'] - np.mean(resnet_accs)), 4)
    _save(R)


def phase4_cv(R, name_to_idx, paths, labels, Xtr, ytr, tr_i):
    n_classes = len(name_to_idx)
    pool_paths = paths[tr_i]   # train-pool paths, aligned with ytr / Xtr (fold idx indexes the pool)
    n = len(ytr)
    print(f"[phase4] CV pool = {n} imgs (train-pool paths aligned with ytr)", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = {}
    for fi, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), ytr)):
        X_te, y_te = Xtr[test_idx], ytr[test_idx]
        X_fit_all, y_fit_all = Xtr[train_idx], ytr[train_idx]
        idx_f, idx_v = train_test_split(np.arange(len(train_idx)), test_size=0.2, random_state=42, stratify=y_fit_all)
        clip_res = _clip_probe(X_fit_all[idx_f], y_fit_all[idx_f], X_te, y_te)
        # best-C on val
        val_accs = {}
        for C in C_GRID:
            lr = LogisticRegression(C=C, max_iter=5000, random_state=42)
            lr.fit(X_fit_all[idx_f], y_fit_all[idx_f]); val_accs[C] = float(accuracy_score(y_fit_all[idx_v], lr.predict(X_fit_all[idx_v])))
        best_C = max(val_accs, key=val_accs.get)
        lr_b = LogisticRegression(C=best_C, max_iter=5000, random_state=42)
        lr_b.fit(X_fit_all[idx_f], y_fit_all[idx_f])
        clip_best = round(float(accuracy_score(y_te, lr_b.predict(X_te))), 4)
        fit_img = [int(train_idx[i]) for i in idx_f]; val_img = [int(train_idx[i]) for i in idx_v]; test_img = [int(i) for i in test_idx]
        train_ds = CornDataset([pool_paths[i] for i in fit_img], y_fit_all[idx_f], TRAIN_TRANSFORM)
        val_ds = CornDataset([pool_paths[i] for i in val_img], y_fit_all[idx_v], EVAL_TRANSFORM)
        test_ds = CornDataset([pool_paths[i] for i in test_img], y_te, EVAL_TRANSFORM)
        model = train_resnet(train_ds, val_ds, n_classes, seed=42)
        rn_acc = round(float(eval_model(model, test_ds)), 4)
        del model; torch.cuda.empty_cache()
        folds[str(fi)] = {'clip_C1': round(clip_res['accuracy'], 4), 'clip_bestC': clip_best, 'best_C': float(best_C),
                          'resnet': rn_acc, 'gap': round(clip_best - rn_acc, 4)}
        print(f"[phase4] fold {fi}: CLIP bestC={clip_best*100:.1f}%(C={best_C}) vs ResNet={rn_acc*100:.1f}% "
              f"(gap {folds[str(fi)]['gap']*100:+.1f}pp)", flush=True)
        R['cv'] = {'folds': folds, 'n_folds_done': len(folds)}; _save(R)
    gaps = [folds[str(fi)]['gap'] for fi in range(5)]
    R['cv']['summary'] = {'clip_bestC_mean': round(float(np.mean([folds[str(fi)]['clip_bestC'] for fi in range(5)])), 4),
                          'resnet_mean': round(float(np.mean([folds[str(fi)]['resnet'] for fi in range(5)])), 4),
                          'gap_pp_mean': round(float(np.mean(gaps) * 100), 2), 'gap_pp_per_fold': [round(g * 100, 2) for g in gaps]}
    _save(R)


def main():
    t0 = time.time()
    print(f'Device: {DEVICE} | N_SEEDS={N_SEEDS} | ResNet epochs={RESNET_EPOCHS}', flush=True)
    R = _load()
    name_to_idx, paths, labels, tr_i, va_i, te_i, Xtr, ytr, Xte, yte = phase0_split_neardup(R)
    if 'zero_shot' not in R:
        phase1_zeroshot(R, name_to_idx, paths, labels, tr_i, va_i, te_i, Xte, yte)
    if 'few_shot' not in R or len(R.get('few_shot', {})) < 3:
        phase2_fewshot(R, name_to_idx, paths, labels, tr_i, va_i, te_i, Xtr, ytr, Xte, yte)
    if 'full_data' not in R or R['full_data'].get('n_seeds_done', 0) < N_SEEDS:
        phase3_fulldata(R, name_to_idx, paths, labels, tr_i, va_i, te_i, Xtr, ytr, Xte, yte)
    if 'cv' not in R or R['cv'].get('n_folds_done', 0) < 5:
        phase4_cv(R, name_to_idx, paths, labels, Xtr, ytr, tr_i)
    R['protocol'] = ('PlantVillage corn 4 classes; fixed stratified 82/9/9 split (seed42); '
                     'zero-shot DFN-CLIP 5 plain captions; few-shot K=1/5/10 seed-paired CLIP C=1 probe '
                     'vs fine-tuned ResNet-18 (15 seeds); full-data CLIP best-C probe vs ResNet-18 (15 seeds); '
                     '5-fold CV (CLIP best-C vs ResNet 1 seed/fold). Same heads/C-grid/recipe as TPDD.')
    R['elapsed_s'] = round(time.time() - t0, 1)
    _save(R)
    print('\n=== DONE ===', flush=True)
    print(json.dumps({'zero_shot': R.get('zero_shot'), 'few_shot': {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"} for k, v in R.get("few_shot", {}).items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
