"""
5-FOLD STRATIFIED CROSS-VALIDATION over the TPDD training pool.

Answers reviewer point #11 ("no cross-validation; a single split may be a fluke"):
is the CLIP-vs-ResNet gap stable across random splits of the training data?

Protocol (per fold, symmetric and honest):
  pool      = the DISJOINT 'train' split (1295 imgs, 16 classes)
  folds     = StratifiedKFold(5, shuffle, seed 42) over pool labels
  per fold:
    train_part = 4 folds (held-out fold is the test fold)
    fit/val    = 80/20 stratified carve-out of train_part (seed 42)
    CLIP       = frozen features (pre-extracted once, canonical path),
                 LogReg C selected on val, fit on fit-split -> eval on test fold
    ResNet-18  = ImageNet-pretrained, AdamW lr=3e-4, cosine, bs32, 30 ep,
                 best-val on val-split -> eval on test fold
  Both are trained on the SAME fit-split, so the comparison is not confounded by a
  difference in training-set size. 1 seed per fold: this CV measures SPLIT stability
  (the reviewer's concern); seed variance is already established by the paper's 15-seed
  ResNet runs.

The held-out test set (138 imgs) is NOT used here; this CV is reported alongside (not as
a replacement for) the original held-out-138 numbers.

Resumable: per-fold results are flushed to data/cv_protocol.json after each fold.
"""
import os
import sys, os, json, time
import numpy as np, torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUT = os.path.join(DATA_DIR, 'cv_protocol.json')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/scripts')
from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from model_zoo import ResNetBaseline, set_seed
from canonical_protocol import extract_clip, C_GRID, CLIP_CKPTS, PRIMARY

import open_clip
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FOLDS = 5
N_RESNET_SEEDS = 1
RESNET_EPOCHS = 30
BATCH = 32


class InMemory(Dataset):
    """In-memory image dataset (no per-epoch disk I/O)."""
    def __init__(self, pil_images, labels, transform):
        self.imgs = pil_images
        self.labels = list(labels)
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        img = self.imgs[i].convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {'image': img, 'label': self.labels[i]}


def _loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)


def train_resnet_cv(train_ds, val_ds, num_classes, seed=0, epochs=RESNET_EPOCHS):
    """Paper's ResNet-18 recipe (AdamW 3e-4, cosine, bs32, best-val) on CV subsets."""
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
            opt.zero_grad()
            loss = crit(model(img), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for batch in _loader(val_ds):
                img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
                p = model(img).argmax(1)
                correct += (p == lbl).sum().item(); n += lbl.size(0)
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
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for batch in _loader(ds):
            img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
            p = model(img).argmax(1)
            correct += (p == lbl).sum().item(); n += lbl.size(0)
    return correct / n


def clip_fold(X_fit, y_fit, X_val, y_val, X_te, y_te):
    val_accs = {}
    for C in C_GRID:
        lr = LogisticRegression(C=C, max_iter=5000, random_state=42)
        lr.fit(X_fit, y_fit)
        val_accs[str(C)] = round(float(accuracy_score(y_val, lr.predict(X_val))), 4)
    best_C = max(val_accs, key=val_accs.get)
    lr_fit = LogisticRegression(C=float(best_C), max_iter=5000, random_state=42)
    lr_fit.fit(X_fit, y_fit)
    p = lr_fit.predict(X_te)
    lr_c1 = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    lr_c1.fit(X_fit, y_fit)
    p_c1 = lr_c1.predict(X_te)
    return {
        'best_C': float(best_C), 'val_accs': val_accs,
        'acc': round(float(accuracy_score(y_te, p)), 4),
        'balanced_acc': round(float(balanced_accuracy_score(y_te, p)), 4),
        'acc_C1': round(float(accuracy_score(y_te, p_c1)), 4),
    }


def _save(folds, summary=None):
    obj = {'folds': folds}
    if summary is not None:
        obj['summary'] = summary
    with open(OUT, 'w') as f:
        json.dump(obj, f, indent=2)


def main():
    t0 = time.time()
    print(f'Device: {DEVICE}', flush=True)
    print(f'CLIP primary = {PRIMARY}', flush=True)

    base_eval = TPDDDataset('train', transform=EVAL_TRANSFORM)
    n = len(base_eval)
    n_classes = len(base_eval.class_names)
    y_all = np.array([base_eval[i]['label'] for i in range(n)])
    print(f'Pool: {n} images, {n_classes} classes', flush=True)

    # load all images into memory once (removes per-epoch disk I/O)
    pil_all = [Image.open(str(base_eval.images[i])).convert('RGB') for i in range(n)]
    print(f'Loaded {n} images into memory', flush=True)

    # frozen CLIP features (canonical path) for the whole pool
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained=CLIP_CKPTS[PRIMARY])
    clip_model = clip_model.to(DEVICE).eval()
    X_all, _ = extract_clip(clip_model, preprocess, base_eval, DEVICE)
    del clip_model
    torch.cuda.empty_cache()
    print(f'CLIP features: {X_all.shape}', flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = {}
    done = set()
    if os.path.exists(OUT):
        # BUG FIX: the resume path used to read only the set of completed fold
        # indices and left `folds` empty, so every fold was skipped and the
        # summary then raised KeyError. Load the stored results themselves.
        with open(OUT) as f:
            folds = json.load(f).get('folds', {}) or {}
        done = {str(k) for k in folds}

    for fi, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), y_all)):
        if str(fi) in done:
            print(f'[fold {fi}] already done, skip', flush=True)
            continue
        X_te, y_te = X_all[test_idx], y_all[test_idx]
        X_fit_all, y_fit_all = X_all[train_idx], y_all[train_idx]
        idx_f, idx_v = train_test_split(np.arange(len(train_idx)), test_size=0.2,
                                        random_state=42, stratify=y_fit_all)
        X_fit, y_fit = X_fit_all[idx_f], y_fit_all[idx_f]
        X_val, y_val = X_fit_all[idx_v], y_fit_all[idx_v]

        clip_res = clip_fold(X_fit, y_fit, X_val, y_val, X_te, y_te)

        fit_img = [int(train_idx[i]) for i in idx_f]
        val_img = [int(train_idx[i]) for i in idx_v]
        test_img = [int(i) for i in test_idx]
        train_ds = InMemory([pil_all[i] for i in fit_img], y_fit_all[idx_f], TRAIN_TRANSFORM)
        val_ds = InMemory([pil_all[i] for i in val_img], y_fit_all[idx_v], EVAL_TRANSFORM)
        test_ds = InMemory([pil_all[i] for i in test_img], y_te, EVAL_TRANSFORM)

        rn_accs = []
        for s in range(N_RESNET_SEEDS):
            model = train_resnet_cv(train_ds, val_ds, n_classes, seed=42 + s)
            acc = eval_model(model, test_ds)
            rn_accs.append(round(float(acc), 4))
            del model
            torch.cuda.empty_cache()
            print(f'[fold {fi}] ResNet seed {s}: test acc {acc*100:.2f}%', flush=True)

        folds[str(fi)] = {
            'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)),
            'n_fit': int(len(idx_f)), 'n_val': int(len(idx_v)),
            'clip': clip_res,
            'resnet': {'seeds': rn_accs, 'acc_mean': round(float(np.mean(rn_accs)), 4),
                       'acc_sd': round(float(np.std(rn_accs, ddof=1)) if len(rn_accs) > 1 else 0.0, 4)},
            'gap_clip_minus_resnet': round(float(clip_res['acc'] - np.mean(rn_accs)), 4),
        }
        print(f'[fold {fi}] CLIP best-C {clip_res["acc"]*100:.2f}% (C={clip_res["best_C"]}) | '
              f'ResNet {np.mean(rn_accs)*100:.2f}% | '
              f'gap {folds[str(fi)]["gap_clip_minus_resnet"]*100:+.2f}pp', flush=True)
        _save(folds)

    clip_accs = [folds[str(fi)]['clip']['acc'] for fi in range(N_FOLDS)]
    res_accs = [folds[str(fi)]['resnet']['acc_mean'] for fi in range(N_FOLDS)]
    gaps = [folds[str(fi)]['gap_clip_minus_resnet'] for fi in range(N_FOLDS)]
    summary = {
        'n_folds': N_FOLDS, 'n_resnet_seeds': N_RESNET_SEEDS, 'resnet_epochs': RESNET_EPOCHS,
        'clip': {'acc_mean': round(float(np.mean(clip_accs)), 4),
                 'acc_sd': round(float(np.std(clip_accs, ddof=1)), 4), 'per_fold': clip_accs},
        'resnet': {'acc_mean': round(float(np.mean(res_accs)), 4),
                   'acc_sd': round(float(np.std(res_accs, ddof=1)), 4), 'per_fold': res_accs},
        'gap_clip_minus_resnet': {'mean_pp': round(float(np.mean(gaps) * 100), 2),
                                  'sd_pp': round(float(np.std(gaps, ddof=1) * 100), 2),
                                  'per_fold_pp': [round(g * 100, 2) for g in gaps]},
        'protocol': ('5-fold stratified CV over TPDD train pool (1295); per-fold 80/20 fit/val '
                     'carve-out; CLIP frozen best-C probe and ImageNet-pretrained ResNet-18 '
                     '(AdamW 3e-4, cosine, bs32, 30ep, best-val) both trained on the same '
                     'fit-split; held-out fold used for evaluation only; 1 seed/fold (split '
                     'stability; seed variance covered by the 15-seed ResNet runs)'),
        'elapsed_s': round(time.time() - t0, 1),
    }
    _save(folds, summary=summary)
    print('\n=== CV SUMMARY ===', flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
