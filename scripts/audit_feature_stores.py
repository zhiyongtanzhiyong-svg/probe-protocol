"""Build a canonical clean feature cache for the LAION-2B checkpoint, and audit
every cached feature store in data/ against the current image corpus.

WHY (2026-10-05)
----------------
Timeline established by mtime:
  data/clip_features_clean.npz      2026-08-29   <- 1 month before the rebuild
  data/clip_features_laion_clean.npz 2026-09-27
  data/clip_aug_views_dfn.npz       2026-09-27 14:32   <- CoOp's 200 views
  data/clip_aug_views_laion.npz     2026-09-27 21:51   <- CoOp's 200 views
  data/canonical_clip_features_dfn.npz 2026-09-28 07:22 <- matches current pixels

The original 11 MB low-resolution TPDD image set was destroyed on 2026-09-28 and
re-derived from the public source. Artifacts written before that re-derivation
were computed on images that no longer exist on disk. This script measures, per
cached store, how far it sits from the current corpus, so the manuscript can
state which results are reproducible from the repository as it now stands.

Writes data/laion_clip_features_clean.npz and data/feature_store_audit.json.
Exit 0 if the LAION cache is fresh; the audit is written either way.
"""
import os
import sys, os, json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

REPO = ROOT
sys.path.insert(0, f'{REPO}/scripts')

import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM, worker_init_fn
from clip_fewshot_probe import tensor_to_pil
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LAION_CKPT = f'{REPO}/../checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin'
OUT = f'{REPO}/data/laion_clip_features_clean.npz'
ATOL = 1e-5


def encode(model, preprocess, split):
    ds = TPDDDataset(split, transform=EVAL_TRANSFORM)
    loader = DataLoader(ds, 32, shuffle=False, num_workers=2,
                        worker_init_fn=worker_init_fn)
    feats, labs = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image']
            x = torch.stack([preprocess(tensor_to_pil(i)) for i in imgs]).cuda()
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


def cos_stats(a, b):
    c = (a * b).sum(1)
    return {'min': round(float(c.min()), 4), 'max': round(float(c.max()), 4),
            'mean': round(float(c.mean()), 4)}


def main():
    audit = {}

    # ---- canonical LAION ------------------------------------------------
    if os.path.exists(OUT):
        z = np.load(OUT)
        Xtr, ytr, Xte, yte = z['Xtr'], z['ytr'], z['Xte'], z['yte']
        print(f'reusing {OUT}', flush=True)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained=LAION_CKPT)
        model = model.eval().cuda()
        print('encoding LAION train (EVAL)...', flush=True)
        Xtr, ytr = encode(model, preprocess, 'train')
        print('encoding LAION test (EVAL)...', flush=True)
        Xte, yte = encode(model, preprocess, 'test')
        np.savez(OUT, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
        print(f'Saved {OUT}', flush=True)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0).fit(Xtr, ytr)
    la_fresh_full = float((clf.predict(Xte) == yte).mean()) * 100
    print(f'LAION full-data C=1 on current corpus: {la_fresh_full:.2f}%', flush=True)
    audit['laion_canonical'] = {
        'file': os.path.basename(OUT), 'fresh': True,
        'full_data_C1_pct': round(la_fresh_full, 2),
    }
    cj = json.load(open(f'{REPO}/data/canonical_results.json'))
    claim = cj['clip_by_checkpoint']['laion2b']['linear_C1']['accuracy'] * 100
    audit['laion_canonical']['canonical_results_json_pct'] = round(claim, 2)
    audit['laion_canonical']['agrees_with_paper'] = abs(la_fresh_full - claim) <= 0.06

    # ---- audit the legacy LAION cache ----------------------------------
    leg = f'{REPO}/data/clip_features_laion_clean.npz'
    if os.path.exists(leg):
        L = np.load(leg)
        d_tr = float(np.abs(Xtr - L['train_feats']).max())
        d_te = float(np.abs(Xte - L['test_feats']).max())
        lab = bool(np.array_equal(ytr, L['train_labels']))
        clf2 = LogisticRegression(max_iter=2000, C=1.0, random_state=0).fit(
            L['train_feats'], L['train_labels'])
        leg_full = float((clf2.predict(L['test_feats']) == L['test_labels']).mean()) * 100
        audit['clip_features_laion_clean.npz'] = {
            'fresh': bool(max(d_tr, d_te) <= ATOL and lab),
            'max_abs_diff_train': round(d_tr, 6), 'max_abs_diff_test': round(d_te, 6),
            'labels_match': lab,
            'per_sample_cosine_train': cos_stats(Xtr, L['train_feats']),
            'full_data_C1_pct': round(leg_full, 2),
            'verdict': ('MATCHES current corpus' if max(d_tr, d_te) <= ATOL
                        else 'DRIFTED from current corpus'),
        }
        print(f'legacy LAION cache: max|diff|={max(d_tr,d_te):.4f} '
              f'full C=1={leg_full:.2f}% -> {audit["clip_features_laion_clean.npz"]["verdict"]}',
              flush=True)

    # ---- audit the legacy DFN cache ------------------------------------
    legd = f'{REPO}/data/clip_features_clean.npz'
    cand = f'{REPO}/data/canonical_clip_features_dfn.npz'
    if os.path.exists(legd) and os.path.exists(cand):
        D = np.load(legd); C = np.load(cand)
        d = max(float(np.abs(C['train'] - D['train_feats']).max()),
                float(np.abs(C['test'] - D['test_feats']).max()))
        audit['clip_features_clean.npz'] = {
            'fresh': bool(d <= ATOL), 'max_abs_diff': round(d, 6),
            'per_sample_cosine_train': cos_stats(C['train'], D['train_feats']),
            'verdict': ('MATCHES current corpus' if d <= ATOL
                        else 'DRIFTED from current corpus'),
        }
        print(f'legacy DFN cache: max|diff|={d:.4f} -> '
              f'{audit["clip_features_clean.npz"]["verdict"]}', flush=True)

    # ---- what the augmented-view stores are pinned to -------------------
    # These cannot be compared pixel-wise (stochastic views), but their mtime
    # predates the rebuild, so record that as the operative fact.
    import datetime
    for name in ('clip_aug_views_dfn.npz', 'clip_aug_views_laion.npz'):
        p = f'{REPO}/data/{name}'
        if os.path.exists(p):
            z = np.load(p, mmap_mode='r')
            mt = datetime.datetime.fromtimestamp(os.path.getmtime(p))
            audit[name] = {
                'mtime': mt.strftime('%Y-%m-%d %H:%M'),
                'predates_2026_09_28_rebuild': mt < datetime.datetime(2026, 9, 28),
                'shape': list(z['views'].shape),
                'note': ('stochastic augmented views; not pixel-comparable, but '
                         'built from the pre-rebuild image set'),
            }
            print(f'{name}: mtime {audit[name]["mtime"]} '
                  f'predates_rebuild={audit[name]["predates_2026_09_28_rebuild"]}',
                  flush=True)

    json.dump(audit, open(f'{REPO}/data/feature_store_audit.json', 'w'), indent=2)
    print(f'\nSaved {REPO}/data/feature_store_audit.json')
    stale = [k for k, v in audit.items()
             if isinstance(v, dict) and v.get('fresh') is False]
    print(f'STALE stores ({len(stale)}): {stale}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
