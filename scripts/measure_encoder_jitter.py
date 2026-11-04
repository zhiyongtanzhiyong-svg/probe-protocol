"""Quantify encoder jitter in the few-shot linear probe.

WHY (2026-10-05)
----------------
Three implementations all described as "the canonical pixel path" produce
materially different few-shot probe accuracies on identical images, identical
support indices and an identical classifier call:

    data/two_model_e4.json      (batch 64, as written)   K=5  58.41
    two_model_e4.encode, rerun  (batch 64)              K=5  59.76
    data/laion_clip_features_clean.npz (batch 32)       K=5  60.19

The features differ by max |diff| ~ 0.065 on L2-normalized 768-d vectors. That
is far too large for a preprocessing difference and about the size expected from
batch-dependent GEMM kernel selection in a 24-layer fp32 ViT. If that is the
mechanism, then the probe has an implementation-dependent jitter that is NOT
covered by the 15-seed confidence interval, because all 15 seeds share a single
feature matrix.

This script measures the jitter directly: encode the same checkpoint at several
batch sizes and report, per K, the spread of the resulting 15-seed probe mean.
Any comparison whose margin is inside that spread is not interpretable.

Writes data/encoder_jitter.json.
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
from coop_experiment import kshot_subset_idx
from clip_fewshot_probe import tensor_to_pil
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPTS = {
    'dfn_clip': f'{REPO}/../checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin',
    'laion2b': f'{REPO}/../checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin',
}
BATCHES = [16, 32, 64]
KS = [1, 2, 5, 10]
N_SEEDS = 15


def encode(model, preprocess, ds, batch):
    loader = DataLoader(ds, batch, shuffle=False, num_workers=2,
                        worker_init_fn=worker_init_fn)
    feats, labs = [], []
    with torch.no_grad():
        for b in loader:
            x = torch.stack([preprocess(tensor_to_pil(i)) for i in b['image']]).cuda()
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labs.extend(b['label'].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


def probe_15seed(X, y, Xte, yte, K):
    accs = []
    for s in range(N_SEEDS):
        idx = kshot_subset_idx(y, K, s + 500)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=s)
        clf.fit(X[idx], y[idx])
        accs.append(float((clf.predict(Xte) == yte).mean()))
    return float(np.mean(accs)) * 100, float(np.std(accs, ddof=1)) * 100


def main():
    out = {'n_seeds': N_SEEDS, 'batches': BATCHES,
           'note': ('Spread of the 15-seed probe mean across encoder batch sizes. '
                    'Seed noise is inside each entry (sd); the range across '
                    'batches is encoder jitter, which the seed CI does NOT cover '
                    'because all seeds share one feature matrix.')}

    for name, ck in CKPTS.items():
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained=ck)
        model = model.eval().cuda()
        tr = TPDDDataset('train', transform=EVAL_TRANSFORM)
        te = TPDDDataset('test', transform=EVAL_TRANSFORM)

        per_batch, ref = {}, None
        for b in BATCHES:
            Xtr, ytr = encode(model, preprocess, tr, b)
            Xte, yte = encode(model, preprocess, te, b)
            if ref is None:
                ref = (Xtr, Xte)
                feat_drift = 0.0
            else:
                feat_drift = float(max(np.abs(Xtr - ref[0]).max(),
                                       np.abs(Xte - ref[1]).max()))
            row = {'feature_max_abs_diff_vs_batch%d' % BATCHES[0]: round(feat_drift, 6)}
            for K in KS:
                m, sd = probe_15seed(Xtr, ytr, Xte, yte, K)
                row[f'K={K}'] = {'probe_mean_pct': round(m, 2), 'seed_sd_pct': round(sd, 2)}
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0).fit(Xtr, ytr)
            row['full_C1_pct'] = round(float((clf.predict(Xte) == yte).mean()) * 100, 2)
            per_batch[f'batch{b}'] = row
            print(f'  {name} batch={b:3d}  feat_drift={feat_drift:.4f}  '
                  + '  '.join(f'K{k}={row[f"K={k}"]["probe_mean_pct"]:.2f}'
                              for k in KS)
                  + f'  full={row["full_C1_pct"]:.2f}', flush=True)

        summary = {}
        for K in KS + ['full']:
            key = f'K={K}' if K != 'full' else 'full_C1_pct'
            vals = [per_batch[f'batch{b}'][key]['probe_mean_pct']
                    if key != 'full_C1_pct' else per_batch[f'batch{b}'][key]
                    for b in BATCHES]
            summary[f'K={K}'] = {
                'values_pct': vals,
                'range_pp': round(max(vals) - min(vals), 2),
            }
        out[name] = {'per_batch': per_batch, 'encoder_jitter': summary}

    for name in CKPTS:
        print(f'\n{name} encoder jitter (range of the 15-seed probe mean):')
        for K in KS + ['full']:
            s = out[name]['encoder_jitter'][f'K={K}']
            print(f"  K={K:<5s} {s['values_pct']}  range={s['range_pp']:.2f} pp")

    json.dump(out, open(f'{REPO}/data/encoder_jitter.json', 'w'), indent=2)
    print(f'\nSaved {REPO}/data/encoder_jitter.json')


if __name__ == '__main__':
    main()
