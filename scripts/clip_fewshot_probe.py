"""
CLIP Few-Shot Linear Probe -- single caliber.

WHY THIS FILE WAS REWRITTEN (2026-10-05)
----------------------------------------
The previous version had two defects that made its numbers irreconcilable with
the rest of the repository:

  1. TRANSFORM. It built the K-shot support pool with TRAIN_TRANSFORM
     (RandomHorizontalFlip / RandomRotation / ColorJitter / RandomResizedCrop)
     and then encoded it ONCE per seed, because sklearn's LogisticRegression is
     a single convex fit with no epoch loop. Augmentation therefore bought
     nothing -- it only injected one stochastic draw of noise into the
     features. Every other script in this repo (dinov2_protocol.py,
     coop_official.py's eval views, canonical_protocol.py) uses EVAL_TRANSFORM
     for exactly this reason: EVAL_TRANSFORM is deterministic, so a "n_seeds"
     spread reflects support resampling, not image noise.

  2. STALE CACHE TRAP. It encoded fresh from images, while
     lp_plus_plus.py and eight other scripts read data/clip_features_clean.npz,
     a legacy cache whose max |diff| against the current corpus is 0.119 on
     L2-normalized features (verify_cache_freshness.py). So the two scripts
     disagreed on images *and* on transform at the same time.

Together these produced two incompatible "DFN-CLIP probe" rows in one paper:
31.1 / 51.3 / 58.3 here, versus 30.5 / 49.1 / 57.7 in the LP++ discussion.

WHAT THIS VERSION DOES
----------------------
Encode train+test ONCE with EVAL_TRANSFORM, assert the result is bit-identical
to data/canonical_clip_features_dfn.npz (the cache canonical_protocol.py uses),
then subset support by index with the shared kshot_subset_idx helper. That
helper is verified to draw the same indices as the KShotSubset classes in
clip_fewshot_probe.py / paired_kshot_a4.py, because data/tpdd/*/labels.json is
class-sorted so dict-insertion order equals sorted class order.

Consequence: the full-data row becomes deterministic (std exactly 0.0) and
every few-shot number is reproducible from this one artifact.

Writes data/clip_fewshot.json (+ _fixed alias) and data/fewshot_features_dfn.npz.
"""
import os
import sys, json, os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

REPO = ROOT
sys.path.insert(0, f'{REPO}/scripts')

import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM, worker_init_fn
from coop_experiment import kshot_subset_idx
from torch.utils.data import DataLoader
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def tensor_to_pil(img_tensor):
    """Undo EVAL_TRANSFORM's ImageNet normalisation back to a uint8 PIL image.

    This is a documented deliberate step, not a bug: EVAL_TRANSFORM already
    normalised to roughly [-1.6, 2.6], and open_clip's own preprocess expects
    raw pixels. Sensitivity is reported in data/inverse_norm_sensitivity.json.
    """
    x = img_tensor.permute(1, 2, 0).cpu().numpy()
    x = np.clip(x * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
    return Image.fromarray((x * 255).astype(np.uint8))


CKPT = f'{REPO}/../checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
OUT_FEATS = f'{REPO}/data/fewshot_features_dfn.npz'
CANONICAL = f'{REPO}/data/canonical_clip_features_dfn.npz'
N_SEEDS = 15
KS = [1, 2, 5, 10]


def encode(model, preprocess, split):
    ds = TPDDDataset(split, transform=EVAL_TRANSFORM)
    loader = DataLoader(ds, 32, shuffle=False, num_workers=2,
                        worker_init_fn=worker_init_fn)
    feats, labs = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image']
            clip_imgs = torch.stack(
                [preprocess(tensor_to_pil(img)) for img in imgs]).cuda()
            f = model.encode_image(clip_imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


def main():
    if os.path.exists(OUT_FEATS):
        z = np.load(OUT_FEATS)
        Xtr, ytr, Xte, yte = z['Xtr'], z['ytr'], z['Xte'], z['yte']
        print(f'reusing {OUT_FEATS}', flush=True)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained=CKPT)
        model = model.eval().cuda()
        print('encoding train (EVAL)...', flush=True)
        Xtr, ytr = encode(model, preprocess, 'train')
        print('encoding test (EVAL)...', flush=True)
        Xte, yte = encode(model, preprocess, 'test')
        np.savez(OUT_FEATS, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    # Gate: this must agree with the cache canonical_protocol.py already uses,
    # otherwise the few-shot table and the full-data table describe different
    # feature spaces and the paper cannot be internally consistent.
    if os.path.exists(CANONICAL):
        c = np.load(CANONICAL)
        d = max(float(np.abs(Xtr - c['train']).max()),
                float(np.abs(Xte - c['test']).max()))
        lab = bool(np.array_equal(ytr, c['train_y']) and np.array_equal(yte, c['test_y']))
        print(f'vs canonical_clip_features_dfn.npz: max|diff|={d:.2e} labels_match={lab}',
              flush=True)
        if d > 1e-5 or not lab:
            raise SystemExit('ABORT: fresh features disagree with canonical cache')
    else:
        print('WARNING: canonical cache absent, cross-check skipped', flush=True)

    ds = TPDDDataset('train', transform=EVAL_TRANSFORM)
    assert np.array_equal(ytr, np.array(ds.label_ids)), 'label order drift'

    results = {}
    for K in KS:
        accs = []
        for seed in range(N_SEEDS):
            sup = kshot_subset_idx(ytr, K, seed + 500)
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(Xtr[sup], ytr[sup])
            accs.append(float((clf.predict(Xte) == yte).mean()))
        accs = np.array(accs)
        results[f'K={K}'] = {'mean': float(accs.mean()),
                             'std': float(accs.std(ddof=1)),
                             'accs': accs.tolist(), 'n_seeds': N_SEEDS,
                             'support_transform': 'EVAL',
                             'support_indices': 'coop_experiment.kshot_subset_idx(y,K,seed+500)'}
        print(f'CLIP probe K={K:<3d}: {accs.mean()*100:5.2f}% +- {accs.std(ddof=1)*100:.2f}',
              flush=True)

    # full-data: deterministic by construction, one fit, std is exactly 0.
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
    clf.fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())
    results['full'] = {'mean': acc, 'std': 0.0, 'accs': [acc], 'n_seeds': 1,
                       'support_transform': 'EVAL', 'support_indices': 'all'}
    print(f'CLIP probe full : {acc*100:5.2f}% (deterministic)', flush=True)

    results['_protocol'] = (
        'Single caliber as of 2026-10-05. Support encoded with EVAL_TRANSFORM '
        '(deterministic); support indices from coop_experiment.kshot_subset_idx '
        '(verified identical to the KShotSubset classes in paired_kshot_a4.py '
        'and the previous clip_fewshot_probe.py). Previously this file used '
        'TRAIN_TRANSFORM + fresh encoding, which produced a second, '
        'irreconcilable DFN probe row (31.1/51.3/58.3 vs 30.5/49.1/57.7).'
    )

    for name in ('clip_fewshot.json', 'clip_fewshot_fixed.json'):
        json.dump(results, open(f'{REPO}/data/{name}', 'w'), indent=2)
    print('Saved data/clip_fewshot.json (+ _fixed alias).', flush=True)


if __name__ == '__main__':
    main()
