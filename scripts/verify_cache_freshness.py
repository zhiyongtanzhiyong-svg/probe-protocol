"""
Gate check: is data/clip_features_clean.npz still a faithful encoding of the
CURRENT data/tpdd corpus?

Motivation: the editor's cross-verification (A14) already flagged
data/p2_*.npy as a stale run. Before switching clip_fewshot_probe.py onto the
cached clean features, the cache must be shown to reproduce from the images
that are on disk now. This script re-encodes train+test with the exact code
path clip_fewshot_probe.py uses (EVAL transform -> inverse ImageNet norm ->
open_clip preprocess) and compares to the cache.

Writes data/cache_freshness.json. Exit 0 iff all splits match to atol=1e-5.
"""
import os
import sys, json
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

CKPT = f'{REPO}/../checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
CACHE = f'{REPO}/data/clip_features_clean.npz'
ATOL = 1e-5


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
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained=CKPT)
    model = model.eval().cuda()

    z = np.load(CACHE)
    report, ok_all = {}, True

    for split, key in [('train', 'train'), ('test', 'test')]:
        feats, labs = encode(model, preprocess, split)
        c_feats, c_labs = z[f'{key}_feats'], z[f'{key}_labels']
        labels_match = bool(np.array_equal(labs, c_labs))
        dmax = float(np.abs(feats - c_feats).max()) if feats.shape == c_feats.shape else None
        feats_match = bool(dmax is not None and dmax <= ATOL)
        ok = labels_match and feats_match
        ok_all &= ok
        report[split] = {
            'n_images': int(len(labs)),
            'shape_fresh': list(feats.shape), 'shape_cache': list(c_feats.shape),
            'labels_match': labels_match,
            'max_abs_feat_diff': dmax, 'atol': ATOL, 'feats_match': feats_match,
            'verdict': 'MATCH' if ok else 'DRIFT',
        }
        print(f'  {split:5s} n={len(labs):5d} labels_match={labels_match} '
              f'max|diff|={dmax} -> {report[split]["verdict"]}', flush=True)

    # Consequential check: does the canonical full-data C=1 probe still give 76.81?
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
    clf.fit(z['train_feats'], z['train_labels'])
    full_acc = float((clf.predict(z['test_feats']) == z['test_labels']).mean())
    report['full_data_C1_recomputed'] = {
        'acc': full_acc, 'acc_pct': round(full_acc * 100, 2),
        'canonical_results_json_pct': json.load(
            open(f'{REPO}/data/canonical_results.json'))['dfn_clip']['linear_C1']['acc_pct']
        if 'dfn_clip' in json.load(open(f'{REPO}/data/canonical_results.json')) else None,
    }
    print(f'  full-data C=1 on cache: {full_acc*100:.2f}%', flush=True)

    report['all_splits_match'] = bool(ok_all)
    json.dump(report, open(f'{REPO}/data/cache_freshness.json', 'w'), indent=2)
    print(f'\nVERDICT: {"CACHE IS FRESH" if ok_all else "CACHE HAS DRIFT -- DO NOT SWITCH"}')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
