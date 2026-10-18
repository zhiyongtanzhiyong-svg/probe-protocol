"""
M0 diagnostic (anti-false-negative for the m0_gate STOP).

The gate's attention head overfit to only 51.5% and lost 6.5pp to the CLS probe.
That is consistent with EITHER (a) patch tokens carrying little class signal, OR
(b) the attention head being under-optimised (near-uniform attention, low lr).

This diagnostic removes the head's training as a confound: it fits plain
LogisticRegression (sklearn, identical to the paper's probe) on patch-DERIVED
features and compares against the CLS probe. If patch-derived features CANNOT
beat CLS on either checkpoint, the C1 hypothesis is genuinely unsupported
(=> STOP is real). If they CAN, the attention head was just under-trained
(=> fix the head, do not stop).

Readouts (all linear, full-data + 10-shot, both ckpts):
  cls            : CLS token                       (paper baseline)
  mean_patch     : mean of 256 patch tokens
  max_patch      : max over patches (per dim)
  cls_mean       : concat[cls ; mean_patch]
  cls_max        : concat[cls ; max_patch]
"""
import os, sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from coop_experiment import kshot_subset_idx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from patch_tokens import CKPT, extract_patch_features
from dataset import TPDDDataset, EVAL_TRANSFORM
import open_clip, torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = os.path.join(REPO, 'data', 'm0_diagnostic.json')


def get_tokens(name):
    cache = os.path.join(REPO, 'data', f'patch_tokens_{name}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return z['cls'], z['patches'], z['labels'], z['test_cls'], z['test_patches'], z['test_labels']
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT[name])
    model = model.to(DEVICE).eval()
    tr = TPDDDataset(split='train', transform=EVAL_TRANSFORM)
    te = TPDDDataset(split='test', transform=EVAL_TRANSFORM)
    cls, patches, labels = extract_patch_features(model, preprocess, tr)
    tcls, tpat, tlab = extract_patch_features(model, preprocess, te)
    del model; torch.cuda.empty_cache()
    np.savez_compressed(cache, cls=cls, patches=patches, labels=labels,
                        test_cls=tcls, test_patches=tpat, test_labels=tlab)
    return cls, patches, labels, tcls, tpat, tlab


def feats(cls, patches):
    cls = cls / (np.linalg.norm(cls, axis=-1, keepdims=True) + 1e-12)
    mn = patches.mean(1); mx = patches.max(1)
    return {
        'cls': cls,
        'mean_patch': mn,
        'max_patch': mx,
        'cls_mean': np.concatenate([cls, mn], 1),
        'cls_max': np.concatenate([cls, mx], 1),
    }


def probe(Xtr, ytr, Xte, yte, seed, C=1.0):
    m = LogisticRegression(C=C, max_iter=3000, random_state=seed)
    m.fit(Xtr, ytr)
    return float((m.predict(Xte) == yte).mean())


def main():
    t0 = time.time()
    R = {'protocol': 'linear LogisticRegression(C=1) on patch-derived vs CLS features; full-data + 10-shot(5 seeds); both ckpts'}
    for name in ['dfn_clip', 'laion2b']:
        cls, patches, labels, tcls, tpat, tlab = get_tokens(name)
        assert len(set(labels.tolist())) == 16 and len(tlab) == 138
        Ftr = feats(cls, patches); Fte = feats(tcls, tpat)
        out = {'n_train': int(len(labels)), 'n_test': int(len(tlab)), 'n_patches': int(patches.shape[1])}

        # full-data (deterministic)
        out['full'] = {k: round(probe(Ftr[k], labels, Fte[k], tlab, seed=42), 4) for k in Ftr}
        # 10-shot, 5 seeds, paired
        full = {}
        for k in Ftr:
            full[k] = [probe(Ftr[k][kshot_subset_idx(labels, 10, s + 500)], labels[kshot_subset_idx(labels, 10, s + 500)],
                            Fte[k], tlab, seed=s) for s in range(5)]
        out['shot10'] = {k: {'mean': round(float(np.mean(full[k])), 4), 'accs': [round(a, 4) for a in full[k]]} for k in Ftr}

        base = out['shot10']['cls']['mean']
        best_key = max(out['shot10'], key=lambda k: out['shot10'][k]['mean'])
        out['best_readout'] = best_key
        out['best_beats_cls_pp'] = round((out['shot10'][best_key]['mean'] - base) * 100, 2)
        R[name] = out
        print(f"\n=== {name} ===  (full / 10-shot mean)")
        for k in ['cls', 'mean_patch', 'max_patch', 'cls_mean', 'cls_max']:
            print(f"  {k:12s}  full {out['full'][k]*100:5.1f}%   10-shot {out['shot10'][k]['mean']*100:5.1f}%"
                  f"   (delta vs cls { (out['shot10'][k]['mean']-base)*100:+.1f}pp)")
        print(f"  -> best readout: {best_key}  ({out['best_beats_cls_pp']:+.1f}pp vs CLS at 10-shot)")

    R['elapsed_sec'] = round(time.time() - t0, 1)
    json.dump(R, open(OUT, 'w'), indent=2)
    print(f"\nSaved {OUT} in {R['elapsed_sec']}s")


if __name__ == '__main__':
    main()
