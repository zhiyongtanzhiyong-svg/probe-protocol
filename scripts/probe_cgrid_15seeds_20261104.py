#!/usr/bin/env python3
"""
M3 (Round-18 external review follow-up): probe C-grid at K=1/2/5/10 with
VALIDATION-SELECTED C, now over 15 support-set seeds (was single-seed seed-0).

Why
---
The manuscript's K=1/2 statement "the probe C-grid at K=1/2 reverses for LAION at
validation-selected C" rests on ONE support-seed draw (scripts/probe_cgrid_valselected_20261026.py,
subset_seed=0). The ResNet side of the same contrast has 15 seeds. This run makes the
probe side equally powered: for each of the 15 seeds, draw the K-shot support set with
the SAME draw function the ResNet matched runs use (kshot_subset_idx(labels, K, seed+500)),
fit the C-grid, select C on the 138-image validation split only, and record the test
accuracy at the selected C. Per-seed values are therefore directly seed-paired with
data/p3b_resnet_kshot_matched{_k1,_k2,_k10}.json.

Protocol (identical to the single-seed run, extended to 15 seeds)
-----------------------------------------------------------------
* Support: K per class from the 1295-image train pool, draw = kshot_subset_idx(y, K, s+500)
  (same function, same offset as the ResNet matched runs -> index-identical support sets).
* C grid: [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000, 3000]; selected by best VALIDATION
  accuracy (first maximum in grid order on ties -- deterministic).
* Test accuracy recorded only at the val-selected C; never used for selection.
* Features: canonical clean-transform stores (same as every probe number in the paper).

Outputs: data/probe_cgrid_15seeds_20261104.json
"""
import os
import sys
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coop_experiment import kshot_subset_idx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUT = os.path.join(DATA_DIR, 'probe_cgrid_15seeds_20261104.json')

C_GRID = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
SEEDS = list(range(15))            # 0..14, identical to the ResNet stage-2 seed list
SEED_OFFSET = 500                  # identical offset as resnet_kshot_matched_budget.py
K_LIST = [1, 2, 5, 10]

FEATURES = {
    'dfn': os.path.join(DATA_DIR, 'canonical_clip_features_dfn.npz'),
    'laion': os.path.join(DATA_DIR, 'clip_features_laion_clean.npz'),
}
VAL_FEATS = {
    'dfn': os.path.join(DATA_DIR, 'val_features_dfn.npz'),
    'laion': os.path.join(DATA_DIR, 'val_features_laion.npz'),
}
RESNET_FILES = {
    1: 'p3b_resnet_kshot_matched_k1.json',
    2: 'p3b_resnet_kshot_matched_k2.json',
    5: 'p3b_resnet_kshot_matched.json',
    10: 'p3b_resnet_kshot_matched_k10.json',
}


def load_store(path):
    d = np.load(path, allow_pickle=True)
    if 'train' in d:
        return d['train'], d['train_y'], d['test'], d['test_y']
    return d['train_feats'], d['train_labels'], d['test_feats'], d['test_labels']


def paired_t(diffs):
    n = len(diffs)
    m = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0 or n < 2:
        return None
    t = m / (sd / np.sqrt(n))
    # two-sided p via scipy if available
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        p = None
    return {"t": round(t, 4), "df": n - 1, "p": p, "mean_diff_pp": round(m * 100, 3)}


def main():
    t0 = time.time()
    results = {
        'purpose': ('M3 follow-up: validation-selected-C probe, 15 support seeds, K=1/2/5/10; '
                    'per-seed seed-paired with the ResNet matched-budget runs'),
        'c_grid': C_GRID,
        'seeds': SEEDS,
        'seed_offset': SEED_OFFSET,
        'support_draw': 'coop_experiment.kshot_subset_idx(y, K, seed+500); identical to the '
                        'ResNet matched runs (resnet_kshot_matched_budget.py)',
        'selection': 'best validation accuracy on the 138-image validation split; first grid '
                     'maximum on ties; test accuracy never used for selection',
        'features': {k: os.path.relpath(v, ROOT) for k, v in FEATURES.items()},
    }

    for ckpt in ['dfn', 'laion']:
        print(f"\n=== {ckpt.upper()} ===", flush=True)
        train_feats, train_labs, test_feats, test_labs = load_store(FEATURES[ckpt])
        vd = np.load(VAL_FEATS[ckpt], allow_pickle=True)
        val_feats, val_labs = vd['features'], vd['labels']
        print(f"  train={train_feats.shape} val={val_feats.shape} test={test_feats.shape}", flush=True)

        results[ckpt] = {}
        for K in K_LIST:
            per_seed = []
            for s in SEEDS:
                idx = kshot_subset_idx(train_labs, K, s + SEED_OFFSET)
                sup_feats = train_feats[idx]
                sup_labs = train_labs[idx]
                best_c, best_val = None, -1.0
                c1_test = None
                best_test_c = None
                best_test_val = -1.0
                for C in C_GRID:
                    clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs',
                                             multi_class='multinomial')
                    clf.fit(sup_feats, sup_labs)
                    va = float(accuracy_score(val_labs, clf.predict(val_feats)))
                    ta = float(accuracy_score(test_labs, clf.predict(test_feats)))
                    if C == 1.0:
                        c1_test = ta
                    if ta > best_test_val:
                        best_test_val, best_test_c = ta, C
                    if va > best_val:
                        best_val, best_c = va, C
                clf = LogisticRegression(C=best_c, max_iter=1000, solver='lbfgs',
                                         multi_class='multinomial').fit(sup_feats, sup_labs)
                sel_test = float(accuracy_score(test_labs, clf.predict(test_feats)))
                per_seed.append({"seed": s, "selected_C": best_c,
                                 "selected_val_acc": round(best_val, 4),
                                 "test_acc_at_selected_C": round(sel_test, 4),
                                 "test_acc_at_C1": c1_test,
                                 "best_test_C": best_test_c,
                                 "best_test_acc": round(best_test_val, 4)})
                print(f"  K={K:<2d} seed={s:>2d}: C_sel={best_c:<6g} "
                      f"val={best_val*100:5.1f}%  test(sel)={sel_test*100:5.1f}%  "
                      f"test(C1)={c1_test*100:5.1f}%  bestTestC={best_test_c:<6g} "
                      f"({best_test_val*100:5.1f}%)", flush=True)

            sel_accs = [r['test_acc_at_selected_C'] for r in per_seed]
            c1_accs = [r['test_acc_at_C1'] for r in per_seed]
            cdist = {str(c): int(sum(1 for r in per_seed if r['selected_C'] == c)) for c in C_GRID}

            # seed-paired comparison vs the 15-seed ResNet matched run (same seeds, same draws)
            rj = json.load(open(os.path.join(DATA_DIR, RESNET_FILES[K])))
            r_accs = rj['stage2_final']['accs']
            assert len(r_accs) == len(sel_accs), "seed counts must match for pairing"
            diffs = [a - b for a, b in zip(sel_accs, r_accs)]
            res = {
                'K': K,
                'per_seed': per_seed,
                'selected_C_distribution': cdist,
                'val_selected_C': {
                    'mean': round(float(np.mean(sel_accs)), 6),
                    'std': round(float(np.std(sel_accs, ddof=1)), 6),
                    'n': len(sel_accs),
                },
                'default_C1': {
                    'mean': round(float(np.mean(c1_accs)), 6),
                    'std': round(float(np.std(c1_accs, ddof=1)), 6),
                    'n': len(c1_accs),
                },
                'resnet_matched_15seed': {
                    'mean': rj['stage2_final']['mean'],
                    'std': rj['stage2_final']['std'],
                    'source': RESNET_FILES[K],
                },
                'gap_vs_resnet_pp': {
                    'val_selected_C': round((float(np.mean(sel_accs)) - rj['stage2_final']['mean']) * 100, 3),
                    'C1': round((float(np.mean(c1_accs)) - rj['stage2_final']['mean']) * 100, 3),
                },
                'seed_paired_probe_minus_resnet': paired_t(diffs),
            }
            results[ckpt][f'K={K}'] = res
            print(f"\n  SUMMARY K={K}: val-selC={res['gap_vs_resnet_pp']['val_selected_C']:+.1f}pp vs ResNet"
                  f" | C1={res['gap_vs_resnet_pp']['C1']:+.1f}pp"
                  f" | paired t={res['seed_paired_probe_minus_resnet']}", flush=True)

    results['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(results, open(OUT, 'w'), indent=2, ensure_ascii=False)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == '__main__':
    main()
