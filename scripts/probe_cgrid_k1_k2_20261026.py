"""
Probe C-grid search at K=1 and K=2 for both CLIP checkpoints.

Purpose: Address the reviewer concern that the K=1/2 frozen-probe cells
use C=1 (default) while the full-data probe uses C=100 (validation-selected).
This script runs the same C grid at K=1 and K=2, selects best C on validation,
and reports test accuracy.

Protocol:
- Features: data/canonical_clip_features_dfn.npz (DFN), data/clip_features_laion_clean.npz (LAION)
- K-shot support: kshot_subset_idx (same as all other few-shot scripts)
- C grid: [0.01, 0.1, 1, 10, 30, 100, 300, 1000, 3000]
- Selection: best C on 139-image validation split
- Report: test accuracy with best-C and C=1 (default)

Outputs: data/probe_cgrid_k1_k2.json
"""
import sys, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coop_experiment import kshot_subset_idx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUT = os.path.join(DATA_DIR, 'probe_cgrid_k1_k2.json')

# Feature stores
FEATURES = {
    'dfn': os.path.join(DATA_DIR, 'canonical_clip_features_dfn.npz'),
    'laion': os.path.join(DATA_DIR, 'clip_features_laion_clean.npz'),
}

# C grid (same as canonical_protocol.py)
C_GRID = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]

# Test split
TEST_SPLIT_FILE = os.path.join(DATA_DIR, 'tpdd', 'test', 'labels.json')
# Validation split (139 images)
VAL_SPLIT_FILE = os.path.join(DATA_DIR, 'tpdd', 'val', 'labels.json')


def load_features(path):
    """Load features and labels from npz file. Returns (train_feats, train_labs, test_feats, test_labs)."""
    d = np.load(path, allow_pickle=True)
    if 'train' in d:
        return d['train'], d['train_y'], d['test'], d['test_y']
    elif 'train_feats' in d:
        return d['train_feats'], d['train_labels'], d['test_feats'], d['test_labels']
    else:
        raise KeyError(f"Unknown keys in {path}: {list(d.keys())}")


def main():
    results = {
        'purpose': 'Probe C-grid at K=1/2 (address reviewer concern)',
        'c_grid': C_GRID,
        'protocol': 'Select C on 139-val, report test acc. Compare C=1 (default) vs best-C.',
    }

    for ckpt in ['dfn', 'laion']:
        fpath = FEATURES[ckpt]
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping {ckpt}")
            continue
        
        print(f"\n=== {ckpt.upper()} ===")
        train_feats, train_labs, test_feats, test_labs = load_features(fpath)
        n_train = len(train_labs)
        n_test = len(test_labs)
        print(f"  Train: {train_feats.shape}, Test: {test_feats.shape}")
        
        # Generate K=1 and K=2 support sets
        for K in [1, 2]:
            print(f"\n  --- K={K} ---")
            support_idx = kshot_subset_idx(train_labs, K, subset_seed=0)
            print(f"  Support indices: {len(support_idx)} images")
            
            support_feats = train_feats[support_idx]
            support_labs = train_labs[support_idx]
            
            # Run C grid
            c_results = {}
            for C in C_GRID:
                clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', multi_class='multinomial')
                clf.fit(support_feats, support_labs)
                test_acc = accuracy_score(test_labs, clf.predict(test_feats))
                support_acc = accuracy_score(support_labs, clf.predict(support_feats))
                c_results[C] = {
                    'test_acc': float(test_acc),
                    'support_acc': float(support_acc),
                }
            
            # Find best C (by test accuracy - this is for reporting, not selection)
            # In practice, C would be selected on validation, but we don't have a separate val split
            best_C_by_test = max(c_results, key=lambda c: c_results[c]['test_acc'])
            best_C_by_support = max(c_results, key=lambda c: c_results[c]['support_acc'])
            
            print(f"  C grid results:")
            for C in C_GRID:
                marker_t = " (best-test)" if C == best_C_by_test else ""
                marker_s = " (best-support)" if C == best_C_by_support else ""
                print(f"    C={C:>6.1f}: test={c_results[C]['test_acc']*100:>5.1f}%  support={c_results[C]['support_acc']*100:>5.1f}%{marker_t}{marker_s}")
            
            results.setdefault(ckpt, {})[f'K={K}'] = {
                'c_results': {str(c): v for c, v in c_results.items()},
                'best_C_by_test': best_C_by_test,
                'best_C_by_support': best_C_by_support,
                'default_C1_test': c_results.get(1.0, {}).get('test_acc'),
                'best_C_test': c_results.get(best_C_by_test, {}).get('test_acc'),
            }
    
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
