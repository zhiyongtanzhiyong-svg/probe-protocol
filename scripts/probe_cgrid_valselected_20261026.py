#!/usr/bin/env python3
"""
Probe C-grid at K=1/2 with VALIDATION-SELECTED C (fair comparison).

Why: The original C-grid script (probe_cgrid_k1_k2_20261026.py) claimed to select
C on validation but actually used test-set selection (best_C_by_test). This is
statistically invalid. This script correctly:
  1. Extracts features for the 138-image validation split
  2. Trains the probe on K-shot support sets
  3. Selects best C by VALIDATION accuracy
  4. Reports test accuracy with validation-selected C
  5. Also reports C=1 (default) and best-C-by-test (for comparison)

This gives a FAIR comparison: both the probe and the ResNet baseline use
validation-selected hyperparameters.

Protocol:
- Support: K per class from 1295-image training pool (seed=0)
- Validation: 138 images (for C selection)
- Test: 138 images (for final evaluation)
- C grid: [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000, 3000]

Outputs: data/probe_cgrid_k1_k2_valselected.json
"""
import os
import sys, json, os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coop_experiment import kshot_subset_idx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUT = os.path.join(DATA_DIR, 'probe_cgrid_k1_k2_valselected.json')

C_GRID = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]

# Existing feature stores (train + test)
FEATURES = {
    'dfn': os.path.join(DATA_DIR, 'canonical_clip_features_dfn.npz'),
    'laion': os.path.join(DATA_DIR, 'clip_features_laion_clean.npz'),
}

# Validation images
VAL_DIR = os.path.join(DATA_DIR, 'tpdd', 'validation')


def extract_val_features(ckpt_name):
    """Extract CLIP features for the 138 validation images."""
    import open_clip
    from PIL import Image
    
    # Check if we already have val features
    val_feat_file = os.path.join(DATA_DIR, f'val_features_{ckpt_name}.npz')
    if os.path.exists(val_feat_file):
        d = np.load(val_feat_file, allow_pickle=True)
        return d['features'], d['labels']
    
    # Load model
    ckpt_paths = {
        'dfn': 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin',
        'laion': 'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin',
    }
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained=ckpt_paths[ckpt_name]
    )
    model.eval()
    model.cuda()
    
    # Load val labels (list of {image, label} dicts)
    with open(os.path.join(VAL_DIR, 'labels.json')) as f:
        val_labels_list = json.load(f)
    
    # Class mapping: use sorted class names from train labels.json
    with open(os.path.join(DATA_DIR, 'tpdd', 'train', 'labels.json')) as f:
        train_labels_list = json.load(f)
    class_names = sorted(set(entry['label'] for entry in train_labels_list))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    
    # Extract features
    all_feats = []
    all_labs = []
    
    for entry in val_labels_list:
        img_rel = entry['image']  # e.g. 'anthracnose/TV3_4_IMG_9137.jpg'
        cls_name = entry['label']
        img_path = os.path.join(VAL_DIR, img_rel)
        if not os.path.exists(img_path):
            continue
        
        img = Image.open(img_path).convert('RGB')
        tensor = preprocess(img).unsqueeze(0).cuda()
        
        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        
        all_feats.append(feat.cpu().numpy())
        all_labs.append(class_to_idx[cls_name])
    
    all_feats = np.vstack(all_feats)
    all_labs = np.array(all_labs)
    
    # Save
    np.savez(val_feat_file, features=all_feats, labels=all_labs)
    print(f"  Saved val features: {all_feats.shape}")
    
    return all_feats, all_labs


def main():
    results = {
        'purpose': 'Probe C-grid K=1/2 with VALIDATION-SELECTED C (fair comparison)',
        'c_grid': C_GRID,
        'protocol': 'Select C on 138-val, report test acc. Compare C=1 (default) vs val-best-C vs test-best-C.',
        'note': 'This corrects the original C-grid which used test-set selection (invalid).',
    }
    
    for ckpt in ['dfn', 'laion']:
        print(f"\n=== {ckpt.upper()} ===")
        
        # Load existing train + test features
        fpath = FEATURES[ckpt]
        d = np.load(fpath, allow_pickle=True)
        if 'train' in d:
            train_feats, train_labs, test_feats, test_labs = d['train'], d['train_y'], d['test'], d['test_y']
        else:
            train_feats, train_labs, test_feats, test_labs = d['train_feats'], d['train_labels'], d['test_feats'], d['test_labels']
        print(f"  Train: {train_feats.shape}, Test: {test_feats.shape}")
        
        # Extract val features
        print("  Extracting val features...")
        val_feats, val_labs = extract_val_features(ckpt)
        print(f"  Val: {val_feats.shape}")
        
        for K in [1, 2]:
            print(f"\n  --- K={K} ---")
            support_idx = kshot_subset_idx(train_labs, K, subset_seed=0)
            support_feats = train_feats[support_idx]
            support_labs = train_labs[support_idx]
            
            c_results = {}
            for C in C_GRID:
                clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', multi_class='multinomial')
                clf.fit(support_feats, support_labs)
                
                val_acc = accuracy_score(val_labs, clf.predict(val_feats))
                test_acc = accuracy_score(test_labs, clf.predict(test_feats))
                support_acc = accuracy_score(support_labs, clf.predict(support_feats))
                
                c_results[C] = {
                    'val_acc': float(val_acc),
                    'test_acc': float(test_acc),
                    'support_acc': float(support_acc),
                }
            
            # Select best C by VALIDATION accuracy (correct protocol)
            best_C_by_val = max(c_results, key=lambda c: c_results[c]['val_acc'])
            best_C_by_test = max(c_results, key=lambda c: c_results[c]['test_acc'])
            
            print(f"  C grid results (val-selected):")
            for C in C_GRID:
                marker_v = " (BEST-VAL)" if C == best_C_by_val else ""
                marker_t = " (best-test)" if C == best_C_by_test else ""
                print(f"    C={C:>6.1f}: val={c_results[C]['val_acc']*100:>5.1f}%  "
                      f"test={c_results[C]['test_acc']*100:>5.1f}%  "
                      f"support={c_results[C]['support_acc']*100:>5.1f}%{marker_v}{marker_t}")
            
            print(f"\n  Summary:")
            print(f"    C=1 (default):      test={c_results.get(1.0,{}).get('test_acc',0)*100:.1f}%")
            print(f"    Best-C (val-sel):   C={best_C_by_val}, test={c_results[best_C_by_val]['test_acc']*100:.1f}%")
            print(f"    Best-C (test-sel):  C={best_C_by_test}, test={c_results[best_C_by_test]['test_acc']*100:.1f}%")
            print(f"    Val-selected C differs from test-selected: {best_C_by_val != best_C_by_test}")
            
            # Compare with matched-budget ResNet
            if ckpt == 'dfn':
                resnet_k1, resnet_k2 = 40.4, 52.5
            else:
                resnet_k1, resnet_k2 = 40.4, 52.5
            
            resnet = resnet_k1 if K == 1 else resnet_k2
            val_best_test = c_results[best_C_by_val]['test_acc'] * 100
            gap = val_best_test - resnet
            print(f"\n  vs matched-budget ResNet ({resnet}%): {gap:+.1f}pp")
            
            results.setdefault(ckpt, {})[f'K={K}'] = {
                'c_results': {str(c): v for c, v in c_results.items()},
                'best_C_by_val': best_C_by_val,
                'best_C_by_test': best_C_by_test,
                'val_best_test_acc': c_results[best_C_by_val]['test_acc'],
                'val_best_val_acc': c_results[best_C_by_val]['val_acc'],
                'default_C1_test': c_results.get(1.0, {}).get('test_acc'),
                'resnet_matched': resnet,
                'gap_vs_resnet_pp': gap,
            }
    
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
