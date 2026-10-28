"""
Cross-crop K=10 ResNet-18 matched-budget (9-config grid search).

Purpose: Address the reviewer concern that the cross-crop K=10 ResNet
baselines use the legacy single-configuration protocol, not the matched-budget
protocol used for TPDD.

Protocol (same as resnet_kshot_matched_budget.py for TPDD):
- Stage 1: 9 configs (LR × EPOCHS), 3 seeds each, select best on validation
- Stage 2: Winner re-run with 15 seeds
- Dataset: PlantVillage tomato (10 classes) and corn (4 classes)

Outputs: data/resnet_k10_matched_tomato.json, data/resnet_k10_matched_corn.json
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from dataset import TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from model_zoo import ResNetBaseline, set_seed
from coop_experiment import kshot_subset_idx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED_OFFSET = 500
VAL_EVERY = 10
BATCH = 32

# 9 configurations (same grid as TPDD matched-budget)
LR_GRID = [3e-4, 1e-3, 3e-3]
EPOCH_GRID = [30, 100, 300]
CONFIGS = [(lr, ep) for lr in LR_GRID for ep in EPOCH_GRID]

# Dataset configs
DATASETS = {
    'tomato': {
        'data_dir': os.path.join(REPO, 'data', 'plantvillage_tomato'),
        'n_classes': 10,
        'out': os.path.join(REPO, 'data', 'resnet_k10_matched_tomato.json'),
    },
    'corn': {
        'data_dir': os.path.join(REPO, 'data', 'plantvillage_corn'),
        'n_classes': 4,
        'out': os.path.join(REPO, 'data', 'resnet_k10_matched_corn.json'),
    },
}

K = 10  # K-shot


def make_subset(labels, K, seed):
    """Create K-shot subset indices."""
    return kshot_subset_idx(labels, K, subset_seed=seed)


def train_resnet(model, train_loader, val_loader, epochs, lr, device, val_every=10):
    """Train ResNet-18 and return best validation accuracy."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val = 0.0
    best_epoch = 0
    best_state = None
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labs in val_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    out = model(imgs)
                    correct += (out.argmax(1) == labs).sum().item()
                    total += len(labs)
            val_acc = correct / total
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    return best_val, best_epoch, best_state


def evaluate(model, loader, device):
    """Evaluate model on a loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs)
            correct += (out.argmax(1) == labs).sum().item()
            total += len(labs)
    return correct / total


def run_crop(crop_name, cfg):
    """Run the matched-budget K=1 experiment for one crop."""
    data_dir = cfg['data_dir']
    n_classes = cfg['n_classes']
    out_path = cfg['out']
    
    print(f"\n{'='*60}")
    print(f"CROP: {crop_name.upper()} ({n_classes} classes)")
    print(f"{'='*60}")
    
    # Load dataset
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    
    # Create full dataset with training transform
    full_transform = TRAIN_TRANSFORM
    full_ds = ImageFolder(data_dir, transform=full_transform)
    
    # Create label array
    labels = np.array([s[1] for s in full_ds.samples])
    n_total = len(full_ds)
    print(f"Total images: {n_total}")
    
    # Create train/val/test split (same as existing protocol)
    # Use the same split as the existing cross-crop protocol
    # For simplicity, use a fixed seed for the split
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15, random_state=42, stratify=labels[train_idx]
    )
    
    # Create datasets
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)
    test_ds = torch.utils.data.Subset(full_ds, test_idx)
    
    # Create K-shot subset of training set
    train_labels = labels[train_idx]
    
    results = {
        'crop': crop_name,
        'K': K,
        'n_classes': n_classes,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'protocol': '9-config grid search (LR x EPOCHS), select on val, 15-seed final',
    }
    
    # Stage 1: Grid search
    print("\n--- Stage 1: Grid Search (9 configs x 3 seeds) ---")
    grid_results = []
    
    for lr, epochs in CONFIGS:
        config_accs = []
        for seed in range(3):
            full_seed = seed + SEED_OFFSET
            set_seed(full_seed)
            
            # Create K-shot subset
            subset_idx = make_subset(train_labels, K, full_seed)
            kshot_idx = train_idx[subset_idx]
            kshot_ds = torch.utils.data.Subset(full_ds, kshot_idx)
            
            train_loader = DataLoader(kshot_ds, batch_size=BATCH, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
            val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
            
            model = ResNetBaseline(num_classes=n_classes)
            model = model.to(DEVICE)
            
            t0 = time.time()
            best_val, best_epoch, best_state = train_resnet(
                model, train_loader, val_loader, epochs, lr, DEVICE, VAL_EVERY
            )
            elapsed = time.time() - t0
            
            # Evaluate on test
            if best_state is not None:
                model.load_state_dict(best_state)
            test_acc = evaluate(model, DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn), DEVICE)
            
            accs_entry = {
                'seed': full_seed,
                'lr': lr,
                'epochs': epochs,
                'best_val': best_val,
                'best_epoch': best_epoch,
                'test_acc': test_acc,
                'elapsed_s': elapsed,
            }
            config_accs.append(accs_entry)
            print(f"  lr={lr:.0e} ep={epochs:>3d} seed={full_seed}: val={best_val*100:.1f}% test={test_acc*100:.1f}% ({elapsed:.0f}s)", flush=True)
        
        mean_val = np.mean([a['best_val'] for a in config_accs])
        mean_test = np.mean([a['test_acc'] for a in config_accs])
        grid_results.append({
            'lr': lr,
            'epochs': epochs,
            'mean_best_val': mean_val,
            'mean_test': mean_test,
            'runs': config_accs,
        })
        print(f"  => lr={lr:.0e} ep={epochs:>3d}: mean_val={mean_val*100:.1f}% mean_test={mean_test*100:.1f}%", flush=True)
    
    # Select best config by mean validation accuracy
    best_config = max(grid_results, key=lambda x: x['mean_best_val'])
    print(f"\n  Best config: lr={best_config['lr']:.0e} epochs={best_config['epochs']} (mean_val={best_config['mean_best_val']*100:.1f}%)")
    
    results['stage1_grid'] = grid_results
    results['selected_config'] = {'lr': best_config['lr'], 'epochs': best_config['epochs']}
    
    # Stage 2: 15-seed run with best config
    print("\n--- Stage 2: 15-seed Final Run ---")
    final_accs = []
    for seed in range(15):
        full_seed = seed + SEED_OFFSET
        set_seed(full_seed)
        
        subset_idx = make_subset(train_labels, K, full_seed)
        kshot_idx = train_idx[subset_idx]
        kshot_ds = torch.utils.data.Subset(full_ds, kshot_idx)
        
        train_loader = DataLoader(kshot_ds, batch_size=BATCH, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
        
        model = ResNetBaseline(num_classes=n_classes)
        model = model.to(DEVICE)
        
        best_val, best_epoch, best_state = train_resnet(
            model, train_loader, val_loader, best_config['epochs'], best_config['lr'], DEVICE, VAL_EVERY
        )
        
        if best_state is not None:
            model.load_state_dict(best_state)
        test_acc = evaluate(model, DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn), DEVICE)
        
        final_accs.append({
            'seed': full_seed,
            'test_acc': test_acc,
            'best_val': best_val,
        })
        print(f"  seed={full_seed}: test={test_acc*100:.1f}%", flush=True)
    
    mean_test = np.mean([a['test_acc'] for a in final_accs])
    std_test = np.std([a['test_acc'] for a in final_accs], ddof=1)
    
    results['stage2_final'] = {
        'n_seeds': 15,
        'per_seed': final_accs,
        'mean_test': mean_test,
        'std_test': std_test,
    }
    
    # Save
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Final: {mean_test*100:.1f}% +/- {std_test*100:.1f}% (n=15)")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', type=str, default='both', choices=['tomato', 'corn', 'both'])
    args = parser.parse_args()
    
    if args.crop in ['tomato', 'both']:
        run_crop('tomato', DATASETS['tomato'])
    if args.crop in ['corn', 'both']:
        run_crop('corn', DATASETS['corn'])
    
    print("\nDone.")
