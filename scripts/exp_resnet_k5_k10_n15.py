"""
ResNet-18 K=5,10 Few-Shot Experiment (n=15 seeds).
Addresses reviewer concern: CLIP probe n=15 vs ResNet K=5,10 n=5 misalignment.
"""
import os
import sys, json, torch, numpy as np, time, os, argparse
sys.path.insert(0, ROOT + '/scripts')

from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from torch.utils.data import DataLoader, Dataset
from model_zoo import ResNetBaseline, set_seed
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class KShotSubset(Dataset):
    def __init__(self, full_ds, K, seed):
        self.transform = full_ds.transform
        rng = np.random.RandomState(seed)
        indices_by_class = {i: [] for i in range(len(full_ds.class_names))}
        for idx, lbl in enumerate(full_ds.label_ids):
            indices_by_class[lbl].append(idx)
        selected = []
        for c, idxs in indices_by_class.items():
            selected.extend(rng.choice(idxs, size=min(K, len(idxs)), replace=False))
        self.full_ds = full_ds
        self.indices = sorted(selected)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.full_ds[self.indices[idx]]


def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for batch in loader:
        img, lbl = batch['image'].to(device), batch['label'].to(device)
        opt.zero_grad()
        logits = model(img)
        loss = crit(logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * img.size(0)
        _, p = logits.max(1)
        correct += (p == lbl).sum().item()
        n += lbl.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    for batch in loader:
        img, lbl = batch['image'].to(device), batch['label'].to(device)
        logits = model(img)
        loss = crit(logits, lbl)
        total_loss += loss.item() * img.size(0)
        _, p = logits.max(1)
        correct += (p == lbl).sum().item()
        n += lbl.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seeds', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    K = args.k
    DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}, K={K}, seeds={args.seeds}')

    train_full = TPDDDataset('train', transform=TRAIN_TRANSFORM)
    val_full = TPDDDataset('validation', transform=EVAL_TRANSFORM)
    test_full = TPDDDataset('test', transform=EVAL_TRANSFORM)
    NUM_CLASSES = len(train_full.class_names)

    val_loader = DataLoader(val_full, 32, shuffle=False, num_workers=2,
                            worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_full, 32, shuffle=False, num_workers=2,
                             worker_init_fn=worker_init_fn)

    accs = []
    t0 = time.time()
    for seed in range(args.seeds):
        set_seed(seed)
        train_subset = KShotSubset(train_full, K, seed + 300)
        train_loader = DataLoader(
            train_subset, batch_size=min(32, len(train_subset)),
            shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)

        model = ResNetBaseline(NUM_CLASSES).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss()

        best_val, best_state = 0, None
        for ep in range(args.epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, crit, DEVICE)
            val_loss, val_acc = evaluate(model, val_loader, crit, DEVICE)
            sch.step()
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        assert best_state is not None, (
            "best_state is None after training: validation accuracy was 0 at "
            "every epoch, so no checkpoint was selected. Usually the validation "
            "split is empty or misaligned, or training diverged from step 1.")
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        _, test_acc = evaluate(model, test_loader, crit, DEVICE)
        accs.append(test_acc)

        if seed % 5 == 0 or seed == args.seeds - 1:
            a = np.array(accs)
            elapsed = time.time() - t0
            print(f'  seed {seed:2d}/{args.seeds}: '
                  f'running mean={a.mean()*100:.1f}% ± {a.std(ddof=1)*100:.1f} '
                  f'[{elapsed:.0f}s]')

    a = np.array(accs)
    result = {
        'K': K,
        'n_seeds': args.seeds,
        'epochs': args.epochs,
        'mean': float(a.mean()),
        'std': float(a.std(ddof=1)),
        'sem': float(a.std(ddof=1) / np.sqrt(args.seeds)),
        'accs': [float(x) for x in accs],
        'min': float(a.min()),
        'max': float(a.max()),
        'elapsed_s': float(time.time() - t0),
    }

    print(f'\n=== ResNet-18 K={K} n={args.seeds} ===')
    print(f'  Mean: {a.mean()*100:.2f}%')
    print(f'  Std:  {a.std(ddof=1)*100:.2f}')
    print(f'  SEM:  {a.std(ddof=1)/np.sqrt(args.seeds)*100:.2f}')
    print(f'  Range: [{a.min()*100:.1f}%, {a.max()*100:.1f}%]')

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'resnet_k{K}_n{args.seeds}.json')
    json.dump(result, open(out_path, 'w'), indent=2)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
