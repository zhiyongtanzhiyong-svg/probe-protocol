"""A4 (reviewer): PAIRED K-shot CLIP vs ResNet.
Both models are trained on the IDENTICAL K-shot support set per seed
(KShotSubset(train_full, K, seed+500), identical to CLIP's probe), so the
comparison is seed-paired. CLIP per-seed accs are loaded from the existing
clip_fewshot.json (seed 0..14); ResNet-18 is re-run with the matching
seed+500 support sets. Reports paired mean delta, paired t, Wilcoxon, and a
bootstrap 95% CI on the mean delta.
"""
import os
import sys, os, json, time
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, ROOT + '/scripts')
from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from torch.utils.data import DataLoader, Dataset
from model_zoo import ResNetBaseline, set_seed
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class KShotSubset(Dataset):
    def __init__(self, full_ds, K, seed):
        self.transform = full_ds.transform
        rng = np.random.RandomState(seed)
        by_class = {i: [] for i in range(len(full_ds.class_names))}
        for idx, lbl in enumerate(full_ds.label_ids):
            by_class[lbl].append(idx)
        sel = []
        for c, idxs in by_class.items():
            sel.extend(rng.choice(idxs, size=min(K, len(idxs)), replace=False))
        self.full_ds, self.indices = full_ds, sorted(sel)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.full_ds[self.indices[idx]]


def train_one_epoch(model, loader, opt, crit):
    model.train()
    correct = n = 0
    for b in loader:
        img, lbl = b['image'].to(DEVICE), b['label'].to(DEVICE)
        opt.zero_grad()
        logits = model(img)
        crit(logits, lbl).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        p = logits.max(1)[1]
        correct += (p == lbl).sum().item(); n += lbl.size(0)
    return correct / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = n = 0
    for b in loader:
        img, lbl = b['image'].to(DEVICE), b['label'].to(DEVICE)
        p = model(img).max(1)[1]
        correct += (p == lbl).sum().item(); n += lbl.size(0)
    return correct / n


def resnet_kshot_acc(train_full, val_loader, test_loader, K, seed, epochs=30):
    set_seed(seed)
    subset = KShotSubset(train_full, K, seed + 500)          # MATCH CLIP
    loader = DataLoader(subset, batch_size=min(32, len(subset)), shuffle=True,
                        num_workers=2, worker_init_fn=worker_init_fn)
    model = ResNetBaseline(len(train_full.class_names)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    for _ in range(epochs):
        train_one_epoch(model, loader, opt, crit)
        va = evaluate(model, val_loader)
        sch.step()
        if va > best:
            best, best_state = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None, (
        "best_state is None after training: validation accuracy was 0 at "
        "every epoch, so no checkpoint was selected. Usually the validation "
        "split is empty or misaligned, or training diverged from step 1.")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return evaluate(model, test_loader)


def main():
    clip_few = json.load(open(os.path.join(ROOT, 'data', 'clip_fewshot.json')))
    train_full = TPDDDataset('train', transform=TRAIN_TRANSFORM)
    val_loader = DataLoader(TPDDDataset('validation', transform=EVAL_TRANSFORM), 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(TPDDDataset('test', transform=EVAL_TRANSFORM), 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    out = {}
    t0 = time.time()
    for K in [1, 5, 10]:
        clip_accs = np.array(clip_few[f'K={K}']['accs'])
        rn = []
        for seed in range(15):
            rn.append(resnet_kshot_acc(train_full, val_loader, test_loader, K, seed))
            if seed % 5 == 0:
                print(f'  K={K} seed {seed}: running RN {np.array(rn).mean()*100:.1f}%', flush=True)
        rn = np.array(rn)
        delta = clip_accs - rn
        tt = stats.ttest_rel(clip_accs, rn)
        try:
            ww = stats.wilcoxon(clip_accs, rn, zero_method='pratt')
        except Exception:
            ww = (np.nan, np.nan)
        B = 5000
        rng = np.random.RandomState(0)
        boots = [delta[rng.randint(0, len(delta), len(delta))].mean() for _ in range(B)]
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        out[f'K={K}'] = {
            'clip_mean': float(clip_accs.mean()), 'resnet_mean': float(rn.mean()),
            'resnet_std': float(rn.std(ddof=1)),
            'paired_delta_mean': float(delta.mean()),
            'delta_ci95_bootstrap': ci,
            'paired_t': float(tt.statistic), 'paired_t_p': float(tt.pvalue),
            'wilcoxon_p': float(ww[1]),
            'clip_accs': [float(x) for x in clip_accs],
            'resnet_accs': [float(x) for x in rn],
        }
        print(f'\n=== K={K} PAIRED (same support set/seed) ===', flush=True)
        print(f'  CLIP   {clip_accs.mean()*100:.1f}%  |  ResNet {rn.mean()*100:.1f}% +/- {rn.std(ddof=1)*100:.1f}', flush=True)
        print(f'  delta (CLIP-ResNet) = {delta.mean()*100:+.2f} pp  95%CI [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]', flush=True)
        print(f'  paired t={tt.statistic:.3f} (p={tt.pvalue:.4f}),  Wilcoxon p={ww[1]:.4f}', flush=True)
    out['elapsed_s'] = time.time() - t0
    out['note'] = 'ResNet re-run with CLIP seed+500 support sets -> identical per-seed support; seed-paired'
    p = os.path.join(ROOT, 'data', 'paired_kshot_a4.json')
    json.dump(out, open(p, 'w'), indent=2)
    print(f'\nSaved {p} in {out["elapsed_s"]:.0f}s')


if __name__ == '__main__':
    main()
