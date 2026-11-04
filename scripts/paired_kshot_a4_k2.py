"""Add the missing K=2 seed-paired ResNet-18 contrast.

WHY (2026-10-05)
----------------
data/paired_kshot_a4.json covered K in {1,5,10}, and
data/two_model_e4.json's comparison table carried a literal null for K=2
("resnet": None). The regime-dependence claim therefore located the CLIP-vs-CNN
crossover only to the interval (1,5): LAION-2B led at K=1 by +7.05 pp and
trailed by -7.05 pp at K=5, with nothing in between. K=2 is the cheapest point
that can break that bracket.

PROVENANCE
----------
ResNet-18 is trained on the identical K=2-per-class support set each CLIP probe
uses per seed (KShotSubset(train_full, 2, seed+500)). The draw is verified
identical to the one clip_fewshot_probe.py now uses, because
data/tpdd/*/labels.json is class-sorted, so dict-insertion order equals sorted
class order. ResNet keeps TRAIN_TRANSFORM because it trains for 30 epochs and
there the augmentation is genuine, not a single stochastic draw.

K in {1,5,10} ResNet per-seed accuracies are NOT recomputed here; they are read
import os
from the existing paired_kshot_a4.json, which used the same recipe and the same
support draws. Only the K=2 row is new.

Writes data/paired_kshot_a4_k2.json.
"""
import sys, os, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = ROOT
sys.path.insert(0, f'{REPO}/scripts')

from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from paired_kshot_a4 import resnet_kshot_acc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_SEEDS = 15
K = 2


def main():
    clip_few = json.load(open(f'{REPO}/data/clip_fewshot.json'))
    prior = json.load(open(f'{REPO}/data/paired_kshot_a4.json'))

    train_full = TPDDDataset('train', transform=TRAIN_TRANSFORM)
    val_loader = DataLoader(TPDDDataset('validation', transform=EVAL_TRANSFORM), 32,
                            shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(TPDDDataset('test', transform=EVAL_TRANSFORM), 32,
                             shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

    t0 = time.time()
    rn = []
    for seed in range(N_SEEDS):
        rn.append(resnet_kshot_acc(train_full, val_loader, test_loader, K, seed))
        print(f'  K={K} seed {seed:2d}: running mean {np.mean(rn)*100:.1f}%', flush=True)
    rn = np.array(rn)

    out = {
        f'K={K}': {
            'resnet_mean': float(rn.mean()),
            'resnet_std': float(rn.std(ddof=1)),
            'resnet_accs': [float(x) for x in rn],
            'n_seeds': N_SEEDS,
            'epochs': 30,
            'support_transform': 'TRAIN_TRANSFORM (30 epochs, genuine augmentation)',
            'support_indices': 'KShotSubset(train_full, 2, seed+500)',
        },
        'elapsed_s': time.time() - t0,
        'prior_run_elapsed_s': prior.get('elapsed_s'),
        'note': ('K=2 added 2026-10-05 to close the crossover bracket (1,5). '
                 'Same ResNet recipe and same support draws as the K in {1,5,10} '
                 'rows in paired_kshot_a4.json, which are reused unchanged.'),
    }
    json.dump(out, open(f'{REPO}/data/paired_kshot_a4_k2.json', 'w'), indent=2)
    print(f'\nResNet K=2: {rn.mean()*100:.2f}% +- {rn.std(ddof=1)*100:.2f} '
          f'(n={N_SEEDS}) in {out["elapsed_s"]:.0f}s')
    print(f'Saved {REPO}/data/paired_kshot_a4_k2.json')


if __name__ == '__main__':
    main()
