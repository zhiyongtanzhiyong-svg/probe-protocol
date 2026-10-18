"""
M0 gate (R002): does a foreground-anchored patch-attention head on the FROZEN
CLIP ViT-L/14 carry class signal in the patch tokens, and does it beat the
paper's CLS linear probe?

This is the go/stop gate for the whole A+B expansion. Two checks on dfn_clip:
  1. OVERFIT  -- head trained on ALL train must reach high train acc
                 (proves patch tokens carry class signal + harness is not broken).
  2. 10-SHOT  -- head vs CLS linear probe, paired per seed on the 138 test imgs
                 (proves patch-token reading is at least as good as the CLS readout).

GO   if  overfit_train_acc >= 0.90  AND  mean(head_10shot) >= mean(probe_10shot).
STOP otherwise  (patch-token reading is dead; do not spend days on M3).

Patch tokens are per-token L2-normalised so the head is invariant to the
28x magnitude gap between dfn_clip (0.013) and laion2b (0.347) embeddings.

Outputs data/m0_gate.json.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM
from patch_tokens import (CKPT, DEVICE, extract_patch_features, get_tokens,
                          _preprocess_batch, tensor_to_pil)
from coop_experiment import kshot_subset_idx

CKPT_NAME = 'dfn_clip'
CACHE = os.path.join(REPO, 'data', f'patch_tokens_{CKPT_NAME}.npz')
OUT = os.path.join(REPO, 'data', 'm0_gate.json')
D = 768
N_CLS = 16


class PatchAttnHead(nn.Module):
    """Foreground-anchored attention pool over patch tokens -> linear head.

    X: (B, P, D) per-token unit-norm patch tokens.
    pooled = softmax(X @ q) @ X   (q free, init N(0,0.1) -> near-uniform start)
    logits = Linear(pooled, n_cls)
    """
    def __init__(self, d=D, n_cls=N_CLS):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.1)
        self.head = nn.Linear(d, n_cls)

    def forward(self, X):
        scores = X @ self.q                       # (B, P)
        w = F.softmax(scores, dim=-1)             # (B, P)
        pooled = (w.unsqueeze(1) @ X).squeeze(1)  # (B, D)
        return self.head(pooled)


def norm_tokens(patches):
    p = patches.astype(np.float32)
    n = np.linalg.norm(p, axis=-1, keepdims=True) + 1e-12
    return (p / n).astype(np.float32)


def load_or_extract():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z['cls'], z['patches'], z['labels'], z['test_cls'], z['test_patches'], z['test_labels']
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT[CKPT_NAME])
    model = model.to(DEVICE).eval()
    tr_ds = TPDDDataset(split='train', transform=EVAL_TRANSFORM)
    te_ds = TPDDDataset(split='test', transform=EVAL_TRANSFORM)
    cls, patches, labels = extract_patch_features(model, preprocess, tr_ds)
    te_cls, te_patches, te_labels = extract_patch_features(model, preprocess, te_ds)
    del model; torch.cuda.empty_cache()
    np.savez_compressed(CACHE, cls=cls, patches=patches, labels=labels,
                        test_cls=te_cls, test_patches=te_patches, test_labels=te_labels)
    return cls, patches, labels, te_cls, te_patches, te_labels


def train_head(X, y, Xte, yte, seed, epochs=200, lr=1e-3):
    """Full-batch Adam train of PatchAttnHead. Returns (train_acc, test_acc)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = PatchAttnHead().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X, device=DEVICE); yt = torch.tensor(y, device=DEVICE)
    Xtt = torch.tensor(Xte, device=DEVICE); ytt = torch.tensor(yte, device=DEVICE)
    best = (-1, 0.0)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt), yt)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = float((model(Xt).argmax(1) == yt).float().mean())
        te_acc = float((model(Xtt).argmax(1) == ytt).float().mean())
    return tr_acc, te_acc


def main():
    t0 = time.time()
    print(f"Device {DEVICE} | ckpt {CKPT_NAME}")
    cls, patches, labels, te_cls, te_patches, te_labels = load_or_extract()
    print(f"train {patches.shape} | test {te_patches.shape} | classes {len(set(labels.tolist()))}")
    assert len(set(labels.tolist())) == N_CLS and len(te_labels) == 138

    cls = cls / (np.linalg.norm(cls, axis=-1, keepdims=True) + 1e-12)
    te_cls = te_cls / (np.linalg.norm(te_cls, axis=-1, keepdims=True) + 1e-12)
    P = norm_tokens(patches); Pte = norm_tokens(te_patches)

    R = {'ckpt': CKPT_NAME, 'n_train': int(len(labels)), 'n_test': int(len(te_labels)),
         'n_patches': int(patches.shape[1]), 'dim': D}

    # --- [1] OVERFIT: head on ALL train must fit (patch tokens carry signal) ---
    tr_acc, te_acc = train_head(P, labels, Pte, te_labels, seed=0, epochs=300)
    R['overfit'] = {'train_acc': round(tr_acc, 4), 'test_acc': round(te_acc, 4)}
    print(f"[1] OVERFIT  train {tr_acc*100:.1f}%  test {te_acc*100:.1f}%")

    # --- [2] 10-SHOT: head vs CLS linear probe, paired per seed ---
    head_accs, probe_accs = [], []
    for seed in range(5):
        sup = kshot_subset_idx(labels, 10, seed + 500)
        # CLS probe (paper's baseline)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        clf.fit(cls[sup], labels[sup])
        probe_accs.append(float((clf.predict(te_cls) == te_labels).mean()))
        # patch-attention head
        _, h_acc = train_head(P[sup], labels[sup], Pte, te_labels, seed=seed, epochs=300)
        head_accs.append(h_acc)
    R['shot10'] = {
        'head': {'mean': round(float(np.mean(head_accs)), 4), 'accs': [round(a, 4) for a in head_accs]},
        'probe_cls': {'mean': round(float(np.mean(probe_accs)), 4), 'accs': [round(a, 4) for a in probe_accs]},
        'delta_pp': round((float(np.mean(head_accs)) - float(np.mean(probe_accs))) * 100, 2),
    }
    print(f"[2] 10-SHOT  head {np.mean(head_accs)*100:.1f}%  probe {np.mean(probe_accs)*100:.1f}%  "
          f"delta {R['shot10']['delta_pp']:+.1f}pp")

    go = (R['overfit']['train_acc'] >= 0.90) and (R['shot10']['delta_pp'] >= 0.0)
    R['verdict'] = 'GO' if go else 'STOP'
    R['criteria'] = {'overfit_train_acc': R['overfit']['train_acc'], 'overfit_ok': R['overfit']['train_acc'] >= 0.90,
                     'shot10_delta_pp': R['shot10']['delta_pp'], 'shot10_ok': R['shot10']['delta_pp'] >= 0.0}
    R['elapsed_sec'] = round(time.time() - t0, 1)
    json.dump(R, open(OUT, 'w'), indent=2)
    print(f"\nVERDICT: {R['verdict']}   (overfit {R['overfit']['train_acc']:.3f}>=0.90, "
          f"10-shot delta {R['shot10']['delta_pp']:+.1f}pp>=0)")
    print(f"Saved {OUT} in {R['elapsed_sec']}s")


if __name__ == '__main__':
    main()
