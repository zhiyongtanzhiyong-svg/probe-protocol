"""
R2 Experiments (fixed):
1. ResNet-18 frozen features + linear probe
2. CLIP tuned head (C=30) n=15 seeds
3. LAION-2B balanced accuracy (K=1,5,10,full)
"""
import os
import sys, json, torch, numpy as np, time, os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from scipy.stats import binomtest
sys.path.insert(0, ROOT + '/scripts')

import open_clip
from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from model_zoo import ResNetBaseline, set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DFN_CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
LAION_CKPT = 'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
N_SEEDS = 15


def tensor_to_pil(img_tensor):
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)


class KShotSubset(Dataset):
    def __init__(self, full_ds, K, seed):
        self.transform = full_ds.transform
        rng = np.random.RandomState(seed)
        indices_by_class = {i: [] for i in range(len(full_ds.class_names))}
        for idx, lbl in enumerate(full_ds.label_ids):
            indices_by_class[lbl].append(idx)
        idxs = []
        for c, idx_list in indices_by_class.items():
            idxs.extend(rng.choice(idx_list, size=min(K, len(idx_list)), replace=False))
        self.full_ds = full_ds
        self.indices = sorted(idxs)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.full_ds[self.indices[idx]]


def clip_extract(model, preprocess, dataset, device):
    loader = DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    feats, labs = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image']
            cimgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).to(device)
            f = model.encode_image(cimgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


def resnet_extract(model, dataset, device):
    loader = DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    feats, labs = [], []
    with torch.no_grad():
        for batch in loader:
            f = model.enc(batch['image'].to(device))
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


def metrics(preds, labels):
    return (float(accuracy_score(labels, preds)),
            float(balanced_accuracy_score(labels, preds)),
            float(f1_score(labels, preds, average='macro', zero_division=0)))


def mcnemar(pa, pb, labels):
    a_ok, b_ok = (pa == labels), (pb == labels)
    n10, n01 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    nd = n01 + n10
    if nd == 0: return 0, 0, 0, 1.0
    return nd, n01, n10, float(binomtest(min(n01, n10), nd, 0.5).pvalue)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    train_full = TPDDDataset('train', transform=TRAIN_TRANSFORM)
    val_full = TPDDDataset('validation', transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)
    NC = len(train_full.class_names)
    test_labels = np.array(test_ds.label_ids)
    results = {'n_test': len(test_ds), 'n_classes': NC, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

    # ========== 1. ResNet-18 FROZEN + linear probe ==========
    print('\n=== 1. ResNet-18 Frozen + Linear Probe ===')
    rn = ResNetBaseline(NC).to(device).eval()
    rn_k1_sub = KShotSubset(train_full, 1, seed=300)
    rn_k1_f, rn_k1_l = resnet_extract(rn, rn_k1_sub, device)
    rn_full_f, rn_full_l = resnet_extract(rn, train_full, device)
    rn_test_f, _ = resnet_extract(rn, test_ds, device)
    del rn; torch.cuda.empty_cache()

    for name, tf, tl in [('rn_frozen_k1', rn_k1_f, rn_k1_l), ('rn_frozen_full', rn_full_f, rn_full_l)]:
        accs, bals, f1s = [], [], []
        for s in range(N_SEEDS):
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=s)
            clf.fit(tf, tl)
            a, b, f = metrics(clf.predict(rn_test_f), test_labels)
            accs.append(a); bals.append(b); f1s.append(f)
        results[name] = {'acc': [float(np.mean(accs)), float(np.std(accs, ddof=1))],
                         'bal': [float(np.mean(bals)), float(np.std(bals, ddof=1))],
                         'f1': [float(np.mean(f1s)), float(np.std(f1s, ddof=1))], 'n': N_SEEDS}
        print(f'  {name:18s}: acc={np.mean(accs)*100:.1f}±{np.std(accs,ddof=1)*100:.1f}  '
              f'bal={np.mean(bals)*100:.1f}±{np.std(bals,ddof=1)*100:.1f}  F1={np.mean(f1s)*100:.1f}')

    # ========== 2. CLIP DFN tuned head C=30, n=15 ==========
    print('\n=== 2. CLIP DFN Tuned Head (C=30) n=15 ===')
    m, _, pp = open_clip.create_model_and_transforms('ViT-L-14', pretrained=DFN_CKPT)
    m = m.eval().to(device)
    dfn_tr_f, dfn_tr_l = clip_extract(m, pp, train_full, device)
    dfn_te_f, _ = clip_extract(m, pp, test_ds, device)
    del m; torch.cuda.empty_cache()

    for C in [1.0, 30.0]:
        accs, bals, f1s, p0 = [], [], [], None
        for s in range(N_SEEDS):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=s)
            clf.fit(dfn_tr_f, dfn_tr_l)
            p = clf.predict(dfn_te_f)
            a, b, f = metrics(p, test_labels)
            accs.append(a); bals.append(b); f1s.append(f)
            if s == 0: p0 = p
        key = f'dfn_c{int(C)}_full'
        results[key] = {'acc': [float(np.mean(accs)), float(np.std(accs, ddof=1))],
                        'bal': [float(np.mean(bals)), float(np.std(bals, ddof=1))],
                        'f1': [float(np.mean(f1s)), float(np.std(f1s, ddof=1))],
                        'range': [float(np.min(accs)), float(np.max(accs))], 'n': N_SEEDS}
        print(f'  DFN C={C:<5}: acc={np.mean(accs)*100:.1f}±{np.std(accs,ddof=1)*100:.1f}  '
              f'bal={np.mean(bals)*100:.1f}±{np.std(bals,ddof=1)*100:.1f}  '
              f'F1={np.mean(f1s)*100:.1f}  [{np.min(accs)*100:.1f}-{np.max(accs)*100:.1f}]')

    # McNemar: DFN C=30 seed0 vs ResNet full (from p2)
    try:
        rn_fp = np.load(ROOT + '/data/p2_resnet_full_preds.npy')
        nd, n01, n10, pm = mcnemar(p0, rn_fp, test_labels)
        results['mcnemar_dfn_c30_vs_rn_full'] = {'n_disc': nd, 'p': pm, 'sig': pm < 0.05}
        print(f'  McNemar DFN-C30 vs RN-full: n={nd}, p={pm:.4f}')
    except Exception as e:
        print(f'  McNemar skipped: {e}')

    # ========== 3. LAION-2B K=1,5,10,full ==========
    print('\n=== 3. LAION-2B All K ===')
    m2, _, pp2 = open_clip.create_model_and_transforms('ViT-L-14', pretrained=LAION_CKPT)
    m2 = m2.eval().to(device)
    la_tr_f, la_tr_l = clip_extract(m2, pp2, train_full, device)
    la_te_f, _ = clip_extract(m2, pp2, test_ds, device)
    # Accumulate per-K: assigning to la_kf/la_kl inside the loop left the
    # variables holding the LAST iteration (K=10), which was then written out
    # under the key 'laion_k1' -- so the reported "K=1" row was K=10 and no
    # K=1 or K=5 row existed at all.
    la_kshots = {}
    for K in [1, 5, 10]:
        sub = KShotSubset(train_full, K, seed=500)
        la_kshots[K] = clip_extract(m2, pp2, sub, device)
    del m2; torch.cuda.empty_cache()

    arms = [(f'laion_k{K}', tf, tl) for K, (tf, tl) in sorted(la_kshots.items())]
    arms.append(('laion_full', la_tr_f, la_tr_l))
    for name, tf, tl in arms:
        accs, bals, f1s = [], [], []
        for s in range(N_SEEDS):
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=s)
            clf.fit(tf, tl)
            a, b, f = metrics(clf.predict(la_te_f), test_labels)
            accs.append(a); bals.append(b); f1s.append(f)
        results[name] = {'acc': [float(np.mean(accs)), float(np.std(accs, ddof=1))],
                        'bal': [float(np.mean(bals)), float(np.std(bals, ddof=1))],
                        'f1': [float(np.mean(f1s)), float(np.std(f1s, ddof=1))], 'n': N_SEEDS}
        print(f'  {name:12s}: acc={np.mean(accs)*100:.1f}±{np.std(accs,ddof=1)*100:.1f}  '
              f'bal={np.mean(bals)*100:.1f}±{np.std(bals,ddof=1)*100:.1f}  F1={np.mean(f1s)*100:.1f}')

    results['elapsed_s'] = float(time.time() - t0)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'r2_results.json')
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\nSaved to {out}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
