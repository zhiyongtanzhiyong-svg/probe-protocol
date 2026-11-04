"""
Full-data provenance regeneration. Closes 3 audit WARNs:
  1. ResNet-18 full-data 15-seed raw accs[]  (was by-reference only)
  2. Per-class full-data accuracy for DFN-CLIP (frozen features) + ResNet-18, labeled
  3. Cost/spec measurements (param counts, inference imgs/s, peak memory, training time)

Protocol matches paper Methods exactly:
  ResNet-18: ImageNet-pretrained, AdamW lr=3e-4, cosine, batch 32, 30 epochs,
             best-validation selection, full TPDD train set (1295), 15 seeds.
  DFN-CLIP full: frozen features (data/clip_features_clean.npz) + LogisticRegression C=1.0.
Output: data/full_provenance.json
"""
import os
import sys, json, time, os, datetime
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ROOT + '/scripts')
from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from model_zoo import ResNetBaseline, set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def perclass_and_overall(model, loader, device, n_classes):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            img, lbl = batch['image'].to(device), batch['label'].to(device)
            preds.append(model(img).max(1)[1].cpu()); labels.append(lbl.cpu())
    preds = torch.cat(preds).numpy(); labels = torch.cat(labels).numpy()
    pc = []
    for c in range(n_classes):
        m = labels == c
        pc.append(float((preds[m] == c).mean()) if m.sum() else float('nan'))
    return np.array(pc), float((preds == labels).mean())

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    train_full = TPDDDataset('train', transform=TRAIN_TRANSFORM)
    val_full = TPDDDataset('validation', transform=EVAL_TRANSFORM)
    test_full = TPDDDataset('test', transform=EVAL_TRANSFORM)
    NC = len(train_full.class_names)
    class_names = list(train_full.class_names)
    val_loader = DataLoader(val_full, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_full, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

    out = {
        'date': datetime.date.today().isoformat(),
        'device': str(DEVICE),
        'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu',
    }

    # ---------- 1. ResNet-18 full, 15 seeds (exact L148 protocol) ----------
    SEEDS, EPOCHS = 15, 30
    accs, perclass_list, train_times = [], [], []
    t0 = time.time()
    for seed in range(SEEDS):
        set_seed(seed)
        train_loader = DataLoader(train_full, batch_size=32, shuffle=True,
                                  num_workers=2, worker_init_fn=worker_init_fn)
        model = ResNetBaseline(NC).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = nn.CrossEntropyLoss()
        best_val, best_state = 0.0, None
        ts = time.time()
        for ep in range(EPOCHS):
            model.train()
            for batch in train_loader:
                img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
                opt.zero_grad()
                loss = crit(model(img), lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            model.eval()
            vc = vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
                    vc += (model(img).max(1)[1] == lbl).sum().item(); vn += lbl.size(0)
            val_acc = vc / vn
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        assert best_state is not None, (
            "best_state is None after training: validation accuracy was 0 at "
            "every epoch, so no checkpoint was selected. Usually the validation "
            "split is empty or misaligned, or training diverged from step 1.")
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        pc, ta = perclass_and_overall(model, test_loader, DEVICE, NC)
        accs.append(ta); perclass_list.append(pc); train_times.append(time.time() - ts)
        a = np.array(accs)
        print(f'  seed {seed:2d}/{SEEDS}: mean={a.mean()*100:.1f}% +/- {a.std(ddof=1)*100:.1f} '
              f'[{time.time()-t0:.0f}s]', flush=True)

    a = np.array(accs)
    out['resnet_full'] = {
        'n_seeds': SEEDS, 'epochs': EPOCHS,
        'mean': float(a.mean()), 'std': float(a.std(ddof=1)),
        'sem': float(a.std(ddof=1) / np.sqrt(SEEDS)),
        'accs': [float(x) for x in accs],
        'min': float(a.min()), 'max': float(a.max()),
        'elapsed_s': float(time.time() - t0),
        'protocol': 'ResNet-18 ImageNet-pretrained, AdamW lr=3e-4, cosine, bs=32, 30ep, best-val, full TPDD train (1295), 15 seeds',
    }
    pc_mat = np.nanmean(np.array(perclass_list, dtype=float), axis=0)
    out['resnet_perclass'] = {class_names[c]: (None if np.isnan(pc_mat[c]) else round(float(pc_mat[c]), 4)) for c in range(NC)}
    print(f"ResNet full: {a.mean()*100:.2f}% +/- {a.std(ddof=1)*100:.2f}", flush=True)
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # ---------- 2. DFN-CLIP full probe (frozen features, LR C=1.0) ----------
    feats = np.load(os.path.join(DATA, 'clip_features_clean.npz'))
    Xtr, ytr = feats['train_feats'].astype('float32'), feats['train_labels'].astype(int)
    Xte, yte = feats['test_feats'].astype('float32'), feats['test_labels'].astype(int)
    t0 = time.time()
    clf = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    clf.fit(Xtr, ytr)
    pred_te = clf.predict(Xte)
    lr_time = time.time() - t0
    overall = float((pred_te == yte).mean())
    pc = np.array([(float((pred_te[yte == c] == c).mean()) if (yte == c).sum() else float('nan')) for c in range(NC)])
    out['dfn_clip_full'] = {
        'model': 'DFN-CLIP ViT-L/14 (frozen)',
        'protocol': 'frozen features (clip_features_clean.npz) + LogisticRegression C=1.0, random_state=42',
        'overall_acc': overall, 'n_train': int(len(ytr)), 'n_test': int(len(yte)),
        'lr_fit_time_s': float(lr_time),
    }
    out['dfn_perclass'] = {class_names[c]: (None if np.isnan(pc[c]) else round(float(pc[c]), 4)) for c in range(NC)}
    print(f"DFN-CLIP full probe: {overall*100:.2f}%", flush=True)

    # ---------- 3. Cost / spec measurements ----------
    cost = {}
    # param counts
    set_seed(0)
    rn = ResNetBaseline(NC)
    cost['resnet_params'] = int(count_params(rn))
    del rn
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    ckpt = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
    cost['clip_ckpt_bytes'] = int(os.path.getsize(ckpt))
    try:
        import open_clip
        clipm, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained=ckpt)
        cost['clip_params'] = int(count_params(clipm))
        clipm = clipm.eval().to(DEVICE)
        # inference throughput: encode_image on all test images (batched)
        imgs = [test_full[i]['image'] for i in range(len(test_full))]
        xte = torch.stack(imgs).to(DEVICE)
        for _ in range(2):
            with torch.no_grad():
                clipm.encode_image(xte)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(DEVICE)
        t = time.time()
        for _ in range(3):
            with torch.no_grad():
                clipm.encode_image(xte)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        dt = (time.time() - t) / 3.0
        cost['clip_encode_imgs_per_s'] = round(len(test_full) / dt, 1)
        if DEVICE.type == 'cuda':
            cost['clip_peak_mem_gb'] = round(torch.cuda.max_memory_allocated(DEVICE) / 1024**3, 3)
        del clipm, xte
        torch.cuda.empty_cache()
    except Exception as e:
        cost['clip_error'] = str(e)
    # ResNet throughput (fresh small model, eval)
    try:
        set_seed(0)
        rn = ResNetBaseline(NC).eval().to(DEVICE)
        imgs = [test_full[i]['image'] for i in range(len(test_full))]
        xte = torch.stack(imgs).to(DEVICE)
        for _ in range(2):
            with torch.no_grad():
                rn(xte)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(DEVICE)
        t = time.time()
        for _ in range(3):
            with torch.no_grad():
                rn(xte)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        dt = (time.time() - t) / 3.0
        cost['resnet_fwd_imgs_per_s'] = round(len(test_full) / dt, 1)
        if DEVICE.type == 'cuda':
            cost['resnet_peak_mem_gb'] = round(torch.cuda.max_memory_allocated(DEVICE) / 1024**3, 3)
        del rn, xte
        torch.cuda.empty_cache()
    except Exception as e:
        cost['resnet_error'] = str(e)
    cost['resnet_train_time_per_seed_s'] = round(float(np.mean(train_times)), 1)
    cost['measurement_note'] = (
        f'Measured {out["date"]} on {out["gpu"]} (12 GB); '
        f'throughput = {len(test_full)} test images, 3 reps after 2 warmup; '
        f'peak memory = torch.cuda.max_memory_allocated; '
        f'ResNet train time = mean of 15 full-data seeds (30 epochs each).')
    out['cost'] = cost
    print('COST:', json.dumps(cost, indent=1), flush=True)

    out_path = os.path.join(DATA, 'full_provenance.json')
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f'\nSaved to {out_path}', flush=True)

if __name__ == '__main__':
    main()
