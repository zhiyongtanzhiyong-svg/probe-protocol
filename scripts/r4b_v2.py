"""R4b v2: Clean protocol - validation-based selection. Simplified."""
import os
import sys, json, torch, numpy as np, time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
sys.path.insert(0, ROOT + '/scripts')
import open_clip
from dataset import TPDDDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM, worker_init_fn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIP_CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device('cuda')
DATA_DIR = ROOT + '/data'
RESULTS = {}

def tensor_to_pil(img_tensor):
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

class SubDS(Dataset):
    def __init__(self, ds, idx):
        self.ds, self.idx = ds, idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return self.ds[self.idx[i]]

def ext_clip(model, preprocess, dataset, device):
    loader = DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    F, L = [], []
    with torch.no_grad():
        for batch in loader:
            ci = torch.stack([preprocess(tensor_to_pil(img)) for img in batch['image']]).to(device)
            f = model.encode_image(ci)
            f = f / f.norm(dim=-1, keepdim=True)
            F.append(f.cpu().numpy())
            L.extend(batch['label'].numpy().tolist())
    return np.concatenate(F), np.array(L)

def ext_rn(resnet, dataset, device):
    loader = DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    F, L = [], []
    orig = resnet.fc
    resnet.fc = nn.Identity()
    with torch.no_grad():
        for batch in loader:
            f = resnet(batch['image'].to(device))
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            F.append(f.cpu().numpy())
            L.extend(batch['label'].numpy().tolist())
    resnet.fc = orig
    return np.concatenate(F), np.array(L)

def zs_eval(model, tokenizer, preprocess, dataset, class_names, template, device):
    texts = [template.format(name=c.replace('_', ' ')) for c in class_names]
    toks = tokenizer(texts).to(device)
    with torch.no_grad():
        tf = model.encode_text(toks)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    loader = DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    P, L = [], []
    with torch.no_grad():
        for batch in loader:
            ci = torch.stack([preprocess(tensor_to_pil(img)) for img in batch['image']]).to(device)
            fi = model.encode_image(ci)
            fi = fi / fi.norm(dim=-1, keepdim=True)
            P.extend((fi @ tf.T).argmax(dim=-1).cpu().numpy().tolist())
            L.extend(batch['label'].numpy().tolist())
    return np.array(P), np.array(L)

def main():
    t0 = time.time()
    train_ds = TPDDDataset(split='train', transform=TRAIN_TRANSFORM)
    test_ds = TPDDDataset(split='test', transform=EVAL_TRANSFORM)
    class_names = list(train_ds.tpd_map.keys())
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {len(class_names)}")

    # Load CLIP
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CLIP_CKPT)
    clip_tok = open_clip.get_tokenizer('ViT-L-14')
    clip_model = clip_model.to(DEVICE).eval()

    # Extract features
    print("Extracting CLIP features...")
    cf_tr, cl_tr = ext_clip(clip_model, preprocess, train_ds, DEVICE)
    cf_te, cl_te = ext_clip(clip_model, preprocess, test_ds, DEVICE)
    print(f"  CLIP: train {cf_tr.shape}, test {cf_te.shape}")

    print("Extracting frozen ResNet features...")
    rn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE).eval()
    rf_tr, rl_tr = ext_rn(rn, train_ds, DEVICE)
    rf_te, rl_te = ext_rn(rn, test_ds, DEVICE)
    print(f"  ResNet: train {rf_tr.shape}, test {rf_te.shape}")

    # === 1. C selection on validation ===
    print("\n[1] C selection on validation...")
    _, idx_val = train_test_split(np.arange(len(cl_tr)), test_size=0.2, random_state=42, stratify=cl_tr)
    X_val, y_val = cf_tr[idx_val], cl_tr[idx_val]
    X_tr = cf_tr[np.setdiff1d(np.arange(len(cl_tr)), idx_val)]
    y_tr = cl_tr[np.setdiff1d(np.arange(len(cl_tr)), idx_val)]

    C_grid = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    val_accs = {}
    for C in C_grid:
        lr = LogisticRegression(C=C, max_iter=5000, random_state=42)
        lr.fit(X_tr, y_tr)
        val_accs[C] = accuracy_score(y_val, lr.predict(X_val))
    best_C = max(val_accs, key=val_accs.get)
    print(f"  Val accs: {{{', '.join(f'C={k}:{v*100:.1f}%' for k,v in val_accs.items())}}}")
    print(f"  Best C on val: {best_C}")

    # Retrain on full train, eval on test
    lr_best = LogisticRegression(C=best_C, max_iter=5000, random_state=42)
    lr_best.fit(cf_tr, cl_tr)
    best_preds = lr_best.predict(cf_te)
    best_acc = accuracy_score(cl_te, best_preds)
    best_bal = balanced_accuracy_score(cl_te, best_preds)
    best_f1 = f1_score(cl_te, best_preds, average='macro')

    lr_c1 = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    lr_c1.fit(cf_tr, cl_tr)
    c1_acc = accuracy_score(cl_te, lr_c1.predict(cf_te))

    print(f"  TEST: C={best_C} -> {best_acc*100:.1f}% (bal={best_bal*100:.1f}%, F1={best_f1*100:.1f}%)")
    print(f"  TEST: C=1 (default) -> {c1_acc*100:.1f}%")

    RESULTS['C_selection'] = {
        'val_accs': {str(k): round(v, 4) for k, v in val_accs.items()},
        'best_C': best_C,
        'test_best_C': round(float(best_acc), 4),
        'test_C1': round(float(c1_acc), 4),
        'test_bal': round(float(best_bal), 4),
        'test_f1': round(float(best_f1), 4),
    }

    # === 2. Prompt selection on validation ===
    print("\n[2] Prompt selection on validation...")
    val_ds = SubDS(train_ds, idx_val.tolist())
    templates = {
        'EN-Label': 'a photo of {name}',
        'EN-Photo': 'a photo of a tobacco leaf with {name}',
        'EN-Symptom': 'a tobacco leaf showing symptoms of {name}',
        'EN-Disease': '{name} disease on tobacco leaf',
        'EN-Closeup': 'a closeup of a tobacco leaf with {name}',
        'EN-Expert': 'tobacco plant pathology: {name} disease identification',
        'CN-Label': '\u4e00\u5f20{name}\u7684\u7167\u7247',
        'CN-Photo': '\u4e00\u5f20\u60a3\u6709{name}\u7684\u9ec4\u8349\u53f6\u7247\u7684\u7167\u7247',
        'CN-Symptom': '\u4e00\u5f20\u663e\u793a{name}\u75c7\u72b6\u7684\u9ec4\u8349\u53f6\u7247',
        'CN-Disease': '\u4e00\u5f20\u611f\u67d3\u4e86{name}\u7684\u9ec4\u8349\u682a\u68f6\u7167\u7247',
        'CN-Closeup': '\u4e00\u5f20{name}\u9ec4\u8349\u53f6\u7247\u7684\u7279\u5199\u7167\u7247',
        'KG-Symptom': 'tobacco leaf: {name} disease',
        'KG-Condition': 'tobacco leaf disease occurring when: {name}',
        'KG-Combined': 'tobacco leaf: {name} disease; when: {name}',
    }
    p_val = {}
    for name, tpl in templates.items():
        p, l = zs_eval(clip_model, clip_tok, preprocess, val_ds, class_names, tpl, DEVICE)
        p_val[name] = accuracy_score(l, p)
    best_p = max(p_val, key=p_val.get)
    print(f"  Best prompt on val: {best_p} ({p_val[best_p]*100:.1f}%)")

    # Test with best prompt
    pt, lt = zs_eval(clip_model, clip_tok, preprocess, test_ds, class_names, templates[best_p], DEVICE)
    p_test = accuracy_score(lt, pt)
    print(f"  TEST ({best_p}): {p_test*100:.1f}%")

    RESULTS['prompt_selection'] = {
        'val_accs': {k: round(v, 4) for k, v in p_val.items()},
        'best_prompt': best_p,
        'test_acc': round(float(p_test), 4),
    }

    # === 3. Representation matrix ===
    print("\n[3] Representation matrix...")
    mlp_c = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=1000, random_state=42)
    mlp_c.fit(cf_tr, cl_tr)
    c_mlp = accuracy_score(cl_te, mlp_c.predict(cf_te))

    mlp_r = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=1000, random_state=42)
    mlp_r.fit(rf_tr, rl_tr)
    r_mlp = accuracy_score(rl_te, mlp_r.predict(rf_te))
    print(f"  CLIP MLP: {c_mlp*100:.1f}%, ResNet MLP: {r_mlp*100:.1f}%")

    RESULTS['matrix'] = {
        'clip_linear_C1': round(float(c1_acc), 4),
        'clip_linear_bestC': round(float(best_acc), 4),
        'clip_mlp': round(float(c_mlp), 4),
        'resnet_linear': 0.681,
        'resnet_mlp': round(float(r_mlp), 4),
        'resnet_finetuned': 0.864,
    }

    # === 4. TOST ===
    print("\n[4] TOST...")
    rn_preds = np.load(f'{DATA_DIR}/p2_resnet_full_preds.npy')
    tlabels = np.load(f'{DATA_DIR}/p2_test_labels.npy')
    ca = (best_preds == tlabels).astype(float)
    cb = (rn_preds == tlabels).astype(float)
    diff = ca - cb
    rng = np.random.RandomState(42)
    n = len(tlabels)
    boot = np.array([diff[rng.choice(n, n, replace=True)].mean() for _ in range(10000)])
    ci_l, ci_h = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    eq = (ci_l > -0.05) and (ci_h < 0.05)
    print(f"  CLIP(C={best_C}): {ca.mean()*100:.1f}%, ResNet: {cb.mean()*100:.1f}%")
    print(f"  Diff: {(ca.mean()-cb.mean())*100:+.1f}pp, CI: [{ci_l*100:.1f}, {ci_h*100:.1f}]pp")
    print(f"  TOST 5pp: equivalent={eq}")

    RESULTS['tost'] = {
        'clip_acc': round(float(ca.mean()), 4),
        'resnet_acc': round(float(cb.mean()), 4),
        'diff_pp': round(float((ca.mean() - cb.mean()) * 100), 2),
        'ci_pp': [round(float(ci_l * 100), 2), round(float(ci_h * 100), 2)],
        'equiv_5pp': bool(eq),
    }

    RESULTS['elapsed'] = round(time.time() - t0, 1)
    with open(f'{DATA_DIR}/r4b_v2.json', 'w') as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nDone in {RESULTS['elapsed']}s. Saved.")

if __name__ == '__main__':
    main()
