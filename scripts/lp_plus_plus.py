"""
LP++ (Huang et al. 2024, arXiv:2404.02285, "LP++: A Surprisingly Strong
Linear Probe for Few-Shot CLIP") head comparison for this paper.

Addresses the reviewer demand to benchmark against the known-stronger few-shot
CLIP probe instead of only the naive linear probe.

Faithful port of the official code (reference_lpp/methods/linear_probe_p2.py,
github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline):

    logit(x) = W x + b + alpha . (x T)

  - W (d x 16) initialized to the per-class feature SUM (the official
    compute_centroids does not divide by class size - the mean-division line
    is commented out in the reference); trained with SGD(momentum=0.9) at the
    implicit MM step lr_w = 4n / lambda_max(X'X).
  - alpha: 16-dim vector initialized to the shared scalar
    alpha0 = 250 * mean over the per-class-mean text-logit matrix (the reference's
    *shots and /shots cancel),
    updated every 10 epochs: alpha -= lr_alpha * grad, with lr_alpha =
    n / (4 * max_c sum_i (x_i . t_c)^2).  alpha.grad is ACCUMULATED across
    epochs (the reference never zeroes it) - reproduced exactly.
  - 300 epochs; early stop on validation accuracy; report test accuracy at
    the best-validation epoch.

Adaptation to the matched-budget protocol (documented deviations):
  - open_clip ViT-L/14 (dfn_clip primary, laion2b); fp32 (reference: fp16).
  - Text side = the paper's own 14-template zero-shot set: per-class mean of
    the 14 normalized template embeddings, NOT renormalized, so the text
    logits reproduce the paper's uniform zero-shot ensemble exactly.  (The
    reference clip_classifier renormalizes; the per-class scale difference is
    absorbed by the learned alpha.)
  - Full data: train = 1295, val = 20% stratified (seed 42) - the SAME
    validation split as the canonical probe C-selection; single run.
  - K-shot (K in 1,2,5,10; 15 seeds): support = the SAME stratified
    K-per-class subsets (seed+500) as the paper's K-shot probe, so LP++ vs
    probe is paired per seed.  LP++ additionally needs a validation subset
    for early stopping - drawn disjoint from the support (seed+1500),
    following the reference's separate K-shot val design.
  - Determinism: torch.manual_seed(seed) per run (the reference does not
    seed; required for reproducible 15-seed tables).

Outputs data/lp_plus_plus.json.
"""
import os
import sys, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, ROOT + "/scripts")
import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM, worker_init_fn
from coop_experiment import kshot_subset_idx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPT = {
    'dfn_clip': 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin',
    'laion2b': 'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin',
}
# REPOINTED 2026-10-05. The previous DFN entry, data/clip_features_clean.npz,
# is a stale store: max |diff| 0.1185 against the current corpus on
# L2-normalized features (per-sample cosine 0.83--0.97), and it yields a
# full-data C=1 of 76.09% where the canonical protocol gives 76.81%
# (scripts/audit_feature_stores.py, data/feature_store_audit.json). It was the
# only stale store; clip_features_laion_clean.npz matches the current corpus
# exactly (max |diff| 0.0000) and is kept.
#
# The two stores use different key conventions, so the loader below normalises.
FEAT_CACHE = {
    'dfn_clip': os.path.join(REPO, 'data', 'canonical_clip_features_dfn.npz'),
    'laion2b': os.path.join(REPO, 'data', 'laion_clip_features_clean.npz'),
}


def load_feature_cache(path):
    """Return (Xtr, ytr, Xte, yte) from either npz key convention."""
    z = np.load(path)
    if 'train_feats' in z:                      # legacy convention
        return z['train_feats'], z['train_labels'], z['test_feats'], z['test_labels']
    if 'Xtr' in z:                              # audit_feature_stores convention
        return z['Xtr'], z['ytr'], z['Xte'], z['yte']
    return z['train'], z['train_y'], z['test'], z['test_y']   # canonical convention
TPD_MAP = ROOT + '/data/tpd_to_kg_map.json'
KG_DATA = os.path.join(REPO, 'data/tobacco_records.json')
OUT = os.path.join(REPO, 'data', 'lp_plus_plus.json')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Same 14 templates / CN map / KG construction as zero_shot_clean.py
CN_MAP = {
    "anthracnose":"炭疽病","black_shank":"黑胫病","brown_spot":"赤星病",
    "cmv":"黄瓜花叶病毒病","frog_eye":"蛙眼病","genetic_abnormality":"遗传异常",
    "healthy":"健康","nematodes":"根结线虫病","potato_tuber_moth":"马铃薯块茎蛾",
    "pvy":"马铃薯Y病毒病","sunscald":"日灼病","target_spot":"靶斑病",
    "tmv":"烟草普通花叶病毒病","tswv":"番茄斑萎病毒病","weather_fleck":"气候斑点病",
    "wildfire":"野火病",
}
TEMPLATES = [
    ("EN-Label","{class_name}"),
    ("CN-Label","{cn_name}"),
    ("EN-Photo","a photo of a tobacco leaf with {class_name} disease"),
    ("CN-Photo","一张患有{cn_name}的烟草叶片照片"),
    ("EN-Symptom","tobacco leaf showing {class_name} symptoms"),
    ("CN-Symptom","烟草叶片表现出{cn_name}的症状"),
    ("EN-Closeup","a close-up photo of {class_name} on tobacco plant"),
    ("CN-Closeup","烟草植株上{cn_name}的特写照片"),
    ("EN-Disease","{class_name} disease on tobacco leaf"),
    ("CN-Disease","烟草叶片上的{cn_name}病害"),
    ("KG-Symptom","tobacco leaf: {kg_symptom}"),
    ("KG-Condition","tobacco leaf disease occurring when: {kg_condition}"),
    ("KG-Combined","tobacco leaf with {kg_symptom}, occurs when {kg_condition}"),
    ("EN-Expert","tobacco plant pathology: {class_name} disease identification"),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def tensor_to_pil(img_tensor):
    from PIL import Image
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    return Image.fromarray((np_img * 255).astype(np.uint8))


def extract_clip(model, preprocess, dataset, device):
    feats, labs = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn):
            imgs = batch['image']
            clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).to(device)
            f = model.encode_image(clip_imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    return np.concatenate(feats), np.array(labs)


def build_kg(class_names):
    tpd_map = json.load(open(TPD_MAP)); kg_data = json.load(open(KG_DATA))
    kg_sym, kg_cond = {}, {}
    for name in class_names:
        if name == 'healthy':
            kg_sym[name] = "healthy tobacco leaf without any disease spots or damage"
            kg_cond[name] = "optimal growing conditions with no disease pressure"
            continue
        info = tpd_map.get(name, {})
        kg_ent = info.get('kg_entity'); kg_name = kg_ent.replace('D_', '') if kg_ent else name
        matched = False
        for r in kg_data:
            if kg_name.lower() in r.get('name', '').lower() or name.lower() in r.get('english_name', '').lower():
                kg_sym[name] = r.get('symptoms', '').split('。')[0][:100]
                kg_cond[name] = r.get('occurrence', '').strip()[:100]; matched = True; break
        if not matched:
            cn = CN_MAP.get(name, name)
            kg_sym[name] = f"{cn} disease symptoms on tobacco leaf"
            kg_cond[name] = f"conditions favoring {cn} development"
    return kg_sym, kg_cond


def text_embeddings(model, tok, class_names):
    """Per-class mean of the 14 normalized template embeddings (NOT
    renormalized): reproduces the paper's uniform zero-shot ensemble."""
    kg_sym, kg_cond = build_kg(class_names)
    embs = []
    for name in class_names:
        prompts = []
        for _, tmpl in TEMPLATES:
            s = tmpl.replace('{class_name}', name).replace('{cn_name}', CN_MAP.get(name, name))
            s = s.replace('{kg_symptom}', kg_sym.get(name, '')).replace('{kg_condition}', kg_cond.get(name, ''))
            prompts.append(s)
        with torch.no_grad():
            t = tok(prompts).to(DEVICE)
            tf = model.encode_text(t)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        embs.append(tf.mean(dim=0).cpu().numpy())
    return np.stack(embs).astype(np.float32)   # (16, d)


def disjoint_kshot_idx(labels, K, sup_idx, seed):
    """K-per-class val subset drawn from the pool EXCLUDING the support."""
    sup_set = set(int(i) for i in sup_idx)
    by_class = {}
    for i in range(len(labels)):
        if i in sup_set:
            continue
        by_class.setdefault(int(labels[i]), []).append(i)
    rng = np.random.RandomState(seed)
    out = []
    for c in sorted(by_class):
        out.extend(rng.choice(np.array(by_class[c]), size=min(K, len(by_class[c])), replace=False))
    return np.sort(np.array(out, dtype=int))


def run_lp_pp(Xs, ys, Xv, yv, Xte, yte, T, shots, seed, epochs=300):
    """One LP++ run; faithful to reference_lpp/methods/linear_probe_p2.py."""
    torch.manual_seed(seed)
    Xt = torch.tensor(Xs, device=DEVICE)
    yt = torch.tensor(ys, device=DEVICE)
    Xvt = torch.tensor(Xv, device=DEVICE)
    yvt = torch.tensor(yv, device=DEVICE)
    Xtt = torch.tensor(Xte, device=DEVICE)
    ytt = torch.tensor(yte, device=DEVICE)
    Tt = torch.tensor(T, device=DEVICE)
    n, d = Xs.shape
    n_cls = T.shape[0]

    # classifier initialized to per-class feature SUM (official; no division)
    onehot = torch.zeros(n, n_cls, device=DEVICE)
    onehot[torch.arange(n), yt] = 1.0
    clf = nn.Linear(d, n_cls, bias=True).to(DEVICE)
    clf.weight.data = (onehot.T @ Xt).clone()
    clf.bias.data  # default nn.Linear init kept (official quirk), seeded

    # implicit MM steps
    lr_w = 4.0 * n / float(torch.linalg.eigvalsh(Xt.T @ Xt).max())
    opt = torch.optim.SGD(clf.parameters(), lr_w, momentum=0.9)

    tl = Xt @ Tt.T                                   # (n, 16) text logits
    per_cls = torch.stack([tl[yt == c].mean(0) if (yt == c).any()
                           else torch.zeros(n_cls, device=DEVICE) for c in range(n_cls)])
    # Reference (reference_lpp/methods/linear_probe_p2.py:24-30):
    #     alpha_tilde = compute_centroids_alpha(...)[0] * shots
    #     alpha_init  = 250 / shots * alpha_tilde
    # The *shots and /shots cancel, giving 250 * mean_c(per_class_mean).
    # This port kept only the 250/shots factor, so alpha started `shots` times
    # too small -- identical at K=1, 5x at K=5, 10x at K=10, 80x at full data
    # (shots = len(y)//n_classes = 1295//16 = 80). The blend term therefore began
    # near zero and could only reach its learned value via the every-10-epoch
    # alpha update, which is why the reported LP++ gains were lower bounds.
    alpha0 = 250.0 * float(per_cls.mean())
    alpha = alpha0 * torch.ones(1, n_cls, device=DEVICE)
    alpha.requires_grad_(True)
    lr_alpha = n / (4.0 * float((tl ** 2).sum(0).max()))

    best_val, best_test, best_ep = -1.0, 0.0, -1
    for ep in range(epochs):
        clf.train()
        logits = clf(Xt) + alpha * tl
        loss = F.cross_entropy(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:                       # alpha.grad accumulates (official)
            alpha.data -= lr_alpha * alpha.grad.data
        clf.eval()
        with torch.no_grad():
            av = float((clf(Xvt) + alpha * (Xvt @ Tt.T)).argmax(1).eq(yvt).float().mean())
            at = float((clf(Xtt) + alpha * (Xtt @ Tt.T)).argmax(1).eq(ytt).float().mean())
        if av >= best_val:
            best_val, best_test, best_ep = av, at, ep

    meta = {
        'best_epoch': int(best_ep), 'best_val_acc': round(best_val, 4),
        'alpha0': round(alpha0, 4), 'lr_w': round(lr_w, 6), 'lr_alpha': round(lr_alpha, 6),
        'final_alpha': np.round(alpha.data.detach().cpu().numpy().ravel(), 4).tolist(),
    }
    return float(best_test), meta


def main():
    print(f"Device: {DEVICE}", flush=True)
    train_ds = TPDDDataset(split='train', transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset(split='test', transform=EVAL_TRANSFORM)
    n_classes = len(train_ds.class_names)
    class_names = [str(n) for n in train_ds.class_names]
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {n_classes}", flush=True)

    train_labels_full = np.asarray(train_ds.label_ids)
    idx_sub, idx_val = train_test_split(np.arange(len(train_ds)), test_size=0.2,
                                        random_state=42, stratify=train_labels_full)
    print(f"Val (20%, seed 42, same as canonical): {len(idx_val)}", flush=True)

    R = {
        'protocol': "LP++ (arXiv:2404.02285, official code port): logit = Wx+b+alpha.(xT); W init per-class feature sum, SGD momentum 0.9 lr 4n/lam_max; alpha scalar-init 250*mean (reference's *shots and /shots cancel), per-class grads, updated every 10 epochs lr n/(4 max sum sq); 300 epochs, early stop on val; text side = 14-template uniform ensemble (paper's zero-shot). Full: val=20% stratified seed42 (single run). K-shot: support seed+500 (paired with probe), val disjoint K/class seed+1500, 15 seeds.",
        'n_train': int(len(train_ds)), 'n_test': int(len(test_ds)), 'n_classes': int(n_classes),
        'checkpoints': {},
    }

    for name, ckpt in CKPT.items():
        print(f"\n=== {name} ===", flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ckpt)
        model = model.to(DEVICE).eval()
        tok = open_clip.get_tokenizer('ViT-L-14')

        cache = FEAT_CACHE[name]
        if os.path.exists(cache):
            X, y, Xte, yte = load_feature_cache(cache)
            print(f"features from cache {os.path.basename(cache)}", flush=True)
        else:
            X, y = extract_clip(model, preprocess, train_ds, DEVICE)
            Xte, yte = extract_clip(model, preprocess, test_ds, DEVICE)
            np.savez_compressed(cache, train_feats=X, train_labels=y,
                                test_feats=Xte, test_labels=yte)
            print(f"features extracted + cached: train {X.shape} test {Xte.shape}", flush=True)
        assert len(set(y.tolist())) == 16 and len(yte) == 138
        # The assert above only validates the CACHE labels; the next line then
        # silently replaced them with dataset-order labels, leaving X in cache
        # order. If the two orders ever differ -- or if len(y) < n_train, since
        # labels.json is class-grouped and a prefix would cover only the first
        # few classes -- every downstream use of X/y would be mispaired with no
        # error. Assert they are the SAME array instead of overwriting.
        assert np.array_equal(y, train_labels_full[:len(y)]), (
            "cached feature labels differ from dataset-order labels; X and y "
            "would be mispaired")

        T = text_embeddings(model, tok, class_names)
        zs_acc = float(((Xte @ T.T).argmax(1) == yte).mean())
        print(f"zero-shot (14-template ensemble) {zs_acc*100:.1f}%", flush=True)

        ck_out = {'zero_shot': round(zs_acc, 4), 'text_norms': [round(float(np.linalg.norm(r)), 3) for r in T]}
        for K in [1, 2, 5, 10, -1]:
            seeds = [0] if K == -1 else range(15)
            accs, metas, probe = [], [], []
            for seed in seeds:
                if K == -1:
                    sup = np.arange(len(y)); val = idx_val
                    shots = len(y) // n_classes
                else:
                    sup = kshot_subset_idx(y, K, seed + 500)
                    val = disjoint_kshot_idx(y, K, sup, seed + 1500)
                    shots = K
                acc, meta = run_lp_pp(X[sup], y[sup], X[val], y[val], Xte, yte, T, shots, seed)
                accs.append(round(acc, 4)); metas.append(meta)
                clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
                clf.fit(X[sup], y[sup])
                probe.append(round(float((clf.predict(Xte) == yte).mean()), 4))
            tag = f"K={K}" if K > 0 else "full"
            ck_out[tag] = {
                'lp_pp': {'mean': round(float(np.mean(accs)), 4), 'std': round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0, 'accs': accs},
                'probe_C1': {'mean': round(float(np.mean(probe)), 4), 'std': round(float(np.std(probe, ddof=1)), 4) if len(probe) > 1 else 0.0, 'accs': probe},
                'meta': metas,
            }
            print(f"{tag:5s}: LP++ {np.mean(accs)*100:5.1f}%  |  probe(C=1) {np.mean(probe)*100:5.1f}%", flush=True)
        R['checkpoints'][name] = ck_out
        del model
        torch.cuda.empty_cache()

    json.dump(R, open(OUT, 'w'), indent=2)
    print(f"\nSaved {OUT}", flush=True)


if __name__ == '__main__':
    main()
