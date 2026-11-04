"""
2026-10-25 second-round review fix: recompute the two cited ECE numbers on the
CURRENT canonical feature store, with unambiguous attribution.

What this replaces
------------------
data/calibration_results.json (2026-10-02) was computed on a superseded feature
store and its "zero-shot" ECE used the KG-Symptom template, not the best
template (EN-Symptom, 14.5%) the manuscript cites. The manuscript sentence
"For the best zero-shot template, ECE is 0.14" is therefore misattributed.

This script:
  1. Zero-shot (DFN-CLIP, EN-Symptom = the best template, tau=100): ECE + a
     tau sweep, on the canonical store. Accuracy must reproduce 14.49% (20/138).
  2. Full-data linear probe C=1 (pre-specified default): accuracy must
     reproduce 76.8%; ECE reported.
  3. Full-data linear probe C=100 (validation-selected, canonical
     protocol: refit on 100% train, single test evaluation): accuracy must
     reproduce 92.5%; ECE reported.

Output: data/calibration_20261025.json
"""
import os
import sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ROOT + '/scripts')
import torch
import open_clip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
STORE = 'data/canonical_clip_features_dfn.npz'
OUT = 'data/calibration_20261025.json'

# Canonical template set (prompt_ablation.py); EN-Symptom is the best template
# on TPDD (14.5% = 20/138 in data/zero_shot_clean.json).
CLASS_NAMES = [
    "anthracnose", "black_shank", "brown_spot", "cmv", "frog_eye",
    "genetic_abnormality", "healthy", "nematodes", "potato_tuber_moth",
    "pvy", "sunscald", "target_spot", "tmv", "tswv", "weather_fleck",
    "wildfire",
]
TEMPLATE = "tobacco leaf showing {class_name} symptoms"  # EN-Symptom

def ece(probs, y, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e, bins = 0.0, []
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1]) if i < n_bins - 1 \
            else (conf >= edges[i]) & (conf <= 1.0)
        n = int(m.sum())
        if n == 0:
            bins.append({"bin": i, "range": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                         "acc": 0, "conf": 0, "size": 0, "gap": 0})
            continue
        a, c = float(correct[m].mean()), float(conf[m].mean())
        e += n / len(y) * abs(a - c)
        bins.append({"bin": i, "range": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                     "acc": round(a, 4), "conf": round(c, 4), "size": n,
                     "gap": round(a - c, 4)})
    return round(e, 4), bins

def softmax(z, tau):
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z / tau)
    return p / p.sum(axis=1, keepdims=True)

def main():
    t0 = time.time()
    z = np.load(STORE)
    tr, tr_y = z['train'], z['train_y']
    te, te_y = z['test'], z['test_y']
    assert tr.shape[0] == 1295 and te.shape[0] == 138

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model = model.eval().to(device)
    prompts = [TEMPLATE.format(class_name=n) for n in CLASS_NAMES]
    toks = tokenizer(prompts).to(device)
    with torch.no_grad():
        txt = model.encode_text(toks)
        txt = (txt / txt.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float64)

    # --- 1. zero-shot, best template, tau sweep ---
    logits = te @ txt.T  # cosine similarities, tau=1
    out = {'date': time.strftime('%Y-%m-%d'),
           'store': STORE, 'checkpoint': 'dfn_clip',
           'zero_shot_template': {'name': 'EN-Symptom', 'text': TEMPLATE}}
    zs = {}
    acc_t100 = (softmax(logits, 100).argmax(axis=1) == te_y).mean() * 100
    out['zero_shot'] = {'tau': 100,
                        'accuracy': round(float(acc_t100), 2),
                        'n_correct': int((softmax(logits, 100).argmax(axis=1) == te_y).sum())}
    for tau in [10, 50, 100, 500, 1000]:
        p = softmax(logits, tau)
        e, bins = ece(p, te_y)
        acc = (p.argmax(axis=1) == te_y).mean() * 100
        zs[str(tau)] = {'accuracy': round(float(acc), 2), 'ece': e}
    out['zero_shot']['ece'] = zs['100']['ece']
    out['zero_shot']['tau_sweep'] = zs
    print(f"zero-shot EN-Symptom: acc(tau=100)={acc_t100:.2f}% ece(tau=100)={zs['100']['ece']}")

    # --- 2 & 3. probes ---
    probes = {}
    for C in [1.0, 100.0]:
        clf = LogisticRegression(max_iter=5000, C=C, random_state=42)
        clf.fit(tr, tr_y)
        probs = clf.predict_proba(te)
        classes = clf.classes_
        pad = np.zeros((probs.shape[0], 16))
        pad[:, classes] = probs
        acc = (clf.predict(te) == te_y).mean() * 100
        e, bins = ece(pad, te_y)
        probes[str(C)] = {'accuracy': round(float(acc), 2), 'ece': e, 'bins': bins}
        print(f"probe C={C}: acc={acc:.2f}% ece={e}")
    out['probe'] = probes
    out['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, 'w'), indent=2, ensure_ascii=False)
    print('saved', OUT)

if __name__ == '__main__':
    main()
