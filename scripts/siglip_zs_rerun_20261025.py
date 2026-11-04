#!/usr/bin/env python3
"""SigLIP zero-shot re-run on CURRENT repo state (CPU, to coexist with K12 GPU runs).
Verifies data/siglip_protocol.json zero-shot (best CN-Symptom 13.04%, 2026-10-22)."""
import os
import sys, json, time
import numpy as np, torch
from PIL import Image
from torchvision import transforms
sys.path.insert(0, ROOT + '/scripts')
import dataset as D
TPDDDataset = D.TPDDDataset
from math import comb
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO = ROOT
DEVICE = 'cpu'
ds = TPDDDataset('test')
class_names = ds.class_names
n = len(ds)
y = np.array([ds.name_to_idx[l] for l in ds.class_labels])
sT = [
    ("EN-Symptom", "a photo of a leaf with {label} symptoms"),
    ("KG-Symptom", "a photo of a leaf showing {label} disease symptoms on tobacco"),
    ("EN-Photo", "a photo of a {label} affected tobacco leaf"),
    ("KG-Combined", "a photo of a tobacco leaf with {label} symptoms and conditions"),
    ("KG-Condition", "a photo of a leaf in the condition of {label}"),
    ("CN-Symptom", "一张患有{label}症状的叶片照片"),
    ("EN-Closeup", "a closeup photo of a leaf with {label}"),
    ("CN-Photo", "一张{label}烟草叶片的照片"),
    ("CN-Label", "{label}"),
    ("CN-Disease", "一种叫做{label}的疾病"),
    ("CN-Closeup", "一张{label}的特写照片"),
    ("EN-Disease", "a photo of a {label} disease"),
    ("EN-Expert", "an expert photo of a leaf diagnosed with {label}"),
    ("EN-Label", "{label}"),
]
SP = f'{REPO}/../checkpoints/siglip'
t0 = time.time()
model = AutoModel.from_pretrained(SP).to(DEVICE).eval()
stok = AutoTokenizer.from_pretrained(SP)
tfs = []
for _, tmpl in sT:
    texts = [tmpl.format(label=c) for c in class_names]
    inputs = stok(texts, padding=True, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        tf = model.get_text_features(**inputs)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    tfs.append(tf.cpu().numpy().astype(np.float64))
imgproc = transforms.Compose([transforms.Resize(384), transforms.CenterCrop(384),
                              transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
feats = []
with torch.no_grad():
    for i in range(0, n, 8):
        batch = [Image.open(ds.images[j]).convert('RGB') for j in range(i, min(i+8, n))]
        x = torch.stack([imgproc(b) for b in batch]).to(DEVICE)
        f = model.get_image_features(pixel_values=x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy().astype(np.float64))
imgf = np.vstack(feats)
per, best = {}, (None, -1.0, None)
for ti, (tname, _) in enumerate(sT):
    sim = imgf @ tfs[ti].T
    pred = sim.argmax(axis=1)
    acc = float((pred == y).mean())
    per[tname] = {'acc_pct': round(acc*100, 2), 'n_correct': int((pred == y).sum())}
    if acc > best[1]:
        best = (tname, acc, pred)
kb = int((best[2] == y).sum())
def binom_sf(k, N, p):
    return sum(comb(N, i) * p**i * (1-p)**(N-i) for i in range(k, N+1))
out = {'date': time.strftime('%Y-%m-%d'), 'device': DEVICE, 'n_test': n,
       'majority_baseline_pct': round(float(np.bincount(y).max()/n)*100, 2),
       'templates': per, 'best_template': best[0], 'best_acc_pct': round(best[1]*100, 2),
       'binomial_vs_uniform_best': {'n_correct': kb, 'p_one_sided': float(binom_sf(kb, n, 1/16))},
       'protocol': 'scripts/siglip_protocol.py templates; 0.5/0.5 norm; 384px; CPU',
       'elapsed_s': round(time.time()-t0, 1)}
json.dump(out, open(f'{REPO}/data/siglip_zs_20261025.json', 'w'), indent=2, ensure_ascii=False)
print(f"siglip: best={best[0]} {best[1]*100:.2f}%  binom_p={float(binom_sf(kb,n,1/16)):.2e}")
print('saved data/siglip_zs_20261025.json')
