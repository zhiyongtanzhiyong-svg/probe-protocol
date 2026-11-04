#!/usr/bin/env python3
"""TLD external zero-shot re-run, CURRENT repo state.
232 base images (augmented variants excluded), DFN-CLIP + LAION-2B, 11 non-KG templates
(same set as data/tld_zero_shot.json). Baseline: 44/232 = 18.97% (majority class)."""
import sys, json, time
from math import comb
import numpy as np, torch
from PIL import Image
sys.path.insert(0, ROOT + '/scripts')
import dataset as D  # noqa
REPO = ROOT
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPTS = {'dfn_clip': 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin',
         'laion2b': 'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin'}
CLASSES = ['Brown spot', 'Frogeye', 'Healthy', 'Mosaic', 'Potato virus Y', 'Target spot',
           'Weather fleck', 'alternaria alternata', 'cercospora nicotianae',
           'no cercospora nicotianae or alternaria alternata present']
CN = {'Brown spot': '赤星病', 'Frogeye': '蛙眼病', 'Healthy': '健康', 'Mosaic': '花叶病',
      'Potato virus Y': '马铃薯Y病毒病', 'Target spot': '靶斑病', 'Weather fleck': '气候斑点病',
      'alternaria alternata': '链格孢', 'cercospora nicotianae': '烟草尾孢',
      'no cercospora nicotianae or alternaria alternata present': '无烟草尾孢或链格孢感染'}
TEMPLATES = [
    ('EN-Label', '{c}'), ('CN-Label', '{cn}'),
    ('EN-Photo', 'a photo of a tobacco leaf with {c}'), ('CN-Photo', '一张患有{cn}的烟草叶片照片'),
    ('EN-Symptom', 'tobacco leaf showing {c} symptoms'), ('CN-Symptom', '烟草叶片表现出{cn}的症状'),
    ('EN-Closeup', 'a close-up photo of {c} on tobacco plant'), ('CN-Closeup', '烟草植株上{cn}的特写照片'),
    ('EN-Disease', '{c} disease on tobacco leaf'), ('CN-Disease', '烟草叶片上的{cn}病害'),
    ('EN-Expert', 'tobacco plant pathology: {c} disease identification'),
]

# collect base images (exclude _aug)
import os
items = []
root = f'{REPO}/data/tld_images'
for cls in CLASSES:
    d = os.path.join(root, 'test', cls)
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if '_aug' in f:
            continue
        items.append((cls, os.path.join(d, f)))
from collections import Counter
cnt = Counter(c for c, _ in items)
n = len(items)
y = np.array([CLASSES.index(c) for c, _ in items])
majority = cnt.most_common(1)
print(f'base images: {n}; majority: {majority}', flush=True)

def tensor_to_pil(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (t * std + mean).clamp(0, 1) * 255
    x = x.round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)

def binom_sf(k, N, p):
    return sum(comb(N, i) * p**i * (1-p)**(N-i) for i in range(k, N+1))

out = {'date': time.strftime('%Y-%m-%d'), 'n_base': n, 'majority': dict(majority),
       'baseline_pct': round(majority[0][1]/n*100, 2), 'checkpoints': {}}
import open_clip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for ck, path in CKPTS.items():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=path)
    model = model.eval().to(DEVICE)
    tok = open_clip.get_tokenizer('ViT-L-14')
    with torch.no_grad():
        feats = []
        for i in range(0, n, 16):
            batch = [Image.open(items[j][1]).convert('RGB') for j in range(i, min(i+16, n))]
            x = torch.stack([preprocess(img) for img in batch]).to(DEVICE)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float64))
            del x, f
    imgf = np.vstack(feats)
    per, best = {}, (None, -1.0)
    for tname, tmpl in TEMPLATES:
        ps = [tmpl.replace('{c}', c).replace('{cn}', CN[c]) for c, _ in items]
        with torch.no_grad():
            t = model.encode_text(tok(ps).to(DEVICE))
            t = t / t.norm(dim=-1, keepdim=True)
            t = t.cpu().numpy().astype(np.float64)
        pred = (imgf @ t.T).argmax(axis=1)
        acc = float((pred == y).mean())
        per[tname] = {'acc_pct': round(acc*100, 2), 'n_correct': int((pred == y).sum())}
        if acc > best[1]:
            best = (tname, acc)
    bname, bacc = best
    kb = int(bacc * n)
    out['checkpoints'][ck] = {
        'templates': per, 'best_template': bname, 'best_acc_pct': round(bacc*100, 2),
        'baseline_pct': round(majority[0][1]/n*100, 2),
        'p_vs_uniform': float(binom_sf(kb, n, 1/10)),
        'p_vs_baseline': float(binom_sf(kb, n, majority[0][1]/n)),
    }
    print(f"{ck}: best={bname} {bacc*100:.2f}%  baseline={majority[0][1]/n*100:.2f}%  "
          f"p_u={out['checkpoints'][ck]['p_vs_uniform']:.2e}  p_b={out['checkpoints'][ck]['p_vs_baseline']:.3f}", flush=True)
    del model
    torch.cuda.empty_cache()
json.dump(out, open(f'{REPO}/data/tld_zs_20261025.json', 'w'), indent=2, ensure_ascii=False)
print('saved data/tld_zs_20261025.json')
