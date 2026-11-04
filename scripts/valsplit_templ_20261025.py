#!/usr/bin/env python3
"""DFN-CLIP: select best of 14 templates on a 20% validation split of TRAIN images,
report that template's TEST accuracy. Guard against test-set-driven prompt selection.
Seed=0, same 14 templates as scripts/zero_shot_clean.py, canonical path."""
import os
import sys, json, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, ROOT + '/scripts')
import dataset as D
TPDDDataset = D.TPDDDataset
REPO = ROOT
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'

ds_tr = TPDDDataset('train')
ds_te = TPDDDataset('test')
class_names = ds_te.class_names
rng = np.random.default_rng(0)
idx = rng.permutation(len(ds_tr))[:max(1, int(0.2 * len(ds_tr)))]
val_imgs = [ds_tr[i]['image'] for i in idx]
val_y = np.array([ds_tr.name_to_idx[ds_tr.class_labels[i]] for i in idx])
te_imgs = [ds_te[j]['image'] for j in range(len(ds_te))]
te_y = np.array([ds_te.name_to_idx[l] for l in ds_te.class_labels])
n_v, n_t = len(val_imgs), len(te_y)

CN = json.load(open('/dev/stdin')) if False else {
 "anthracnose": "炭疽病", "black_shank": "黑胫病", "brown_spot": "赤星病", "cmv": "黄瓜花叶病毒病",
 "frog_eye": "蛙眼病", "genetic_abnormality": "遗传异常", "healthy": "健康", "nematodes": "根结线虫病",
 "potato_tuber_moth": "马铃薯块茎蛾", "pvy": "马铃薯Y病毒病", "sunscald": "日灼病", "target_spot": "靶斑病",
 "tmv": "烟草普通花叶病毒病", "tswv": "番茄斑萎病毒病", "weather_fleck": "气候斑点病", "wildfire": "野火病"}
tpd_map = json.load(open(f'{REPO}/data/tpd_to_kg_map.json'))
kg_data = json.load(open(f'{REPO}/data/tobacco_records.json'))
kg_sym, kg_cond = {}, {}
for name, info in tpd_map.items():
    if name == 'healthy':
        kg_sym[name] = "healthy tobacco leaf without any disease spots or damage"
        kg_cond[name] = "optimal growing conditions with no disease pressure"
        continue
    kg_ent = info.get('kg_entity'); kg_name = kg_ent.replace('D_', '') if kg_ent else name
    m = False
    for r in kg_data:
        if kg_name.lower() in r.get('name', '').lower() or name.lower() in r.get('english_name', '').lower():
            kg_sym[name] = r.get('symptoms', '').split('。')[0][:100]
            kg_cond[name] = r.get('occurrence', '').strip()[:100]; m = True; break
    if not m:
        cn = CN.get(name, name)
        kg_sym[name] = f"{cn} disease symptoms on tobacco leaf"
        kg_cond[name] = f"conditions favoring {cn} development"
TEMPLATES = [
    ("EN-Label", "{class_name}"), ("CN-Label", "{cn_name}"),
    ("EN-Photo", "a photo of a tobacco leaf with {class_name} disease"), ("CN-Photo", "一张患有{cn_name}的烟草叶片照片"),
    ("EN-Symptom", "tobacco leaf showing {class_name} symptoms"), ("CN-Symptom", "烟草叶片表现出{cn_name}的症状"),
    ("EN-Closeup", "a close-up photo of {class_name} on tobacco plant"), ("CN-Closeup", "烟草植株上{cn_name}的特写照片"),
    ("EN-Disease", "{class_name} disease on tobacco leaf"), ("CN-Disease", "烟草叶片上的{cn_name}病害"),
    ("KG-Symptom", "tobacco leaf: {kg_symptom}"),
    ("KG-Condition", "tobacco leaf disease occurring when: {kg_condition}"),
    ("KG-Combined", "tobacco leaf with {kg_symptom}, occurs when {kg_condition}"),
    ("EN-Expert", "tobacco plant pathology: {class_name} disease identification"),
]

import open_clip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT)
model = model.eval().to(DEVICE)
tok = open_clip.get_tokenizer('ViT-L-14')

def tensor_to_pil(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (t * std + mean).clamp(0, 1)
    x = (x * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)

def encode(imgs):
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), 16):
            b = [tensor_to_pil(im) for im in imgs[i:i+16]]
            x = torch.stack([preprocess(im) for im in b]).to(DEVICE)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float64))
            del x, f
    return np.vstack(feats)

fv, ft = encode(val_imgs), encode(te_imgs)
res = {'n_val': n_v, 'n_test': n_t, 'val_seed': 0, 'templates': {}}
best_v = (None, -1.0)
for tname, tmpl in TEMPLATES:
    ps = []
    for name in class_names:
        s = tmpl.replace('{class_name}', name).replace('{cn_name}', CN.get(name, name))
        s = s.replace('{kg_symptom}', kg_sym.get(name, '')).replace('{kg_condition}', kg_cond.get(name, ''))
        ps.append(s)
    with torch.no_grad():
        t = model.encode_text(tok(ps).to(DEVICE))
        t = t / t.norm(dim=-1, keepdim=True)
        t = t.cpu().numpy().astype(np.float64)
    pv = (fv @ t.T).argmax(axis=1)
    pt = (ft @ t.T).argmax(axis=1)
    av = float((pv == val_y).mean())
    at = float((pt == te_y).mean())
    res['templates'][tname] = {'val_acc_pct': round(av * 100, 2), 'test_acc_pct': round(at * 100, 2),
                               'val_n_correct': int((pv == val_y).sum()), 'test_n_correct': int((pt == te_y).sum())}
    if av > best_v[1]:
        best_v = (tname, av)
res['val_best_template'] = best_v[0]
res['val_best_val_acc_pct'] = round(best_v[1] * 100, 2)
tb = res['templates'][best_v[0]]
res['val_best_test_acc_pct'] = tb['test_acc_pct']
res['test_majority_baseline_pct'] = round(float(np.bincount(te_y).max() / n_t) * 100, 2)
res['date'] = time.strftime('%Y-%m-%d')
json.dump(res, open(f'{REPO}/data/valsplit_template_20261025.json', 'w'), indent=2, ensure_ascii=False)
print(f"val-best template: {best_v[0]} (val {best_v[1]*100:.2f}%) -> test {tb['test_acc_pct']}% (baseline {res['test_majority_baseline_pct']}%)")
print('saved data/valsplit_template_20261025.json')
