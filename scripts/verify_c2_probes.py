"""C-task verification probes (review items C1/C2 + ECE protocol check).

1. Canonical-protocol ECE: fit C=1 and validation-selected C probes on the
   canonical npz features (20% stratified val, seed 42, refit on 100%),
   then 10-bin ECE on the 138 test images.
2. EN-Label / EN-Photo zero-shot: prediction distribution on the 138 test
   images + 10-bin ECE (softmax of 100*cosine), to check for degenerate
   collapse (e.g., always-predict-wildfire) behind EN-Label 0.7%.
3. Sanity: reproduce C-selection (expect best_C=300, test 85.5%).
"""
import os
import json, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, ROOT + '/scripts')

CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
NPZ = ROOT + '/data/clip_features_clean.npz'
TPD_MAP = ROOT + '/data/tpd_to_kg_map.json'
KG_DATA = ROOT + '/data/tobacco_records.json'

CLASS_NAMES = list(json.load(open(TPD_MAP)).keys())  # label ids 0..15
CN_MAP = {
    "anthracnose":"炭疽病","black_shank":"黑胫病","brown_spot":"赤星病",
    "cmv":"黄瓜花叶病毒病","frog_eye":"蛙眼病","genetic_abnormality":"遗传异常",
    "healthy":"健康","nematodes":"根结线虫病","potato_tuber_moth":"马铃薯块茎蛾",
    "pvy":"马铃薯Y病毒病","sunscald":"日灼病","target_spot":"靶斑病",
    "tmv":"烟草普通花叶病毒病","tswv":"番茄斑萎病毒病","weather_fleck":"气候斑点病",
    "wildfire":"野火病",
}

def ece10(probs, labels):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    acc = (pred == labels)
    bounds = np.linspace(0, 1, 11); ece = 0.0; bins = []
    for i in range(10):
        m = (conf >= bounds[i]) & (conf < bounds[i+1]) if i < 9 else (conf >= bounds[i]) & (conf <= bounds[i+1])
        if m.sum() > 0:
            ece += abs(conf[m].mean() - acc[m].mean()) * (m.sum() / len(conf))
            bins.append([round(float(conf[m].mean()),3), round(float(acc[m].mean()),3), int(m.sum())])
        else:
            bins.append([0,0,0])
    return round(float(ece),4), bins

z = np.load(NPZ)
Xtr, ytr, Xte, yte = z['train_feats'], z['train_labels'], z['test_feats'], z['test_labels']
print('features:', Xtr.shape, Xte.shape, 'classes:', len(np.unique(ytr)))

# ---- 1. canonical probe ECE ----
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=42)
grid = [0.01,0.1,1.0,3.0,10.0,30.0,100.0,300.0,1000.0,3000.0]
val = {C: (LogisticRegression(C=C, max_iter=2000, random_state=42).fit(Xa, ya).score(Xv, yv)) for C in grid}
best_C = max(grid, key=lambda C: val[C])
probes = {}
for C in [1.0, best_C]:
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)
    e, b = ece10(p, yte)
    correct = clf.predict(Xte) == yte
    probes[f'C={C}'] = {
        'test_acc': round(float((clf.predict(Xte)==yte).mean()),4),
        'ece10': e, 'bins_conf_acc_n': b,
        'mean_conf': round(float(p.max(1).mean()),3),
        'correct_conf': round(float(p.max(1)[correct].mean()),3),
        'incorrect_conf': round(float(p.max(1)[~correct].mean()),3),
    }
print('best_C:', best_C, 'val@C:', {k: round(v,3) for k,v in val.items() if k in (1.0,30.0,300.0,1000.0,3000.0)})
print('probes:', json.dumps(probes, indent=1))

# ---- 2. zero-shot EN-Label / EN-Photo distributions + ECE ----
import open_clip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT)
model.eval().cuda(); tok = open_clip.get_tokenizer('ViT-L-14')

tpd = json.load(open(TPD_MAP)); kg = json.load(open(KG_DATA))
kg_sym = {}
for name, info in tpd.items():
    if name == 'healthy':
        kg_sym[name] = "healthy tobacco leaf without any disease spots or damage"; continue
    kg_ent = info.get('kg_entity'); kg_name = kg_ent.replace('D_','') if kg_ent else name
    for r in kg:
        if kg_name.lower() in r.get('name','').lower() or name.lower() in r.get('english_name','').lower():
            kg_sym[name] = r.get('symptoms','').split('。')[0][:100]; break
    kg_sym.setdefault(name, f"{CN_MAP[name]} disease symptoms on tobacco leaf")

TEMPLATES = {
    'EN-Label': [n for n in CLASS_NAMES],
    'EN-Photo': [f"a photo of a tobacco leaf with {n} disease" for n in CLASS_NAMES],
    'KG-Symptom': [f"tobacco leaf: {kg_sym[n]}" for n in CLASS_NAMES],
}
Xte_t = torch.from_numpy(Xte).cuda()
out = {}
for tname, prompts in TEMPLATES.items():
    with torch.no_grad():
        tf = model.encode_text(tok(prompts).cuda())
        tf = tf / tf.norm(dim=-1, keepdim=True)
        sim = 100.0 * (tf @ Xte_t.T)          # [16, 138]
    sim_n = sim.T.cpu().numpy()            # [138, 16]
    pred = sim_n.argmax(axis=1)
    correct = pred == yte
    exp = np.exp(sim_n - sim_n.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    e, b = ece10(probs, yte)
    dist = {CLASS_NAMES[i]: int((pred==i).sum()) for i in range(16)}
    out[tname] = {
        'acc': round(float(correct.mean()),4), 'n_correct': int(correct.sum()),
        'ece10': e, 'bins_conf_acc_n': b,
        'pred_dist': dist,
        'top_pred': sorted(dist.items(), key=lambda kv:-kv[1])[:5],
        'mean_conf': round(float(probs.max(axis=1).mean()),3),
    }
    print(tname, out[tname]['acc'], 'top:', out[tname]['top_pred'], 'ece:', e)

json.dump({'class_names': CLASS_NAMES, 'best_C': best_C, 'val_grid': val,
           'canonical_probes': probes, 'zero_shot': out},
          open(ROOT + '/data/verification_c2.json','w'), indent=2)
print('saved data/verification_c2.json')
