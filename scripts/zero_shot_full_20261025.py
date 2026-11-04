#!/usr/bin/env python3
"""
2026-10-25 second-round review fix -- full zero-shot recompute, CURRENT repo state.

Why: the manuscript's zero-shot table (data/zero_shot_clean.json, 2026-09-28) and the
BioCLIP-2 control (data/domain_vlm_zeroshot.json) predate the final environment; they
cannot be reproduced from the current images + checkpoints + open_clip 2.32.0 (e.g.
DFN EN-Symptom was 14.5% then, 6.5% now; the LAION-2B zero-shot table is missing
entirely). This script recomputes, from the current repo state, a single consistent
zero-shot result set:

  * DFN-CLIP, LAION-2B, BioCLIP-2 (all ViT-L/14, open_clip) under ONE protocol:
    canonical image path (EVAL tensor -> tensor_to_pil uint8 -> open_clip preprocess),
    the 14-template set of scripts/zero_shot_clean.py, argmax cosine.
  * SigLIP (So400M-patch14-384, transformers) under its own protocol
    (scripts/siglip_protocol.py TEMPLATES) -- verification of the 13.0% figure.
  * For each model: per-template acc, best template, 14-template uniform ensemble,
    ECE of the best template at tau=100 (+ sweep), one-sided exact binomial p of the
    best template vs. uniform chance (1/16), and comparison to the 13.04%
    majority-class baseline.
  * DFN-CLIP image-text alignment stats (align_a6 protocol, canonical feature store,
    base template "a photo of a tobacco leaf with {name} disease"):
    top-1/top-3/top-10 share, mean rank, mean correct-vs-best-incorrect cosine, margin.

Output: data/zero_shot_full_20261025.json
"""
import os
import sys, json, time
from math import comb
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, ROOT + '/scripts')
import dataset as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPDDDataset = D.TPDDDataset

REPO = ROOT
OUT = f'{REPO}/data/zero_shot_full_20261025.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPTS = {
    'dfn_clip': f'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin',
    'laion2b': f'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin',
    'bioclip2': f'checkpoints/bioclip2/open_clip_pytorch_model.bin',
}

CN = {"anthracnose": "炭疽病", "black_shank": "黑胫病", "brown_spot": "赤星病", "cmv": "黄瓜花叶病毒病",
      "frog_eye": "蛙眼病", "genetic_abnormality": "遗传异常", "healthy": "健康", "nematodes": "根结线虫病",
      "potato_tuber_moth": "马铃薯块茎蛾", "pvy": "马铃薯Y病毒病", "sunscald": "日灼病", "target_spot": "靶斑病",
      "tmv": "烟草普通花叶病毒病", "tswv": "番茄斑萎病毒病", "weather_fleck": "气候斑点病", "wildfire": "野火病"}

TEMPLATES = [
    ("EN-Label", "{class_name}"),
    ("CN-Label", "{cn_name}"),
    ("EN-Photo", "a photo of a tobacco leaf with {class_name} disease"),
    ("CN-Photo", "一张患有{cn_name}的烟草叶片照片"),
    ("EN-Symptom", "tobacco leaf showing {class_name} symptoms"),
    ("CN-Symptom", "烟草叶片表现出{cn_name}的症状"),
    ("EN-Closeup", "a close-up photo of {class_name} on tobacco plant"),
    ("CN-Closeup", "烟草植株上{class_name}的特写照片"),
    ("EN-Disease", "{class_name} disease on tobacco leaf"),
    ("CN-Disease", "烟草叶片上的{class_name}病害"),
    ("KG-Symptom", "tobacco leaf: {kg_symptom}"),
    ("KG-Condition", "tobacco leaf disease occurring when: {kg_condition}"),
    ("KG-Combined", "tobacco leaf with {kg_symptom}, occurs when {kg_condition}"),
    ("EN-Expert", "tobacco plant pathology: {class_name} disease identification"),
]


def tensor_to_pil(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = t * std + mean
    x = (x.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def build_kg(class_names):
    tpd_map = json.load(open(f'{REPO}/data/tpd_to_kg_map.json'))
    kg_data = json.load(open(f'{REPO}/data/tobacco_records.json'))
    kg_sym, kg_cond = {}, {}
    for name, info in tpd_map.items():
        if name == 'healthy':
            kg_sym[name] = "healthy tobacco leaf without any disease spots or damage"
            kg_cond[name] = "optimal growing conditions with no disease pressure"
            continue
        kg_ent = info.get('kg_entity')
        kg_name = kg_ent.replace('D_', '') if kg_ent else name
        m = False
        for r in kg_data:
            if kg_name.lower() in r.get('name', '').lower() or name.lower() in r.get('english_name', '').lower():
                kg_sym[name] = r.get('symptoms', '').split('。')[0][:100]
                kg_cond[name] = r.get('occurrence', '').strip()[:100]
                m = True
                break
        if not m:
            cn = CN.get(name, name)
            kg_sym[name] = f"{cn} disease symptoms on tobacco leaf"
            kg_cond[name] = f"conditions favoring {cn} development"
    return kg_sym, kg_cond


def ece(probs, y, n_bins=10):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1]) if i < n_bins - 1 else (conf >= edges[i])
        c = int(m.sum())
        if c == 0:
            continue
        e += c / len(y) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return round(e, 4)


def binom_sf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    t0 = time.time()
    ds = TPDDDataset('test')
    class_names = ds.class_names
    n = len(ds)
    y = np.array([ds.name_to_idx[l] for l in ds.class_labels])
    assert n == 138
    kg_sym, kg_cond = build_kg(class_names)
    baseline = round(float(np.bincount(y).max() / n) * 100, 2)

    out = {
        'date': time.strftime('%Y-%m-%d'),
        'n_test': n,
        'majority_baseline_pct': baseline,
        'protocol_openclip': 'canonical image path: EVAL_TRANSFORM (ImageNet norm) -> tensor_to_pil uint8 '
                             '-> open_clip preprocess (CLIP norm); identical to the frozen-probe feature store; '
                             '14 templates of scripts/zero_shot_clean.py; argmax cosine',
        'checkpoints': {},
    }

    # ---------------- open_clip ViT-L/14 models ----------------
    import open_clip
    for ck, path in CKPTS.items():
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=path)
        model = model.eval().to(DEVICE)
        tok = open_clip.get_tokenizer('ViT-L-14')
        with torch.no_grad():
            feats = []
            for i in range(0, n, 16):
                batch = [ds[j]['image'] for j in range(i, min(i + 16, n))]
                clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in batch]).to(DEVICE)
                f = model.encode_image(clip_imgs)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy().astype(np.float64))
            imgf = np.vstack(feats)
        sims = []
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
            sims.append(t @ imgf.T)
        S = np.stack(sims)  # [14, 16, N]
        per = {}
        best = (None, -1.0, None)
        for ti, (tname, _) in enumerate(TEMPLATES):
            pred = S[ti].argmax(axis=0)
            acc = float((pred == y).mean())
            per[tname] = {'acc_pct': round(acc * 100, 2), 'n_correct': int((pred == y).sum()), 'preds': pred.tolist()}
            if acc > best[1]:
                best = (tname, acc, pred)
        bname, bacc, bpred = best
        ens = S.mean(axis=0).argmax(axis=0)
        kb = int((bpred == y).sum())
        # ECE of best template
        tb = S[np.argmax([t == bname for t, _ in TEMPLATES])]
        sweep = {}
        for tau in [10, 50, 100, 500, 1000]:
            z = tb / tau - (tb / tau).max(axis=0, keepdims=True)
            p = np.exp(z)
            p = p / p.sum(axis=0, keepdims=True)
            sweep[str(tau)] = {'ece': ece(p.T, y), 'mean_conf': round(float(p.max(axis=0).mean()), 4)}
        out['checkpoints'][ck] = {
            'templates': per,
            'best_template': bname,
            'best_acc_pct': round(bacc * 100, 2),
            'ensemble14_acc_pct': round(float((ens == y).mean()) * 100, 2),
            'binomial_vs_uniform_best': {'n_correct': kb, 'p_one_sided': float(binom_sf(kb, n, 1 / 16))},
            'ece_best_template': sweep,
            'gap_vs_baseline_pp': round((bacc - baseline / 100) * 100, 2),
        }
        print(f"{ck}: best={bname} {bacc*100:.2f}%  ens={float((ens==y).mean())*100:.2f}%  "
              f"binom_p={float(binom_sf(kb, n, 1/16)):.2e}  ece(tau100)={sweep['100']['ece']}", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---------------- SigLIP (transformers, own protocol) ----------------
    try:
        from transformers import AutoModel, AutoTokenizer
        SP = f'checkpoints/siglip'
        smodel = AutoModel.from_pretrained(SP).to(DEVICE)
        stok = AutoTokenizer.from_pretrained(SP)
        smodel.eval()
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
        # class names for SigLIP templates: raw names (as in siglip_protocol.py)
        tfs = []
        for _, tmpl in sT:
            texts = [tmpl.format(label=c) for c in class_names]
            inputs = stok(texts, padding=True, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                tf = smodel.get_text_features(**inputs)
                tf = tf / tf.norm(dim=-1, keepdim=True)
            tfs.append(tf.cpu().numpy().astype(np.float64))
        del inputs
        imgproc = transforms.Compose([transforms.Resize(384), transforms.CenterCrop(384),
                                      transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        feats = []
        with torch.no_grad():
            for i in range(0, n, 4):
                batch = [Image.open(ds.images[j]).convert('RGB') for j in range(i, min(i + 4, n))]
                x = torch.stack([imgproc(b) for b in batch]).to(DEVICE)
                f = smodel.get_image_features(pixel_values=x)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy().astype(np.float64))
                del x, f
        imgf = np.vstack(feats)
        per = {}
        best = (None, -1.0, None)
        for ti, (tname, _) in enumerate(sT):
            sim = imgf @ tfs[ti].T
            pred = sim.argmax(axis=1)
            acc = float((pred == y).mean())
            per[tname] = {'acc_pct': round(acc * 100, 2), 'n_correct': int((pred == y).sum())}
            if acc > best[1]:
                best = (tname, acc, pred)
        kb = int((best[2] == y).sum())
        out['checkpoints']['siglip'] = {
            'templates': per,
            'best_template': best[0],
            'best_acc_pct': round(best[1] * 100, 2),
            'binomial_vs_uniform_best': {'n_correct': kb, 'p_one_sided': float(binom_sf(kb, n, 1 / 16))},
            'protocol': 'own templates from scripts/siglip_protocol.py; 0.5/0.5 norm; 384px',
        }
        print(f"siglip: best={best[0]} {best[1]*100:.2f}%", flush=True)
        del smodel
        torch.cuda.empty_cache()
    except Exception as e:  # keep the rest if SigLIP is unavailable
        out['checkpoints']['siglip_error'] = repr(e)
        print('siglip FAILED:', repr(e), flush=True)

    # ---------------- DFN alignment stats (canonical store) ----------------
    try:
        z = np.load(f'{REPO}/data/canonical_clip_features_dfn.npz')
        imgf = z['test'].astype(np.float64)
        import open_clip as oc
        model, _, _ = oc.create_model_and_transforms('ViT-L-14',
                                                     pretrained=CKPTS['dfn_clip'])
        model = model.eval().to(DEVICE)
        tok = oc.get_tokenizer('ViT-L-14')
        base = 'a photo of a tobacco leaf with {name} disease'
        ps = [base.format(name=c) for c in class_names]
        with torch.no_grad():
            t = model.encode_text(tok(ps).to(DEVICE))
            t = t / t.norm(dim=-1, keepdim=True)
            t = t.cpu().numpy().astype(np.float64)
        S = imgf @ t.T  # [N, 16]
        n2 = len(y)
        correct = S[np.arange(n2), y]
        S2 = S.copy()
        S2[np.arange(n2), y] = -1e9
        maxinc = S2.max(axis=1)
        order = np.argsort(-S, axis=1)
        rank = np.array([list(order[i]).index(int(y[i])) + 1 for i in range(n2)])
        out['dfn_alignment'] = {
            'base_template': base,
            'correct_is_top1_pct': round(float((rank == 1).mean()) * 100, 1),
            'in_top3_pct': round(float((rank <= 3).mean()) * 100, 1),
            'in_top10_pct': round(float((rank <= 10).mean()) * 100, 1),
            'mean_rank': round(float(rank.mean()), 2),
            'mean_cos_correct': round(float(correct.mean()), 4),
            'mean_cos_best_incorrect': round(float(maxinc.mean()), 4),
            'mean_margin': round(float((correct - maxinc).mean()), 4),
        }
        print("dfn_alignment:", json.dumps(out['dfn_alignment']), flush=True)
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        out['dfn_alignment_error'] = repr(e)
        print('alignment FAILED:', repr(e), flush=True)

    out['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, 'w'), indent=2, ensure_ascii=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
