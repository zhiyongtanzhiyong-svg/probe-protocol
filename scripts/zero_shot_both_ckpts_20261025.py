"""
2026-10-25 second-round review fix: zero-shot on TPDD under the EXACT
zero_shot_clean protocol (raw PIL + ImageNet normalization), for BOTH
checkpoints, with:
  - per-template accuracy (DFN must reproduce data/zero_shot_clean.json:
    EN-Symptom 14.49% = 20/138, best-of-14 = EN-Symptom);
  - LAION-2B best template (currently ABSENT from the manuscript -- the
    abstract's 'no template beats the majority class' is stated for the
    DFN checkpoint only; the second-round reviewers flagged this);
  - ECE of each checkpoint's best template at tau=100 + tau sweep
    (replaces the misattributed 0.14 in data/calibration_results.json,
    which was computed on a superseded store with the KG-Symptom template);
  - one-sided exact binomial p for best-vs-uniform (1/16), as requested
    in the second-round review (C.1).

Output: data/zero_shot_both_ckpts_20261025.json
"""
import os
import sys, json, time
from math import comb
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, ROOT + '/scripts')
from dataset import TPDDDataset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPTS = {
    "dfn_clip": "checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin",
    "laion2b": "checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin",
}
OUT = 'data/zero_shot_both_ckpts_20261025.json'

CN_MAP = {
    "anthracnose": "炭疽病", "black_shank": "黑胫病", "brown_spot": "赤星病",
    "cmv": "黄瓜花叶病毒病", "frog_eye": "蛙眼病", "genetic_abnormality": "遗传异常",
    "healthy": "健康", "nematodes": "根结线虫病", "potato_tuber_moth": "马铃薯块茎蛾",
    "pvy": "马铃薯Y病毒病", "sunscald": "日灼病", "target_spot": "靶斑病",
    "tmv": "烟草普通花叶病毒病", "tswv": "番茄斑萎病毒病", "weather_fleck": "气候斑点病",
    "wildfire": "野火病",
}


def build_kg(class_names):
    tpd_map = json.load(open(ROOT + '/data/tpd_to_kg_map.json'))
    kg_data = json.load(open(ROOT + '/data/tobacco_records.json'))
    kg_sym, kg_cond = {}, {}
    for name in class_names:
        if name == 'healthy':
            kg_sym[name] = "healthy tobacco leaf without any disease spots or damage"
            kg_cond[name] = "normal growing conditions"
            continue
        info = tpd_map.get(name, {})
        kg_ent = info.get('kg_entity')
        if kg_ent:
            kg_name = kg_ent.replace('D_', '')
            for r in kg_data:
                if kg_name.lower() in r.get('name', '').lower() or name.lower() in r.get('english_name', '').lower():
                    kg_sym[name] = r.get('symptoms', '').split('。')[0][:100]
                    kg_cond[name] = r.get('condition', '').split('。')[0][:100]
                    break
        if name not in kg_sym:
            cn = CN_MAP.get(name, name)
            kg_sym[name] = f"{cn} disease symptoms on tobacco leaf"
            kg_cond[name] = f"when the plant is affected by {cn}"
    return kg_sym, kg_cond


def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = TPDDDataset('test')
    class_names = ds.class_names
    n = len(ds)
    y = np.array([ds.name_to_idx[lab] for lab in ds.class_labels])
    assert n == 138
    kg_sym, kg_cond = build_kg(class_names)

    templates = [
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

    out = {'date': time.strftime('%Y-%m-%d'), 'n_test': n,
           'protocol': 'raw PIL -> Resize(256)+CenterCrop(224)+ToTensor+ImageNet-normalize '
                       '(identical to scripts/zero_shot_clean.py); argmax cosine, tau irrelevant for argmax',
           'majority_baseline': round(float(np.bincount(y).max() / n) * 100, 3),
           'checkpoints': {}}

    for ckpt, path in CKPTS.items():
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=path)
        model = model.eval().to(device)
        tok = open_clip.get_tokenizer('ViT-L-14')
        tf = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad():
            imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in ds.images]).to(device)
            imgf = model.encode_image(imgs)
            imgf = imgf / imgf.norm(dim=-1, keepdim=True)
            imgf = imgf.cpu().numpy().astype(np.float64)

        res = {'templates': {}}
        best = (None, -1)
        for tname, tmpl in templates:
            prompts = []
            for name in class_names:
                s = tmpl
                s = s.replace('{class_name}', name).replace('{cn_name}', CN_MAP.get(name, name))
                s = s.replace('{kg_symptom}', kg_sym.get(name, '')).replace('{kg_condition}', kg_cond.get(name, ''))
                prompts.append(s)
            with torch.no_grad():
                txt = model.encode_text(tok(prompts).to(device))
                txt = txt / txt.norm(dim=-1, keepdim=True)
                txt = txt.cpu().numpy().astype(np.float64)
            sims = imgf @ txt.T
            pred = sims.argmax(axis=1)
            acc = (pred == y).mean()
            res['templates'][tname] = {'acc': round(float(acc) * 100, 2),
                                       'n_correct': int((pred == y).sum()),
                                       'preds': pred.tolist()}
            if acc > best[1]:
                best = (tname, float(acc))
        # ECE of best template at tau=100 + sweep
        tname = best[0]
        prompts = [tmpl for (_, tmpl) in templates if _ == tname][0]
        ps = []
        for name in class_names:
            s = prompts.replace('{class_name}', name).replace('{cn_name}', CN_MAP.get(name, name))
            s = s.replace('{kg_symptom}', kg_sym.get(name, '')).replace('{kg_condition}', kg_cond.get(name, ''))
            ps.append(s)
        with torch.no_grad():
            txt = model.encode_text(tok(ps).to(device))
            txt = txt / txt.norm(dim=-1, keepdim=True)
            txt = txt.cpu().numpy().astype(np.float64)
        sims = imgf @ txt.T

        def ece(probs, n_bins=10):
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

        sweep = {}
        for tau in [10, 50, 100, 500, 1000]:
            z = sims / tau - (sims / tau).max(axis=1, keepdims=True)
            p = np.exp(z)
            p = p / p.sum(axis=1, keepdims=True)
            sweep[str(tau)] = {'ece': ece(p),
                               'mean_conf': round(float(p.max(axis=1).mean()), 4)}
        res['best_template'] = best[0]
        res['best_acc'] = round(best[1] * 100, 2)
        res['ece_best_template'] = sweep
        # binomial: best vs uniform
        k = int((np.array(res['templates'][best[0]]['preds']) == y).sum())
        p_uniform = sum(comb(n, i) * (1 / 16) ** i * (15 / 16) ** (n - i) for i in range(k, n + 1))
        res['binomial_vs_uniform'] = {'n_correct': k, 'p_one_sided': float(p_uniform)}
        out['checkpoints'][ckpt] = res
        del model
        torch.cuda.empty_cache()
        print(f"{ckpt}: best={best[0]} {best[1]*100:.2f}%  ece(tau100)={sweep['100']['ece']}  "
              f"binom_vs_uniform p={p_uniform:.2e}")

    out['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, 'w'), indent=2, ensure_ascii=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
