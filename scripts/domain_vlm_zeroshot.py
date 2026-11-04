"""
P1: domain-VLM (BioCLIP 2) zero-shot on TPDD + TLD, using the SAME 14-template
harness as scripts/prompt_ablation.py (comparable with the CLIP best=13.0% TPDD /
19.0% TLD) and the SAME TLD path as scripts/tld_zero_shot.py.

Answers the review's question: does a domain-specific VLM fix the near-zero-shot
collapse? Compares BioCLIP 2 best-template vs CLIP best-template vs ResNet-18.

Output: data/domain_vlm_zeroshot.json
"""
import os, sys, json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import open_clip
from torch.utils.data import DataLoader
from PIL import Image

REPO = ROOT
CKPT = "checkpoints/bioclip2/open_clip_pytorch_model.bin"
PAPER1 = ROOT + "/scripts"
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, PAPER1)
from dataset import TPDDDataset, EVAL_TRANSFORM
from prompt_ablation import tensor_to_pil, worker_init_fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL = "ViT-L-14"  # BioCLIP 2 arch (24 layers, width 1024, patch 14) -> ViT-L/14

CLASS_NAMES = ["anthracnose","black_shank","brown_spot","cmv","frog_eye",
    "genetic_abnormality","healthy","nematodes","potato_tuber_moth","pvy",
    "sunscald","target_spot","tmv","tswv","weather_fleck","wildfire"]

CN_MAP = {
    'black_shank':'黑胫病','cmv':'黄瓜花叶病毒','frog_eye':'蛙眼病','healthy':'健康叶',
    'nematodes':'根结线虫','pvy':'马铃薯Y病毒','tswv':'番茄斑萎病毒','wildfire':'野火病'}
CN_MAP.update({'anthracnose':'炭疽病','brown_spot':'褐斑病','genetic_abnormality':'遗传异常',
    'potato_tuber_moth':'马铃薯薯蛾','sunscald':'日灼病','target_spot':'靶斑病',
    'tmv':'烟草花叶病毒','weather_fleck':'气候斑病'})

def _resnet_ref():
    """Fine-tuned ResNet-18 full-data accuracy, read from canonical artifacts."""
    import json as _json
    from pathlib import Path as _P
    return _json.loads((_P(__file__).resolve().parent.parent
                        / "data" / "canonical_fig_artifacts.json").read_text()
                        )["resnet"]["mean"]


def _clip_best_ref():
    """(template name, accuracy) of the best zero-shot CLIP template, from data."""
    import json as _json
    from pathlib import Path as _P
    d = _P(__file__).resolve().parent.parent / "data"
    zs = _json.loads((d / "zero_shot_clean.json").read_text())["templates"]
    k = max(zs, key=lambda t: zs[t]["acc"])
    return f"{k} {zs[k]['acc'] * 100:.1f}%"


def get_kg():
    kg_path = os.path.join(REPO, "data/tpd_to_kg_map.json")
    kg_name_map = json.load(open(kg_path)) if os.path.exists(kg_path) else {}
    rec_path = os.path.join(REPO, "data/tobacco_records.json")
    kg_data = json.load(open(rec_path)) if os.path.exists(rec_path) else []
    kg_sym, kg_cond = {}, {}
    for name in CLASS_NAMES:
        kg_sym[name] = f"{CN_MAP.get(name,name)} disease symptoms on tobacco leaf"
        kg_cond[name] = f"conditions favoring {CN_MAP.get(name,name)} development"
    for name in CLASS_NAMES:
        # tpd_to_kg_map.json maps class -> {kg_entity, kg_id}, a dict. Assigning
        # the dict itself and then calling .lower() on it raised AttributeError
        # on the first class, so this script could not produce its own artifact.
        # Sibling scripts all unwrap it as info.get('kg_entity') and strip the
        # 'D_' prefix; do the same.
        info = kg_name_map.get(name) or {}
        kg_ent = info.get('kg_entity') or ''
        kg_name = kg_ent.replace('D_', '') if kg_ent else name
        for r in kg_data:
            if kg_name.lower() in r.get('name','').lower() or name.lower() in r.get('english_name','').lower():
                kg_sym[name] = r.get('symptoms','').split('。')[0][:100]
                kg_cond[name] = r.get('occurrence','').strip()[:100]
                break
    return kg_sym, kg_cond

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

def template_fill(name, cn, ks, kc, template):
    s = template.replace('{class_name}', name).replace('{cn_name}', cn)
    s = s.replace('{kg_symptom}', ks).replace('{kg_condition}', kc)
    return s

def run_dataset(model, preprocess, tokenizer, loader, nclass, names, kg_sym, kg_cond):
    results = []
    for tname, template in TEMPLATES:
        prompts = [template_fill(n, CN_MAP.get(n,n), kg_sym.get(n,''), kg_cond.get(n,''), template) for n in names]
        tokens = tokenizer(prompts).cuda()
        with torch.no_grad():
            tf = model.encode_text(tokens); tf = tf / tf.norm(dim=-1, keepdim=True)
        correct = total = 0
        per_class = {i:{'c':0,'t':0} for i in range(nclass)}
        for batch in loader:
            imgs, labels = batch['image'], batch['label']
            clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).cuda()
            with torch.no_grad():
                ife = model.encode_image(clip_imgs); ife = ife / ife.norm(dim=-1, keepdim=True)
                logits = (ife @ tf.T) * 100
            preds = logits.argmax(-1).cpu()
            correct += (preds==labels).sum().item(); total += labels.size(0)
            for p,l in zip(preds.tolist(), labels.tolist()):
                per_class[l]['c'] += (p==l); per_class[l]['t'] += 1
        acc = correct/total*100
        nz = sum(1 for i in range(nclass) if per_class[i]['c']>0)
        results.append({"template":tname, "accuracy":round(acc,2), "nonzero":nz,
                        "correct":correct, "total":total,
                        "per_class":{names[i]: per_class[i]['c']/max(per_class[i]['t'],1)*100 for i in range(nclass)}})
        print(f"  {tname:16s} {acc:5.2f}%  ({nz}/{nclass} nonzero)  {correct}/{total}")
    return results

def majority_baseline(loader, nclass):
    from collections import Counter
    cnt = Counter()
    for batch in loader:
        for l in batch['label'].tolist():
            cnt[l]+=1
    total = sum(cnt.values()); maj = max(cnt.values())
    return round(maj/total*100,2), maj, total

def load_model():
    print("Loading BioCLIP 2 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=CKPT)
    model = model.cuda().eval()
    tokenizer = open_clip.get_tokenizer(MODEL)
    return model, preprocess, tokenizer

def main():
    assert os.path.exists(CKPT), f"missing {CKPT}; run scripts/download_bioclip2.py first"
    model, preprocess, tokenizer = load_model()
    kg_sym, kg_cond = get_kg()
    out = {"model":"BioCLIP 2 (imageomics/bioclip-2, ViT-L/14)", "templates":14, "datasets":{}}

    # ---- TPDD (16 classes) ----
    print("\n=== TPDD test (16 classes) ===")
    ds = TPDDDataset('test', transform=EVAL_TRANSFORM)
    loader = DataLoader(ds, 16, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    tpdd = run_dataset(model, preprocess, tokenizer, loader, 16, CLASS_NAMES, kg_sym, kg_cond)
    bl, maj, tot = majority_baseline(loader, 16)
    best = max(tpdd, key=lambda r: r['accuracy'])
    out["datasets"]["tpdd"] = {"n_test":tot, "majority_baseline":bl,
        "best_template":best['template'], "best_accuracy":best['accuracy'],
        "gap_to_baseline":round(best['accuracy']-bl,2), "templates":tpdd,
        "clip_best_ref":_clip_best_ref(), "resnet18_ref":_resnet_ref()}
    print(f"  TPDD best={best['template']} {best['accuracy']}%  baseline={bl}%"
          f"  (CLIP best {_clip_best_ref()[1]}, ResNet-18 {_resnet_ref() * 100:.1f}%)")

    # ---- TLD (10 classes, distinct base images) ----
    print("\n=== TLD external test (10 classes) ===")
    import numpy as np
    from tld_zero_shot import (TLD_CLASSES, CN as TLD_CN, TEMPLATES as TLD_TEMPLATES,
                               MAN, build_kg as tld_build_kg, encode_images as tld_encode)
    man = json.load(open(MAN))
    cls_index = {c: i for i, c in enumerate(TLD_CLASSES)}
    items = []
    for m in man:
        cdir = m['relpath'].split('/')[1]
        if cdir not in cls_index:
            continue
        items.append((m['local'], cls_index[cdir]))
    imgf, true, base_mask = tld_encode(model, preprocess, items, cls_index)
    Nb = int(base_mask.sum())
    kg2_sym, kg2_cond = tld_build_kg()
    tld = []
    for tname, tmpl in TLD_TEMPLATES:
        prompts = []
        for c in TLD_CLASSES:
            s = tmpl.replace('{class_name}', c).replace('{cn_name}', TLD_CN.get(c, c))
            s = s.replace('{kg_symptom}', kg2_sym.get(c, '')).replace('{kg_condition}', kg2_cond.get(c, ''))
            prompts.append(s)
        with torch.no_grad():
            tf = model.encode_text(tokenizer(prompts).cuda()); tf = tf / tf.norm(dim=-1, keepdim=True)
        sim = 100.0 * (tf.cpu().numpy() @ imgf.T)          # [K, N]
        pred_b = sim[:, base_mask].argmax(axis=0); true_b = true[base_mask]
        cb = pred_b == true_b
        acc = float(cb.mean())
        per = {}
        for ci in range(len(TLD_CLASSES)):
            msk = true_b == ci; n = int(msk.sum()); cor = int((pred_b[msk] == ci).sum())
            per[TLD_CLASSES[ci]] = {'n': n, 'acc': round(cor / max(n, 1), 4)}
        tld.append({'template': tname, 'accuracy': round(acc * 100, 2),
                    'n_correct': int(cb.sum()), 'n': int(cb.size), 'per_class': per})
        print(f"  {tname:16s} {acc*100:5.2f}%  {int(cb.sum())}/{Nb}")
    cc_b = np.bincount(true[base_mask], minlength=len(TLD_CLASSES))
    bl2 = round(float(cc_b.max() / cc_b.sum()) * 100, 2)
    best2 = max(tld, key=lambda r: r['accuracy'])
    out["datasets"]["tld"] = {"n_test": Nb, "n_total": len(items), "majority_baseline": bl2,
        "best_template": best2['template'], "best_accuracy": best2['accuracy'],
        "gap_to_baseline": round(best2['accuracy'] - bl2, 2), "templates": tld,
        "clip_best_ref": "19.0%"}
    print(f"  TLD best={best2['template']} {best2['accuracy']}%  baseline={bl2}%  (CLIP best 19.0%)")

    out["verdict"] = (
        f"BioCLIP 2 best-template: TPDD {best['accuracy']}% (vs CLIP 13.0%, ResNet-18 86.4%), "
        f"TLD {best2['accuracy']}% (vs CLIP 19.0%). "
    )
    os.makedirs(os.path.join(REPO,"data"), exist_ok=True)
    jpath = os.path.join(REPO, "data/domain_vlm_zeroshot.json")
    json.dump(out, open(jpath,'w'), indent=2, ensure_ascii=False)
    print(f"\nSaved {jpath}")
    print(out["verdict"])

if __name__ == '__main__':
    main()
