"""14-template uniform-logit ensemble zero-shot, both checkpoints, current repo state.
Replaces the 09-28-era ensemble numbers (7.97% DFN / 17.39% LAION) used as the
CoOp 'zero-shot prior' context in the manuscript."""
import os
import sys, json
sys.path.insert(0, ROOT + '/scripts')
import dataset as D  # must import BEFORE torch (import-order quirk in this env)
TPDDDataset = D.TPDDDataset
import torch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
import open_clip  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tf = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ds = TPDDDataset('test')
y = np.array([ds.name_to_idx[l] for l in ds.class_labels])
CKPTS = {
    'dfn_clip': 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin',
    'laion2b': 'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin'}

CN = {"anthracnose": "炭疽病", "black_shank": "黑胫病", "brown_spot": "赤星病", "cmv": "黄瓜花叶病毒病",
      "frog_eye": "蛙眼病", "genetic_abnormality": "遗传异常", "healthy": "健康", "nematodes": "根结线虫病",
      "potato_tuber_moth": "马铃薯块茎蛾", "pvy": "马铃薯Y病毒病", "sunscald": "日灼病", "target_spot": "靶斑病",
      "tmv": "烟草普通花叶病毒病", "tswv": "番茄斑萎病毒病", "weather_fleck": "气候斑点病", "wildfire": "野火病"}


def build_prompts(class_names):
    tpd_map = json.load(open('data/tpd_to_kg_map.json'))
    kg_data = json.load(open('data/tobacco_records.json'))
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
    T = [("EN-Label", "{class_name}"), ("CN-Label", "{cn_name}"),
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
         ("EN-Expert", "tobacco plant pathology: {class_name} disease identification")]
    out = []
    for tname, tmpl in T:
        ps = []
        for name in class_names:
            s = tmpl.replace('{class_name}', name).replace('{cn_name}', CN.get(name, name))
            s = s.replace('{kg_symptom}', kg_sym.get(name, '')).replace('{kg_condition}', kg_cond.get(name, ''))
            ps.append(s)
        out.append(ps)
    return out


res = {'protocol': '14-template uniform logit ensemble (identical to scripts/coop_official.py zero-shot), '
                   'raw PIL + ImageNet normalization, current images + checkpoints'}
for ck, path in CKPTS.items():
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=path)
    model.cuda().eval()
    tok = open_clip.get_tokenizer('ViT-L-14')
    with torch.no_grad():
        feats = []
        for i in range(0, len(ds.images), 16):
            batch = torch.stack([tf(Image.open(p).convert('RGB')) for p in ds.images[i:i + 16]]).cuda()
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float64))
        imgf = np.vstack(feats)
    sims = []
    for ps in build_prompts(ds.class_names):
        with torch.no_grad():
            t = model.encode_text(tok(ps).cuda())
            t = t / t.norm(dim=-1, keepdim=True)
            t = t.cpu().numpy().astype(np.float64)
        sims.append(t @ imgf.T)
    S = np.stack(sims)
    pred = S.mean(axis=0).argmax(axis=0)
    acc = float((pred == y).mean())
    res[ck] = {'acc': round(acc, 4), 'n_correct': int((pred == y).sum()), 'preds': pred.tolist()}
    print(f'{ck}: 14-template ensemble = {acc * 100:.2f}%  ({int((pred == y).sum())}/138)')
    del model
    torch.cuda.empty_cache()
json.dump(res, open('data/zero_shot_ensemble_20261025.json', 'w'), indent=2, ensure_ascii=False)
print('saved data/zero_shot_ensemble_20261025.json')
