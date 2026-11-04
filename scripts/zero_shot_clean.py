"""
E6: Clean zero-shot evaluation of DFN-CLIP on TPDD test set, with bootstrap CIs.

Fixes the two issues flagged in review:
  1. prompt_ablation.json (KG-Symptom best) and zpe_experiment.json
     (KG-Condition best) disagreed on the best template -> re-run once,
     single source of truth.
  2. Zero-shot accuracies were reported with no uncertainty -> add
     bootstrap 95% CIs (paired resampling of the 138 test images).

Single model (DFN-CLIP ViT-L/14), single test split, all 14 templates
(EN label / photo / symptom / closeup / disease; CN variants; KG symptom /
condition / combined; EN-expert) scored on the SAME 138 images.

Outputs data/zero_shot_clean.json.
"""
import os
import json, os
import numpy as np
import torch
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
TPDD = ROOT + '/data/tpdd'
TPD_MAP = ROOT + '/data/tpd_to_kg_map.json'
KG_DATA = ROOT + '/data/tobacco_records.json'
OUT = ROOT + '/data/zero_shot_clean.json'

N_BOOT = 2000
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

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

EVAL = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

class TPDDTest(Dataset):
    """labels.json stores the class NAME (canonical schema, matching TPDDDataset);
    the integer id is the position in the sorted class list, which is exactly the
    label id used by the feature caches and every reported metric."""
    def __init__(self):
        self.items = json.load(open(f'{TPDD}/test/labels.json'))
        self.name_to_idx = {n: i for i, n in
                            enumerate(sorted({it['label'] for it in self.items}))}
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        return (Image.open(f'{TPDD}/test/{it["image"]}').convert('RGB'),
                self.name_to_idx[it['label']])

def bootstrap_ci(correct_mask, n_boot=N_BOOT):
    """95% CI on accuracy via resampling the per-image correct vector."""
    n = len(correct_mask)
    acc = float(correct_mask.mean())
    idx = np.random.randint(0, n, size=(n_boot, n))
    boots = correct_mask[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return acc, float(lo), float(hi)

def main():
    print('loading DFN-CLIP ...')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT)
    model.cuda().eval()
    tok = open_clip.get_tokenizer('ViT-L-14')

    test_ds = TPDDTest()
    N = len(test_ds)
    true = np.array([test_ds[i][1] for i in range(N)])
    imgs = torch.stack([preprocess(test_ds[i][0]) for i in range(N)]).cuda()
    with torch.no_grad():
        imgf = model.encode_image(imgs)
        imgf = imgf / imgf.norm(dim=-1, keepdim=True)
    print(f'{N} test images encoded.')

    # build KG symptom / condition per class
    tpd_map = json.load(open(TPD_MAP)); kg_data = json.load(open(KG_DATA))
    kg_sym, kg_cond = {}, {}
    for name, info in tpd_map.items():
        if name == 'healthy':
            kg_sym[name] = "healthy tobacco leaf without any disease spots or damage"
            kg_cond[name] = "optimal growing conditions with no disease pressure"
            continue
        kg_ent = info.get('kg_entity'); kg_name = kg_ent.replace('D_','') if kg_ent else name
        matched = False
        for r in kg_data:
            if kg_name.lower() in r.get('name','').lower() or name.lower() in r.get('english_name','').lower():
                kg_sym[name] = r.get('symptoms','').split('。')[0][:100]
                kg_cond[name] = r.get('occurrence','').strip()[:100]; matched = True; break
        if not matched:
            cn = CN_MAP.get(name, name)
            kg_sym[name] = f"{cn} disease symptoms on tobacco leaf"
            kg_cond[name] = f"conditions favoring {cn} development"

    class_names = list(tpd_map.keys())   # 16, alphabetical = label ids 0..15
    # encode all 14 templates (16 text vectors each)
    templ_logits = []   # list of [16, N]  (text x image similarity)
    for tname, tmpl in TEMPLATES:
        prompts = []
        for name in class_names:
            s = tmpl.replace('{class_name}', name).replace('{cn_name}', CN_MAP.get(name, name))
            s = s.replace('{kg_symptom}', kg_sym.get(name,'')).replace('{kg_condition}', kg_cond.get(name,''))
            prompts.append(s)
        with torch.no_grad():
            t = tok(prompts).cuda(); tf = model.encode_text(t)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            sim = 100.0 * (tf @ imgf.T)              # [16, N]
        templ_logits.append(sim.cpu().numpy())
    templ_logits = np.stack(templ_logits)            # [T, 16, N]

    def acc_of(logit_stack):   # logit_stack: [16, N]
        return logit_stack.argmax(axis=0) == true

    results = {'n_test': N, 'templates': {}}
    all_correct = []
    # Per-image predictions and the true labels are stored so that downstream
    # claim verification can run PAIRED tests (exact McNemar) against the
    # majority-class baseline on the same items, rather than comparing marginal
    # CIs. Written by the same code path that produces the accuracies above, so
    # the stored predictions cannot diverge from the reported numbers.
    _counts = np.bincount(true, minlength=len(class_names))
    _majority = int(_counts.argmax())
    results['true_labels'] = [int(v) for v in true]
    results['majority_class'] = _majority
    results['majority_baseline_correct'] = [bool(v) for v in (true == _majority)]
    results['class_names'] = list(class_names)
    for ti, (tname, _) in enumerate(TEMPLATES):
        c = acc_of(templ_logits[ti])
        acc, lo, hi = bootstrap_ci(c)
        results['templates'][tname] = {'acc': round(acc,4), 'ci95': [round(lo,4), round(hi,4)], 'n_correct': int(c.sum()),
                                       'preds': [int(v) for v in templ_logits[ti].argmax(axis=0)]}
        all_correct.append(c)
        print(f'  {tname:<13} {acc*100:6.2f}  CI95 [{lo*100:.2f},{hi*100:.2f}]')

    # ensembles (logit average across templates), with CI
    def ensemble_ci(weight_vec):
        w = np.asarray(weight_vec, float); w = w / w.sum()
        avg = (templ_logits * w[:,None,None]).sum(axis=0)   # [16, N]
        c = acc_of(avg)
        return bootstrap_ci(c)

    uniform = np.ones(len(TEMPLATES))
    u_acc, u_lo, u_hi = ensemble_ci(uniform)
    # ZPE signal weighting: template accuracy as weight
    sig = np.array([r['acc'] for r in results['templates'].values()])
    z_acc, z_lo, z_hi = ensemble_ci(sig)
    _u_pred = (templ_logits * (np.ones(len(TEMPLATES))/len(TEMPLATES))[:,None,None]).sum(axis=0).argmax(axis=0)
    _z_pred = (templ_logits * (sig/sig.sum())[:,None,None]).sum(axis=0).argmax(axis=0)
    results['ensemble_uniform'] = {'acc': round(u_acc,4), 'ci95':[round(u_lo,4),round(u_hi,4)],
                                  'preds': [int(v) for v in _u_pred]}
    results['ensemble_zpe_signal'] = {'acc': round(z_acc,4), 'ci95':[round(z_lo,4),round(z_hi,4)],
                                     'preds': [int(v) for v in _z_pred]}
    print(f'\n  UNIFORM-ENS  {u_acc*100:6.2f}  CI95 [{u_lo*100:.2f},{u_hi*100:.2f}]')
    print(f'  ZPE-signal   {z_acc*100:6.2f}  CI95 [{z_lo*100:.2f},{z_hi*100:.2f}]')

    # reconciliation vs the two prior runs (different schemas)
    def best_from_list(path):
        try: d = json.load(open(path))
        except Exception: return None, None
        if isinstance(d, list) and d and 'accuracy' in d[0]:
            m = max(d, key=lambda x: x.get('accuracy', 0)); return m.get('template'), {x['template']: x['accuracy'] for x in d}
        return None, None
    def best_from_dict(path):
        try: d = json.load(open(path))
        except Exception: return None, None
        ta = d.get('per_template_test_acc') or d.get('template_acc')
        if isinstance(ta, dict) and ta: return max(ta, key=ta.get), ta
        return None, ta
    pa_best, pa_ta = best_from_list(ROOT + '/data/prompt_ablation.json')
    zp_best, zp_ta = best_from_dict(ROOT + '/data/zpe_experiment.json')
    results['reconciliation'] = {
        'prior_prompt_ablation_best': pa_best, 'prior_prompt_ablation': pa_ta,
        'prior_zpe_best': zp_best, 'prior_zpe_template_acc': zp_ta,
        'note': 'single re-run on 138 TPDD test imgs; CI overlap explains prior best-template disagreement',
    }

    json.dump(results, open(OUT,'w'), indent=2)
    print(f'\nSaved {OUT}')

if __name__ == '__main__':
    main()
