"""
E5-TLD: External cross-dataset zero-shot evaluation of DFN-CLIP on the TLD
dataset (Joker5800/Tobacco_Leaf_Diseases), replacing the flawed
"84 images" number (which was a fabricated 6%-of-test carve-out in
kg_tld_validation.py, 2 templates, no CIs, lossy TLD->TPDD KG mapping).

Design (clean & self-contained, comparable to E6 zero_shot_clean):
  * Label set = TLD's OWN 10 classes (a true external vocabulary the model has
    never seen), NOT a lossy remap onto TPDD's 16.
  * Same 14 prompt templates as E6. The 11 name-based templates need only the
    class names (clean). The 3 KG templates use best-effort KG text per TLD
    class (7 map to a known TPDD disease; 3 special classes use documented
    name-based KG text) -- reported as SECONDARY because KG transfer to a new
    label set is inherently confounded.
  * Primary metric = 232 DISTINCT base images (independent observations).
    Augmented-inclusive (1392) reported as a sensitivity check.
  * Bootstrap 95% CIs (paired resampling), baselines (majority / uniform),
    uniform ensembles, per-class (best name-based template).

Outputs data/tld_zero_shot.json.
"""
import os, json
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import numpy as np, torch
import open_clip
from PIL import Image
from torchvision import transforms

ROOT   = ROOT
CKPT   = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
MAN    = f'{ROOT}/data/tld_local_manifest.json'
TPD_MAP= ROOT + '/data/tpd_to_kg_map.json'
KG_DATA= f'{ROOT}/data/tobacco_records.json'
OUT    = f'{ROOT}/data/tld_zero_shot.json'
N_BOOT = 2000
SEED   = 0
torch.manual_seed(SEED); np.random.seed(SEED)

# Fixed 10-class TLD vocabulary (matches the on-disk class dirs / manifest).
TLD_CLASSES = [
    'Brown spot', 'Frogeye', 'Healthy', 'Mosaic', 'Potato virus Y',
    'Target spot', 'Weather fleck', 'alternaria alternata',
    'cercospora nicotianae', 'no cercospora nicotianae or alternaria alternata present',
]
CN = {
    'Brown spot':'赤星病', 'Frogeye':'蛙眼病', 'Healthy':'健康', 'Mosaic':'花叶病',
    'Potato virus Y':'马铃薯Y病毒病', 'Target spot':'靶斑病', 'Weather fleck':'气候斑点病',
    'alternaria alternata':'早疫病', 'cercospora nicotianae':'链格孢叶斑病',
    'no cercospora nicotianae or alternaria alternata present':'未感染链格孢或早疫病',
}
# KG-text source disease (7 clean maps). 3 special classes get documented fallbacks below.
KG_SRC = {
    'Brown spot':'brown_spot', 'Frogeye':'frog_eye', 'Healthy':'healthy',
    'Mosaic':'tmv', 'Potato virus Y':'pvy', 'Target spot':'target_spot',
    'Weather fleck':'weather_fleck',
}
KG_FALLBACK = {
    'alternaria alternata': dict(sym='early blight (Alternaria alternata) dark brown circular spots with concentric rings and yellow halos on a tobacco leaf',
                                 cond='warm humid weather with free surface water favoring Alternaria'),
    'cercospora nicotianae': dict(sym='Cercospora frog-eye leaf spot: circular gray-centered spots with dark raised margins on a tobacco leaf',
                                  cond='warm humid conditions favoring Cercospora leaf spot'),
    'no cercospora nicotianae or alternaria alternata present': dict(sym='tobacco leaf free of Cercospora frog-eye spots and early blight lesions',
                                                                      cond='healthy conditions with no Cercospora or Alternaria pressure'),
}
TEMPLATES = [
    ("EN-Label","{class_name}"),
    ("EN-Photo","a photo of a tobacco leaf with {class_name}"),
    ("EN-Symptom","tobacco leaf showing {class_name}"),
    ("EN-Closeup","a close-up photo of {class_name} on a tobacco leaf"),
    ("EN-Disease","{class_name} on a tobacco leaf"),
    ("EN-Expert","tobacco plant pathology: {class_name} identification"),
    ("CN-Label","{cn_name}"),
    ("CN-Photo","一张患有{cn_name}的烟草叶片照片"),
    ("CN-Symptom","烟草叶片表现出{cn_name}"),
    ("CN-Closeup","烟草叶片上{cn_name}的特写照片"),
    ("CN-Disease","烟草叶片上的{cn_name}"),
    ("KG-Symptom","tobacco leaf: {kg_symptom}"),
    ("KG-Condition","tobacco leaf disease occurring when: {kg_condition}"),
    ("KG-Combined","tobacco leaf with {kg_symptom}, occurs when {kg_condition}"),
]
NONKG = [t for t,_ in TEMPLATES if not t.startswith('KG')]

EVAL = transforms.Compose([
    transforms.Resize((256,256)), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def bootstrap_ci(correct_mask, n_boot=N_BOOT):
    n = len(correct_mask); acc = float(correct_mask.mean())
    if n == 0: return 0.0, 0.0, 0.0
    idx = np.random.randint(0, n, size=(n_boot, n))
    boots = correct_mask[idx].mean(axis=1)
    return acc, float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))

def build_kg():
    tpd_map = json.load(open(TPD_MAP)); kg = json.load(open(KG_DATA))
    sym, cond = {}, {}
    def from_tpdd(tpdd_name):
        if tpdd_name == 'healthy':
            return ("healthy tobacco leaf without any disease spots or damage",
                    "optimal growing conditions with no disease pressure")
        info = tpd_map.get(tpdd_name, {}); kg_ent = info.get('kg_entity')
        if kg_ent:
            kn = kg_ent.replace('D_','')
            for r in kg:
                if kn.lower() in r.get('name','').lower() or tpdd_name.lower() in r.get('english_name','').lower():
                    return r.get('symptoms','').split('。')[0][:100], r.get('occurrence','').strip()[:100]
        cn = tpdd_name
        return (f"{cn} disease symptoms on tobacco leaf", f"conditions favoring {cn} development")
    for c in TLD_CLASSES:
        if c in KG_SRC: sym[c], cond[c] = from_tpdd(KG_SRC[c])
        else: sym[c] = KG_FALLBACK[c]['sym']; cond[c] = KG_FALLBACK[c]['cond']
    return sym, cond

def encode_images(model, preprocess, items, cls_index, batch=64):
    # items: list of (local_path, class_idx); returns imgf [N, D] and true [N], base_mask [N]
    tensors = []; true = []; base = []
    for path, ci in items:
        im = Image.open(path).convert('RGB')
        tensors.append(preprocess(im)); true.append(ci)
        base.append('_aug' not in os.path.basename(path))
    feats = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch):
            X = torch.stack(tensors[i:i+batch]).cuda()
            f = model.encode_image(X); f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
            del X
    imgf = torch.cat(feats, dim=0).numpy()
    return imgf, np.array(true), np.array(base)

def main():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT)
    model.cuda().eval(); tok = open_clip.get_tokenizer('ViT-L-14')

    man = json.load(open(MAN))
    cls_index = {c: i for i, c in enumerate(TLD_CLASSES)}
    items = []
    for m in man:
        rel = m['relpath']                      # test/<class>/<file>.jpg
        cdir = rel.split('/')[1]
        if cdir not in cls_index:
            continue
        items.append((m['local'], cls_index[cdir]))
    print(f'loaded {len(items)} local images')
    imgf, true, base_mask = encode_images(model, preprocess, items, cls_index)
    N = len(items); Nb = int(base_mask.sum())
    print(f'encoded {N} (base {Nb}, aug {N-Nb})')

    kg_sym, kg_cond = build_kg()
    # encode all 14 templates -> [T, 10, N]
    tl = []
    for tname, tmpl in TEMPLATES:
        prompts = []
        for c in TLD_CLASSES:
            s = tmpl.replace('{class_name}', c).replace('{cn_name}', CN.get(c, c))
            s = s.replace('{kg_symptom}', kg_sym.get(c,'')).replace('{kg_condition}', kg_cond.get(c,''))
            prompts.append(s)
        with torch.no_grad():
            t = tok(prompts).cuda(); tf = model.encode_text(t); tf = tf/tf.norm(dim=-1, keepdim=True)
        sim = 100.0 * (tf.cpu().numpy() @ imgf.T)               # [10, N]
        tl.append(sim)
    tl = np.stack(tl)                                            # [T,10,N]

    def acc_of(stack):      # stack [10,N]
        return stack.argmax(axis=0) == true

    res = {'n_total': N, 'n_base': Nb, 'classes': TLD_CLASSES, 'class_counts': None,
           'templates_nonkg': {}, 'templates_kg': {}, 'ensemble_uniform_nonkg': None,
           'ensemble_uniform_all': None, 'baselines': {}, 'per_class_best_nonkg': {},
           'sensitivity_augmented_inclusive': {}, 'notes': {}}

    cc = np.bincount(true, minlength=10)
    res['class_counts'] = {TLD_CLASSES[i]: int(cc[i]) for i in range(10)}

    for ti,(tname,_) in enumerate(TEMPLATES):
        # PRIMARY: base (distinct) images only
        pred_b = tl[ti][:, base_mask].argmax(axis=0)
        true_b = true[base_mask]
        cb = pred_b == true_b
        acc, lo, hi = bootstrap_ci(cb)
        rec = {'acc': round(float(acc),4), 'ci95':[round(float(lo),4), round(float(hi),4)], 'n_correct': int(cb.sum()), 'n': int(cb.size)}
        (res['templates_nonkg'] if not tname.startswith('KG') else res['templates_kg'])[tname] = rec
        print(f'  {tname:<13} base {acc*100:6.2f}  CI95 [{lo*100:.2f},{hi*100:.2f}]')

    # ensembles (logit mean over templates), base only
    def ens(names):
        idx = [i for i,(tn,_) in enumerate(TEMPLATES) if tn in names]
        avg = tl[idx].mean(axis=0)                              # [10,N]
        pred_b = avg[:, base_mask].argmax(axis=0); cb = pred_b == true[base_mask]
        return bootstrap_ci(cb)
    a,l,h = ens(NONKG);   res['ensemble_uniform_nonkg'] = {'acc':round(float(a),4),'ci95':[round(float(l),4),round(float(h),4)]}
    a,l,h = ens([t for t,_ in TEMPLATES]); res['ensemble_uniform_all'] = {'acc':round(float(a),4),'ci95':[round(float(l),4),round(float(h),4)]}
    print(f'  ENSEMBLE(non-KG) base {a*100:.2f}  CI95 [{l*100:.2f},{h*100:.2f}]')
    print(f'  ENSEMBLE(all)    base {a*100:.2f}')

    # baselines on base
    cc_b = np.bincount(true[base_mask], minlength=10); maj = cc_b.max()/cc_b.sum()
    res['baselines'] = {'majority': round(float(maj),4), 'uniform_1overK': round(1/10,4),
                        'largest_class': TLD_CLASSES[int(cc_b.argmax())]}

    # per-class for the best non-KG template
    best_nk = max(res['templates_nonkg'].items(), key=lambda kv: kv[1]['acc'])[0]
    pred_b = tl[[i for i,(tn,_) in enumerate(TEMPLATES) if tn==best_nk][0]][:, base_mask].argmax(axis=0)
    per = {}
    for ci in range(10):
        m = true[base_mask]==ci; n = int(m.sum()); cor = int((pred_b[m]==ci).sum())
        per[TLD_CLASSES[ci]] = {'n': n, 'acc': round(cor/max(n,1),4)}
    res['per_class_best_nonkg'] = {'template': best_nk, 'per_class': per}

    # sensitivity: augmented-inclusive (all N) for the non-KG set
    sens = {}
    for ti,(tname,_) in enumerate(TEMPLATES):
        if tname.startswith('KG'): continue
        cb = acc_of(tl[ti])
        a,l,h = bootstrap_ci(cb)
        sens[tname] = {'acc':round(float(a),4),'ci95':[round(float(l),4),round(float(h),4)]}
    idx = [i for i,(tn,_) in enumerate(TEMPLATES) if not tn.startswith('KG')]
    avg = tl[idx].mean(axis=0); cb = avg.argmax(axis=0)==true; a,l,h = bootstrap_ci(cb)
    sens['ensemble_uniform_nonkg'] = {'acc':round(float(a),4),'ci95':[round(float(l),4),round(float(h),4)]}
    res['sensitivity_augmented_inclusive'] = sens

    res['notes'] = {
        'label_set': 'TLD own 10 classes (external vocabulary), NOT remapped onto TPDD 16',
        'primary_split': f'{Nb} distinct base images (augmented _aug0-4 variants excluded = independent obs)',
        'kg_mapping': {'clean': KG_SRC, 'fallback_documented': list(KG_FALLBACK.keys()),
                       'caveat': 'KG transfer to a new label set is inherently confounded -> reported as secondary'},
        'fixes': ['replaces 84-image fabricated 6%-of-test split (kg_tld_validation.py:96-102)',
                  'adds bootstrap CIs', '14 templates vs 2', 'removes lossy TLD->TPDD KG remap from primary'],
        'prior_flawed': 'kg_tld_validation.py reported only 2 templates, no CI, on a shuffled carve-out',
    }
    json.dump(res, open(OUT,'w'), indent=2)
    print(f'\nSaved {OUT}')

if __name__ == '__main__':
    main()
