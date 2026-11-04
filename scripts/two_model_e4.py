"""
E4: Second, generic CLIP ViT-L/14 (LAION-2B, standard CLIP training) evaluated on
TPDD with the SAME protocol as DFN-CLIP, to test whether the CLIP<=ResNet gap is
systematic (paradigm-level) rather than an artifact of the DFN-CLIP checkpoint.

Self-contained (reads TPDD labels.json directly; no Paper1 dataset.py dependency):
  * Zero-shot: same 14 templates as E6 (zero_shot_clean.py) + bootstrap 95% CIs.
  * Few-shot linear probe: K per class in {1,2,5,10,full}, LogisticRegression(C=1.0),
    15 seeds, on L2-normed frozen features -- identical protocol to
    clip_fewshot_probe.py / the paper's CLIP few-shot.
  * Comparison block vs DFN-CLIP (data/clip_fewshot.json, data/zero_shot_clean.json)
    and ResNet fine-tune (data/resnet_k1/k5/k10_n15.json).

Outputs data/two_model_e4.json.
"""
import os, json
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
import open_clip
from PIL import Image
from torchvision import transforms

ROOT   = ROOT
CKPT   = 'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin'
TPDD   = ROOT + '/data/tpdd'
TPD_MAP= ROOT + '/data/tpd_to_kg_map.json'
KG_DATA= f'{ROOT}/data/tobacco_records.json'
OUT    = f'{ROOT}/data/two_model_e4.json'
N_BOOT = 2000
N_SEEDS= 15
SEED   = 0
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
    ("EN-Label","{class_name}"), ("CN-Label","{cn_name}"),
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
    transforms.Resize((256,256)), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def load_split(split):
    # labels.json stores the class NAME (canonical schema, matching TPDDDataset);
    # the integer id is its position in the sorted class list, i.e. the label id
    # used by the feature caches and every reported metric.
    items = json.load(open(f'{TPDD}/{split}/labels.json'))
    name_to_idx = {n: i for i, n in
                   enumerate(sorted({it['label'] for it in items}))}
    paths = [f'{TPDD}/{split}/{it["image"]}' for it in items]
    labels = np.array([name_to_idx[it['label']] for it in items])
    return paths, labels

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def tensor_to_pil(t):
    """Undo EVAL_TRANSFORM's ImageNet normalisation back to a uint8 PIL image.

    This is the canonical CLIP pixel path used by
    canonical_fig_artifacts.extract_clip: round-trip the eval tensor through
    uint8, then re-apply open_clip's own preprocess.

    The previous implementation skipped this and fed the ImageNet-normalised
    tensor straight to encode_image. That is a THIRD pixel path, matching
    neither the canonical nor the clean pipeline, and it left the LAION-2B arm
    non-comparable with the DFN-CLIP and ResNet arms in the few-shot table
    (K=1 probe 39.8% here vs 40.1% on the canonical path).
    """
    a = t.permute(1, 2, 0).cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def encode(model, preprocess, paths, batch=64):
    """Extract L2-normalised CLIP features on the canonical pixel path.

    `preprocess` is now actually used; it was previously accepted and ignored.
    """
    assert preprocess is not None, "preprocess must be open_clip's transform"
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            X = torch.stack([preprocess(tensor_to_pil(EVAL(Image.open(p).convert('RGB'))))
                             for p in paths[i:i+batch]]).cuda()
            f = model.encode_image(X); f = f/f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu()); del X
    return torch.cat(feats, dim=0).numpy()

def bootstrap_ci(mask, n_boot=N_BOOT):
    n=len(mask); acc=float(mask.mean())
    if n==0: return 0.,0.,0.
    idx=np.random.randint(0,n,size=(n_boot,n)); b=mask[idx].mean(1)
    return acc, float(np.percentile(b,2.5)), float(np.percentile(b,97.5))

def build_kg():
    tpd_map=json.load(open(TPD_MAP)); kg=json.load(open(KG_DATA))
    sym, cond = {}, {}
    for name, info in tpd_map.items():
        if name=='healthy':
            sym[name]="healthy tobacco leaf without any disease spots or damage"
            cond[name]="optimal growing conditions with no disease pressure"; continue
        kg_ent=info.get('kg_entity'); kg_name=kg_ent.replace('D_','') if kg_ent else name; matched=False
        for r in kg:
            if kg_name.lower() in r.get('name','').lower() or name.lower() in r.get('english_name','').lower():
                sym[name]=r.get('symptoms','').split('。')[0][:100]; cond[name]=r.get('occurrence','').strip()[:100]; matched=True; break
        if not matched:
            cn=CN_MAP.get(name,name); sym[name]=f"{cn} disease symptoms on tobacco leaf"; cond[name]=f"conditions favoring {cn} development"
    return sym, cond

def kshot_idx(labels, K, seed):
    rng=np.random.RandomState(seed+500)
    by={c:[] for c in np.unique(labels)}
    for i,l in enumerate(labels): by[l].append(i)
    out=[]
    for c in sorted(by): out.extend(rng.choice(by[c], size=min(K,len(by[c])), replace=False))
    return np.sort(out)

def main():
    print('loading LAION-2B ViT-L/14 ...')
    model, pp, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT)
    model.cuda().eval(); tok=open_clip.get_tokenizer('ViT-L-14')

    tr_p, tr_l = load_split('train'); te_p, te_l = load_split('test')
    print(f'train {len(tr_p)} / test {len(te_p)}')
    trf = encode(model, pp, tr_p); tef = encode(model, pp, te_p)

    # ---- zero-shot (14 templates + CIs) ----
    class_names=list(json.load(open(TPD_MAP)).keys())   # 16, alphabetical = label ids 0..15
    kg_sym, kg_cond = build_kg()
    imgf_t = torch.from_numpy(tef)
    tl=[]
    for tname,tmpl in TEMPLATES:
        prompts=[]
        for name in class_names:
            s=tmpl.replace('{class_name}',name).replace('{cn_name}',CN_MAP.get(name,name))
            s=s.replace('{kg_symptom}',kg_sym.get(name,'')).replace('{kg_condition}',kg_cond.get(name,''))
            prompts.append(s)
        with torch.no_grad():
            t=tok(prompts).cuda(); tf=model.encode_text(t); tf=tf/tf.norm(dim=-1,keepdim=True)
        tl.append((100.0*(tf.cpu().numpy() @ tef.T)))     # [16, N]
    tl=np.stack(tl)
    zs={}
    for ti,(tname,_) in enumerate(TEMPLATES):
        c=tl[ti].argmax(0)==te_l
        a,l,h=bootstrap_ci(c)
        zs[tname]={'acc':round(float(a),4),'ci95':[round(float(l),4),round(float(h),4)],'n_correct':int(c.sum())}
    best=max(zs.items(), key=lambda kv:kv[1]['acc'])
    # ensemble uniform
    avg=tl.mean(0); c=avg.argmax(0)==te_l; ea,el,eh=bootstrap_ci(c)
    print('zero-shot best:', best[0], f"{best[1]['acc']*100:.2f}  CI[{best[1]['ci95'][0]*100:.2f},{best[1]['ci95'][1]*100:.2f}]")
    for tn,v in sorted(zs.items(), key=lambda x:-x[1]['acc'])[:5]: print(f'   {tn:13s} {v["acc"]*100:6.2f}')

    # ---- few-shot linear probe ----
    probe={}
    for K in [1,2,5,10,-1]:
        accs=[]
        for seed in range(N_SEEDS):
            if K==-1: idx=np.arange(len(tr_l))
            else: idx=kshot_idx(tr_l, K, seed)
            clf=LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(trf[idx], tr_l[idx]); p=clf.predict(tef)
            accs.append(float((p==te_l).mean()))
        accs=np.array(accs)
        label=f"K={K}" if K>0 else "full"
        probe[label]={'mean':float(accs.mean()),'std':float(accs.std(ddof=1)),'accs':accs.tolist(),'n_seeds':N_SEEDS}
        print(f'LAION-2B probe {label:5s}: {accs.mean()*100:.1f}% ± {accs.std(ddof=1)*100:.1f}')

    # ---- comparison anchors ----
    def rd(p):
        try: return json.load(open(p))
        except Exception: return None
    def probe_mean(d,label):
        return round(d[label]['mean']*100,2) if d and label in d else None
    dfn_probe=rd(f'{ROOT}/data/clip_fewshot.json')
    rn1=rd(f'{ROOT}/data/resnet_k1_n15.json'); rn5=rd(f'{ROOT}/data/resnet_k5_n15.json'); rn10=rd(f'{ROOT}/data/resnet_k10_n15.json')
    def rn_mean(d): return round(d['mean']*100,2) if d else None
    dfn_zs=rd(f'{ROOT}/data/zero_shot_clean.json')
    dfn_zs_best=None
    if dfn_zs and 'templates' in dfn_zs:
        bt=max(dfn_zs['templates'].items(), key=lambda kv:kv[1]['acc']); dfn_zs_best={'template':bt[0],'acc':round(bt[1]['acc']*100,2)}
    res={
      'model':'CLIP-ViT-L-14-laion2B_s32b_b82k (standard CLIP, LAION-2B)',
      'n_train':int(len(tr_p)),'n_test':int(len(te_p)),
      'zero_shot':zs,'zero_shot_best':best[0],'zero_shot_ensemble_uniform':{'acc':round(float(ea),4),'ci95':[round(float(el),4),round(float(eh),4)]},
      'few_shot_probe':probe,
      'comparison':{
        'few_shot_acc_pct':{
            'K=1':{'laion2b':probe_mean(probe,'K=1'),'dfn_clip':probe_mean(dfn_probe,'K=1'),'resnet':rn_mean(rn1)},
            'K=2':{'laion2b':probe_mean(probe,'K=2'),'dfn_clip':probe_mean(dfn_probe,'K=2'),'resnet':None},
            'K=5':{'laion2b':probe_mean(probe,'K=5'),'dfn_clip':probe_mean(dfn_probe,'K=5'),'resnet':rn_mean(rn5)},
            'K=10':{'laion2b':probe_mean(probe,'K=10'),'dfn_clip':probe_mean(dfn_probe,'K=10'),'resnet':rn_mean(rn10)},
            'full':{'laion2b':probe_mean(probe,'full'),'dfn_clip':probe_mean(dfn_probe,'full'),'resnet':None},
        },
        'zero_shot_best_pct':{'dfn_clip':dfn_zs_best,'laion2b':round(best[1]['acc']*100,2)},
        'note':'same protocol (K/class, LR C=1.0, 15 seeds, 138 TPDD test imgs); ResNet = full fine-tune (30 epochs, 15 seeds). '
               'If LAION-2B CLIP also <= ResNet and tracks DFN-CLIP, the gap is systematic across CLIP variants.',
      },
    }
    json.dump(res, open(OUT,'w'), indent=2)
    print(f'\nSaved {OUT}')

if __name__=='__main__':
    main()
