"""
Multi-Model × Multi-Dataset CLIP Evaluation for Tobacco Disease Classification.
2 VLMs (DFN-CLIP, SigLIP) × 2 datasets (TPDD 16-class, TLD 10-class).
Evaluates zero-shot + few-shot linear probe.
"""
import os
import sys, json, torch, numpy as np, os
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from PIL import Image
sys.path.insert(0, ROOT + '/scripts')

import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM, TRAIN_TRANSFORM, worker_init_fn
from exp_crossdataset import TLD_CLASS_NAMES, TLD_TO_TPDD, TLDDataset

# ===== Setup =====
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_pil(img_tensor):
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

class KShotSubset(Dataset):
    def __init__(self, full_ds, K, seed, class_names):
        self.transform = full_ds.transform
        self.class_names = class_names
        # Build label index by sampling from the dataset
        rng = np.random.RandomState(seed)
        all_labels = []
        loader = DataLoader(full_ds, 32, shuffle=False, num_workers=0)
        for batch in loader:
            all_labels.extend(batch['label'].tolist() if hasattr(batch['label'], 'tolist') else [batch['label']])
        
        indices_by_class = {i: [] for i in range(len(class_names))}
        for idx, lbl in enumerate(all_labels):
            if lbl < len(class_names):
                indices_by_class[lbl].append(idx)
        
        selected = []
        for c, idx_list in indices_by_class.items():
            if idx_list:
                selected.extend(rng.choice(idx_list, size=min(K, len(idx_list)), replace=False))
        self.indices = sorted(selected)
        self.full_ds = full_ds
    
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.full_ds[self.indices[idx]]

# ===== Models to evaluate =====
MODELS = {
    "DFN-CLIP-ViTL14": ("open_clip", "ViT-L-14", "checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin"),
    "OpenAI-CLIP-ViTL14": ("openai_clip", "ViT-L/14", "checkpoints/clip_openai_vitl14/pytorch_model.bin"),
}

# ===== Datasets =====
from datasets import load_dataset
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TPDD
tpd_map = json.load(open(ROOT + '/data/tpd_to_kg_map.json'))
tpd_eng_names = list(tpd_map.keys())
tpd_test = TPDDDataset('test', transform=EVAL_TRANSFORM)
tpd_train_full = TPDDDataset('train', transform=TRAIN_TRANSFORM)
tpd_class_names = tpd_eng_names  # use English names for prompts
print(f"TPDD: {len(tpd_test)} test images, {len(tpd_class_names)} classes")

# TLD
hf_ds = load_dataset('Joker5800/Tobacco_Leaf_Diseases', split='test', trust_remote_code=True)
rng = np.random.RandomState(42)
indices = list(range(len(hf_ds)))
rng.shuffle(indices)
n_train = int(len(indices) * 0.06)
n_val = int(len(indices) * 0.04)
train_idx = indices[:n_train]
test_idx = indices[n_train + n_val:]
tld_test = TLDDataset(hf_ds, test_idx, transform=EVAL_TRANSFORM)
tld_train_full = TLDDataset(hf_ds, train_idx, transform=TRAIN_TRANSFORM)
tld_classes = TLD_CLASS_NAMES
print(f"TLD: {len(tld_test)} test images, {len(tld_classes)} classes")
print(f"  Classes: {tld_classes}")

# ===== Evaluation =====
RESULTS = {}

for model_name, (lib, arch, ckpt_path) in MODELS.items():
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    
    if lib == "open_clip":
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=ckpt_path)
        tokenizer_fn = open_clip.get_tokenizer(arch)
        model = model.eval().cuda()
    elif lib == "openai_clip":
        import clip as openai_clip
        model, preprocess = openai_clip.load(arch, device='cpu')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        tokenizer_fn = lambda texts: openai_clip.tokenize(texts).cuda()
        model = model.cuda().eval()
    else:
        continue
    
    for ds_name, test_ds, train_full, class_names in [
        ("TPDD", tpd_test, tpd_train_full, tpd_class_names),
        ("TLD", tld_test, tld_train_full, tld_classes)
    ]:
        print(f"\n  --- {ds_name} ({len(test_ds)} test images) ---")
        
        # Zero-shot
        prompts = [f"a photo of a tobacco leaf with {n} disease" for n in class_names]
        tokens = tokenizer_fn(prompts)
        with torch.no_grad():
            tf = model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        
        loader = DataLoader(test_ds, 16, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
        correct, total = 0, 0
        for batch in loader:
            imgs, labels = batch['image'], batch['label']
            clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).cuda()
            with torch.no_grad():
                ife = model.encode_image(clip_imgs)
                ife = ife / ife.norm(dim=-1, keepdim=True)
                logits = (ife @ tf.T) * 100
            preds = logits.argmax(-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        zs_acc = correct / total * 100
        print(f"    Zero-shot (EN-Photo): {zs_acc:.1f}%")
        
        # Few-shot probe
        n_seeds = 5  # Reduced for speed; TPDD already has n=15
        for K in [1, 5, -1]:
            accs = []
            for seed in range(n_seeds):
                if K == -1:
                    train_subset = train_full
                else:
                    train_subset = KShotSubset(train_full, K, seed + 500, class_names)
                
                # Extract features
                tr_loader = DataLoader(train_subset, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
                tr_feats, tr_labs = [], []
                for batch in tr_loader:
                    imgs, labels = batch['image'], batch['label']
                    clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).cuda()
                    with torch.no_grad():
                        f = model.encode_image(clip_imgs)
                        f = f / f.norm(dim=-1, keepdim=True)
                    tr_feats.append(f.cpu().numpy())
                    tr_labs.extend(labels.numpy().tolist() if hasattr(labels, 'numpy') else [labels])
                tr_feats = np.concatenate(tr_feats)
                tr_labs = np.array(tr_labs)
                
                # Test features (extract once)
                if seed == 0:
                    te_loader = DataLoader(test_ds, 32, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
                    te_feats, te_labs = [], []
                    for batch in te_loader:
                        imgs, labels = batch['image'], batch['label']
                        clip_imgs = torch.stack([preprocess(tensor_to_pil(img)) for img in imgs]).cuda()
                        with torch.no_grad():
                            f = model.encode_image(clip_imgs)
                            f = f / f.norm(dim=-1, keepdim=True)
                        te_feats.append(f.cpu().numpy())
                        te_labs.extend(labels.numpy().tolist() if hasattr(labels, 'numpy') else [labels])
                    te_feats = np.concatenate(te_feats)
                    te_labs = np.array(te_labs)
                
                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
                clf.fit(tr_feats, tr_labs)
                preds = clf.predict(te_feats)
                accs.append(float((preds == te_labs).mean()))
            
            accs = np.array(accs)
            label = f"K={K}" if K > 0 else "full"
            key = f"{model_name}_{ds_name}_{label}"
            RESULTS[key] = {"mean": float(accs.mean()), "std": float(accs.std(ddof=1)), 
                           "n_seeds": n_seeds, "accs": accs.tolist()}
            print(f"    Probe {label:5s}: {accs.mean()*100:.1f}% ± {accs.std(ddof=1)*100:.1f} (n={n_seeds})")
        
        RESULTS[f"{model_name}_{ds_name}_zero-shot"] = {"accuracy": zs_acc}

# Save
os.makedirs(ROOT + '/data/', exist_ok=True)
json.dump(RESULTS, open(ROOT + '/data/multimodel_results.json', 'w'), indent=2)

# Summary table
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"{'':25s} {'Zero-Shot':>10s} {'K=1':>12s} {'K=5':>12s} {'Full':>12s}")
print(f"{'-'*70}")
for model_name in MODELS:
    for ds_name in ["TPDD", "TLD"]:
        zs = RESULTS.get(f"{model_name}_{ds_name}_zero-shot", {}).get('accuracy', 0)
        k1 = RESULTS.get(f"{model_name}_{ds_name}_K=1", {})
        k5 = RESULTS.get(f"{model_name}_{ds_name}_K=5", {})
        fl = RESULTS.get(f"{model_name}_{ds_name}_full", {})
        print(f"{model_name+' '+ds_name:25s} {zs:9.1f}% {k1.get('mean',0)*100:10.1f}%±{k1.get('std',0)*100:4.1f} {k5.get('mean',0)*100:10.1f}%±{k5.get('std',0)*100:4.1f} {fl.get('mean',0)*100:10.1f}%±{fl.get('std',0)*100:4.1f}")

print("\nSaved to multimodel_results.json")
