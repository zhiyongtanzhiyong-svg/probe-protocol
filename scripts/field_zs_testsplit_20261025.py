"""
N2 fix (second-round review): the field zero-shot table was evaluated on all
246 field images (193 of which are training images of the field fine-tune
runs), while the fine-tuned models are evaluated on the 53-image test split.
This re-runs the zero-shot evaluation on exactly the 53-image test split
(same stratified_split as scripts/field_finetune.py: per-class shuffle with
random.Random(0), test_frac=0.25), using the same 5-template scheme and
transforms as scripts/field_eval.py, so the zero-shot and fine-tuned
comparisons in the field table are on the identical image set.

Runs on CPU to avoid GPU contention with the K=1/2 fairness experiment.

Output: data/field_zs_testsplit_20261025.json
"""
import os, sys, json, time, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
DEVICE = "cpu"  # deliberate: GPU is occupied by the K12 fairness run

ZH_EN = {
    "healthy": "healthy",
    "烟草花叶病": "tobacco mosaic virus",
    "烟草曲叶病": "tobacco leaf rolling disease",
    "烟叶赤星病": "tobacco target spot disease",
    "烟草枯萎病": "tobacco wilt disease",
    "烟草炭疽病": "tobacco anthracnose",
    "烟叶黑胫病": "tobacco black shank disease",
    "烟草白粉病": "tobacco powdery mildew",
    "烟草立枯病": "tobacco damping off disease",
    "烟草野火病": "tobacco wildfire disease",
    "烟草蛙眼病": "tobacco frog eye leaf spot",
    "烟草剑叶病": "tobacco sword leaf abnormality",
    "灰霉病": "tobacco gray mold botrytis",
    "二氯喹啉酸残留药害": "tobacco chemical herbicide damage",
    "烟草角斑病": "tobacco angular leaf spot",
    "烟草青枯病": "tobacco bacterial wilt",
    "烟草霜霉病": "tobacco downy mildew",
    "烟草斑点病": "tobacco leaf spot disease",
    "烟草黑斑病": "tobacco black spot disease",
}

def make_prompts(label):
    if label == "healthy":
        return [
            "a photo of a healthy tobacco leaf",
            "a photo of a normal tobacco leaf with no disease",
            "a closeup photo of a healthy green tobacco leaf",
            "a tobacco leaf in good health",
            "healthy tobacco leaf",
        ]
    return [
        f"a photo of a tobacco leaf with {label}",
        f"a photo of a {label} affected tobacco leaf",
        f"a closeup photo of a leaf with {label}",
        f"a photo of a leaf showing {label} symptoms",
        label,
    ]

def stratified_split(entries, test_frac=0.25, seed=0):
    by_cls = {}
    for e in entries:
        by_cls.setdefault(e["label"], []).append(e)
    rng = random.Random(seed)
    train, test = [], []
    for cls, items in by_cls.items():
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_frac))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test

def get_transform(model_type="clip", img_size=224):
    if model_type == "clip":
        mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    elif model_type == "siglip":
        mean = (0.5, 0.5, 0.5); std = (0.5, 0.5, 0.5); img_size = 384
    else:
        mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def main():
    field_dir = os.path.join(REPO, "data/field")
    entries = json.load(open(os.path.join(field_dir, "labels.json")))
    train_e, test_e = stratified_split(entries, 0.25, 0)
    print(f"total={len(entries)} train={len(train_e)} test={len(test_e)}")
    assert len(test_e) == 53, f"expected 53 test images, got {len(test_e)}"

    class_names = sorted(set(e["label"] for e in test_e))
    class_names_en = {c: ZH_EN.get(c, c) for c in class_names}
    test_ids = [e["image"] for e in test_e]

    def run_clip(model, tokenizer, transform, tag):
        # text features (5 templates per class)
        text_feats = []
        for c in class_names:
            for text in make_prompts(class_names_en[c]):
                tokens = tokenizer([text]).to(DEVICE)
                with torch.no_grad():
                    tf = model.encode_text(tokens)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                text_feats.append(tf)
        text_feats = torch.cat(text_feats)
        # image features
        all_preds, all_labels = [], []
        bs = 8
        for i in range(0, len(test_ids), bs):
            idx = list(range(i, min(i + bs, len(test_ids))))
            imgs = []
            for j in idx:
                p = os.path.join(field_dir, test_ids[j])
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            imgs = torch.stack(imgs).to(DEVICE)
            with torch.no_grad():
                img_feats = model.encode_image(imgs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = (img_feats @ text_feats.T)
            n_cls = len(class_names)
            logits = logits.view(imgs.size(0), n_cls, 5).mean(dim=-1)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend([class_names.index(e["label"]) for e in test_e[idx[0] - i + 0: idx[0] - i + len(idx)]] if False else [class_names.index(e["label"]) for e in [test_e[k] for k in idx]])
            del imgs, img_feats, logits
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        per_class = {}
        for ci, c in enumerate(class_names):
            cp = [p for p, l in zip(all_preds, all_labels) if l == ci]
            cc = sum(1 for p in cp if p == ci)
            per_class[c] = {"n": len(cp), "correct": cc, "acc": round(cc / len(cp) * 100, 1) if cp else 0.0}
        counts = Counter(all_labels)
        maj = max(counts.values()) / len(all_labels) * 100
        return {
            "tag": tag,
            "accuracy": round(correct / len(all_preds) * 100, 1),
            "n_correct": correct,
            "n_images": len(all_preds),
            "majority_baseline": round(maj, 1),
            "per_class": per_class,
        }

    out = {
        "date": time.strftime("%Y-%m-%d"),
        "note": "Zero-shot re-evaluated on the 53-image field TEST split (same split as field_finetune.py: stratified_split test_frac=0.25, seed=0), matching the fine-tuned models' evaluation set. Same 5-template scheme and transforms as scripts/field_eval.py. Runs on CPU.",
        "n_test": len(test_e),
        "test_ids": test_ids,
        "class_distribution_test": dict(Counter(e["label"] for e in test_e).most_common()),
    }

    import open_clip
    print("=== DFN-CLIP ===")
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin")
    m = m.to(DEVICE).eval()
    tok = open_clip.get_tokenizer("ViT-L-14")
    out["clip_dfn_zeroshot_test53"] = run_clip(m, tok, get_transform("clip"), "dfn")
    del m
    print(out["clip_dfn_zeroshot_test53"]["accuracy"], "baseline", out["clip_dfn_zeroshot_test53"]["majority_baseline"])

    print("=== LAION-2B ===")
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin")
    m = m.to(DEVICE).eval()
    out["clip_laion_zeroshot_test53"] = run_clip(m, tok, get_transform("clip"), "laion")
    del m
    print(out["clip_laion_zeroshot_test53"]["accuracy"], "baseline", out["clip_laion_zeroshot_test53"]["majority_baseline"])

    print("=== SigLIP ===")
    try:
        from transformers import AutoModel, AutoTokenizer
        sm = AutoModel.from_pretrained("checkpoints/siglip").to(DEVICE).eval()
        st = AutoTokenizer.from_pretrained("checkpoints/siglip")
        text_feats = []
        for c in class_names:
            for text in make_prompts(class_names_en[c]):
                inputs = st(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
                with torch.no_grad():
                    tf = sm.get_text_features(**inputs)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                text_feats.append(tf)
        text_feats = torch.cat(text_feats)
        all_preds, all_labels = [], []
        bs = 4
        for i in range(0, len(test_ids), bs):
            idx = list(range(i, min(i + bs, len(test_ids))))
            imgs = torch.stack([get_transform("siglip")(Image.open(os.path.join(field_dir, test_ids[j])).convert("RGB")) for j in idx]).to(DEVICE)
            with torch.no_grad():
                img_feats = sm.get_image_features(pixel_values=imgs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = (img_feats @ text_feats.T).view(imgs.size(0), len(class_names), 5).mean(dim=-1)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend([class_names.index(test_e[k]["label"]) for k in idx])
            del imgs, img_feats, logits
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        per_class = {}
        for ci, c in enumerate(class_names):
            cp = [p for p, l in zip(all_preds, all_labels) if l == ci]
            cc = sum(1 for p in cp if p == ci)
            per_class[c] = {"n": len(cp), "correct": cc, "acc": round(cc / len(cp) * 100, 1) if cp else 0.0}
        counts = Counter(all_labels)
        out["siglip_zeroshot_test53"] = {
            "tag": "siglip",
            "accuracy": round(correct / len(all_preds) * 100, 1),
            "n_correct": correct,
            "n_images": len(all_preds),
            "majority_baseline": round(max(counts.values()) / len(all_labels) * 100, 1),
            "per_class": per_class,
        }
        del sm
        print(out["siglip_zeroshot_test53"]["accuracy"], "baseline", out["siglip_zeroshot_test53"]["majority_baseline"])
    except Exception as e:
        out["siglip_zeroshot_test53"] = {"error": str(e)}
        print("SigLIP error:", e)

    outpath = os.path.join(REPO, "data/field_zs_testsplit_20261025.json")
    with open(outpath, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved", outpath)

if __name__ == "__main__":
    main()
