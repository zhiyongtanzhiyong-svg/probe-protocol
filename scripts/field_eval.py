"""
Field evaluation: zero-shot + transfer fine-tune on 210 real-field tobacco images.

Protocol:
1. Zero-shot: CLIP DFN, CLIP LAION, SigLIP with English prompts
2. Transfer: CLIP DFN/LAION fine-tuned on TPDD → test on field (domain gap)
3. Fine-tune on field: all models, 15 seeds, stratified split

Output: data/field_eval.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chinese → English name mapping for prompts
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


class FieldDataset(Dataset):
    def __init__(self, field_dir, transform=None):
        self.entries = json.load(open(os.path.join(field_dir, "labels.json")))
        self.field_dir = field_dir
        self.transform = transform
        self.class_names = sorted(set(e["label"] for e in self.entries))
        self.name_to_idx = {n: i for i, n in enumerate(self.class_names)}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        path = os.path.join(self.field_dir, e["image"])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.name_to_idx[e["label"]]
        return img, label, e["label"]


def get_transform(model_type="clip", img_size=224):
    if model_type == "clip":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif model_type == "siglip":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        img_size = 384
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def zero_shot_eval(model, image_proc_or_transform, tokenizer, class_names_en, field_dir, model_type="clip", batch_size=16):
    ds = FieldDataset(field_dir, transform=image_proc_or_transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build text features
    en_names = [class_names_en.get(c, c) for c in ds.class_names]
    text_feats = []
    n_prompts = 5
    for name in en_names:
        for text in make_prompts(name):
            if model_type == "siglip":
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
                with torch.no_grad():
                    tf = model.get_text_features(**inputs)
            else:
                tokens = tokenizer([text]).to(DEVICE)
                with torch.no_grad():
                    tf = model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            text_feats.append(tf)
    text_feats = torch.cat(text_feats)

    # Image features + predict
    all_preds = []
    all_labels = []
    for images, labels, names in dl:
        images = images.to(DEVICE)
        if model_type == "siglip":
            with torch.no_grad():
                img_feats = model.get_image_features(pixel_values=images)
        else:
            with torch.no_grad():
                img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        # (batch, n_classes*n_templates)
        logits = (img_feats @ text_feats.T)
        # Average over templates per class
        n_cls = len(ds.class_names)
        n_tmpl = 5
        logits = logits.view(images.size(0), n_cls, n_tmpl).mean(dim=-1)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        del images, img_feats, logits, preds
    torch.cuda.empty_cache()

    # Accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = correct / len(all_preds) * 100

    # Per-class
    per_class = {}
    for i, name in enumerate(ds.class_names):
        cls_preds = [p for p, l in zip(all_preds, all_labels) if l == i]
        cls_correct = sum(1 for p in cls_preds if p == i)
        per_class[name] = {
            "n": len(cls_preds),
            "correct": cls_correct,
            "acc": cls_correct / len(cls_preds) * 100 if cls_preds else 0,
        }

    # Majority baseline
    label_counts = Counter(all_labels)
    majority = max(label_counts.values()) / len(all_labels) * 100

    return {
        "accuracy": round(acc, 1),
        "majority_baseline": round(majority, 1),
        "n_images": len(all_preds),
        "n_classes": len(ds.class_names),
        "per_class": per_class,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", default=os.path.join(REPO, "data/field"))
    ap.add_argument("--out", default=os.path.join(REPO, "data/field_eval.json"))
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    results = {"date": time.strftime("%Y-%m-%d"), "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"}

    # Load class names
    entries = json.load(open(os.path.join(args.field_dir, "labels.json")))
    class_names = sorted(set(e["label"] for e in entries))
    class_names_en = {c: ZH_EN.get(c, c) for c in class_names}
    results["n_images"] = len(entries)
    results["n_classes"] = len(class_names)
    results["class_distribution"] = dict(Counter(e["label"] for e in entries).most_common())
    print(f"Field dataset: {len(entries)} images, {len(class_names)} classes")

    # === 1. CLIP DFN zero-shot ===
    print("\n=== CLIP DFN-CLIP zero-shot ===")
    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin")
        clip_model = clip_model.to(DEVICE).eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        tf = get_transform("clip")
        res = zero_shot_eval(clip_model, tf, clip_tokenizer, class_names_en, args.field_dir, "clip", args.batch)
        results["clip_dfn_zeroshot"] = res
        print(f"  accuracy: {res['accuracy']}% (majority: {res['majority_baseline']}%)")
        del clip_model
        torch.cuda.empty_cache()
    except Exception as e:
        results["clip_dfn_zeroshot"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # === 2. CLIP LAION zero-shot ===
    print("\n=== CLIP LAION-2B zero-shot ===")
    try:
        clip_model2, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin")
        clip_model2 = clip_model2.to(DEVICE).eval()
        res = zero_shot_eval(clip_model2, tf, clip_tokenizer, class_names_en, args.field_dir, "clip", args.batch)
        results["clip_laion_zeroshot"] = res
        print(f"  accuracy: {res['accuracy']}% (majority: {res['majority_baseline']}%)")
        del clip_model2
        torch.cuda.empty_cache()
    except Exception as e:
        results["clip_laion_zeroshot"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # === 3. SigLIP zero-shot ===
    print("\n=== SigLIP zero-shot ===")
    try:
        from transformers import AutoModel, AutoTokenizer
        siglip_model = AutoModel.from_pretrained("checkpoints/siglip").to(DEVICE).eval()
        siglip_tokenizer = AutoTokenizer.from_pretrained("checkpoints/siglip")
        tf_siglip = get_transform("siglip")
        res = zero_shot_eval(siglip_model, tf_siglip, siglip_tokenizer, class_names_en, args.field_dir, "siglip", 8)
        results["siglip_zeroshot"] = res
        print(f"  accuracy: {res['accuracy']}% (majority: {res['majority_baseline']}%)")
        del siglip_model
        torch.cuda.empty_cache()
    except Exception as e:
        results["siglip_zeroshot"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # Save
    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
