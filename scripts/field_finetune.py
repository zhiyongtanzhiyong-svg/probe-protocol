import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from collections import Counter
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

class FieldDS(Dataset):
    def __init__(self, entries, field_dir, transform, class_names):
        self.entries = entries
        self.field_dir = field_dir
        self.transform = transform
        self.name_to_idx = {n: i for i, n in enumerate(class_names)}
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(os.path.join(self.field_dir, e["image"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.name_to_idx[e["label"]]

def make_tf(aug=True):
    m, s = (0.485,0.456,0.406), (0.229,0.224,0.225)
    base = [T.Resize(int(IMG_SIZE*1.14)), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(m, s)]
    if aug:
        base = [T.RandomHorizontalFlip(), T.RandomRotation(15), T.ColorJitter(0.2,0.2,0.2)] + base
    return T.Compose(base)

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

def eval_model(model, dl, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in dl:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

def train_resnet(train_e, test_e, field_dir, n_cls, seed, epochs=30, lr=3e-4, device=DEVICE):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    import torchvision.models as tm
    model = tm.resnet18(weights=tm.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, n_cls).to(device)
    model = model.to(device)
    tr_dl = DataLoader(FieldDS(train_e, field_dir, make_tf(True), cls_names), batch_size=16, shuffle=True)
    te_dl = DataLoader(FieldDS(test_e, field_dir, make_tf(False), cls_names), batch_size=32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0
    for ep in range(epochs):
        model.train()
        for imgs, labels in tr_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(imgs), labels)
            loss.backward(); opt.step()
        sched.step()
        acc = eval_model(model, te_dl, device)
        best = max(best, acc)
        model.train()
    del model; torch.cuda.empty_cache()
    return best

def train_convnext(train_e, test_e, field_dir, n_cls, seed, epochs=30, lr=1e-4, device=DEVICE):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    import torchvision.models as tm
    model = tm.convnext_tiny(weights=tm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(768, n_cls).to(device)
    model = model.to(device)
    tr_dl = DataLoader(FieldDS(train_e, field_dir, make_tf(True), cls_names), batch_size=8, shuffle=True)
    te_dl = DataLoader(FieldDS(test_e, field_dir, make_tf(False), cls_names), batch_size=16)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0
    for ep in range(epochs):
        model.train()
        for imgs, labels in tr_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(imgs), labels)
            loss.backward(); opt.step()
        sched.step()
        acc = eval_model(model, te_dl, device)
        best = max(best, acc)
        model.train()
    del model; torch.cuda.empty_cache()
    return best

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def clip_train_tf():
    return T.Compose([
        T.Resize(256), T.RandomResizedCrop(224, scale=(0.85,1.0)),
        T.RandomHorizontalFlip(), T.RandomRotation(15), T.ColorJitter(0.2,0.2,0.2,0.05),
        T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD),
    ])

def clip_eval_tf():
    return T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD),
    ])

def train_clip(train_e, test_e, field_dir, n_cls, seed, epochs=30, device=DEVICE):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    import open_clip
    ckpt = "checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin"
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained=ckpt)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.visual.parameters():
        p.requires_grad = True
    D = model.visual.output_dim
    head = nn.Linear(D, n_cls).to(device)
    pretrained_visual = {k: v.detach().cpu().clone() for k, v in model.visual.state_dict().items()}
    model.visual.load_state_dict(pretrained_visual)
    nn.init.kaiming_uniform_(head.weight, a=5)
    torch.nn.init.zeros_(head.bias)
    tr_dl = DataLoader(FieldDS(train_e, field_dir, clip_train_tf(), cls_names), batch_size=8, shuffle=True)
    te_dl = DataLoader(FieldDS(test_e, field_dir, clip_eval_tf(), cls_names), batch_size=16)
    vis_params = [p for p in model.visual.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": vis_params, "lr": 5e-6},
        {"params": head.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0
    for ep in range(epochs):
        model.visual.train()
        for imgs, labels in tr_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = head(model.visual(imgs))
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward(); opt.step()
        sched.step()
        model.visual.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in te_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = head(model.visual(imgs))
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100
        best = max(best, acc)
    del model, head; torch.cuda.empty_cache()
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", default=os.path.join(REPO, "data/field"))
    ap.add_argument("--out", default=os.path.join(REPO, "data/field_finetune.json"))
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--models", nargs="+", default=["resnet18", "convnext", "clip"])
    args = ap.parse_args()

    global cls_names
    entries = json.load(open(os.path.join(args.field_dir, "labels.json")))
    cls_names = sorted(set(e["label"] for e in entries))
    n_cls = len(cls_names)
    print(f"Field: {len(entries)} images, {n_cls} classes, GPU: {torch.cuda.get_device_name(0)}")

    train_e, test_e = stratified_split(entries, 0.25, seed=0)
    print(f"Split: {len(train_e)} train / {len(test_e)} test")
    print(f"Test classes: {dict(Counter(e['label'] for e in test_e).most_common())}")

    results = {
        "date": time.strftime("%Y-%m-%d"),
        "n_train": len(train_e), "n_test": len(test_e), "n_classes": n_cls,
        "seeds": args.seeds, "epochs": args.epochs,
    }

    for model_name in args.models:
        print(f"\n=== {model_name} ({args.seeds} seeds) ===")
        accs = []
        for i in range(args.seeds):
            seed = 42 + i
            t0 = time.time()
            if model_name == "resnet18":
                acc = train_resnet(train_e, test_e, args.field_dir, n_cls, seed, args.epochs)
            elif model_name == "convnext":
                acc = train_convnext(train_e, test_e, args.field_dir, n_cls, seed, args.epochs)
            elif model_name == "clip":
                acc = train_clip(train_e, test_e, args.field_dir, n_cls, seed, args.epochs)
            dt = time.time() - t0
            accs.append(acc)
            print(f"  seed {seed}: {acc:.1f}%  ({dt:.0f}s)")
            if (i+1) % 5 == 0:
                partial = np.array(accs)
                print(f"  partial: {partial.mean():.1f} +/- {partial.std():.1f}%")
        arr = np.array(accs)
        results[model_name] = {
            "mean": round(float(arr.mean()), 1),
            "std": round(float(arr.std()), 1),
            "seeds": accs,
        }
        print(f"  FINAL: {arr.mean():.1f} +/- {arr.std():.1f}%")

    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {args.out}")

if __name__ == "__main__":
    main()
