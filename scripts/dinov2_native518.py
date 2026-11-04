"""
DINOv2 ViT-L/14 @ NATIVE 518px (reviewer #7 fix).

The earlier DINOv2 result (77.5% best-C) was evaluated at 224px (Resize256->CenterCrop224),
but DINOv2 ViT-L/14 is pretrained at 518px with a 37x37 pos-embed grid (518/14). Reviewer
point #7 asks for the native-resolution evaluation.

This script feeds the ORIGINAL image (no prior downscale/crop) into
Resize(518, bicubic) -> CenterCrop(518) -> ImageNet-normalize, so the patch grid is exactly
37x37 and the pos-embed is used with NO interpolation (true native resolution). The 224px
number is kept as a reference; this adds the 518px number.

Same canonical probe protocol (C-grid, best-C on 20% val, MLP, kNN) and same held-out test set.
Outputs data/dinov2_native518.json.
"""
import os
import sys, os, json, time
import numpy as np, torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, ROOT + "/scripts")
from dataset import TPDDDataset
from canonical_protocol import run_clip_protocol
from dinov2_protocol import load_dinov2, PTH, DEVICE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NATIVE = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class RawPIL(Dataset):
    """Returns the ORIGINAL image (no 224-crop) as a PIL plus its label."""
    def __init__(self, split):
        base = TPDDDataset(split, transform=None)
        self.paths = [str(base.images[i]) for i in range(len(base))]
        self.labels = list(base.label_ids)
        self.class_names = base.class_names
        self.label_ids = base.label_ids
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return Image.open(self.paths[i]).convert("RGB"), self.labels[i]


def extract_native(model, ds, device, batch=8):
    feats, labs = [], []
    model.eval()
    idxs = list(range(len(ds)))
    for i in range(0, len(idxs), batch):
        chunk = idxs[i:i + batch]
        pils = [ds[j] for j in chunk]  # (PIL, label)
        x = torch.stack([NATIVE(p) for (p, _) in pils]).to(device)
        with torch.no_grad():
            f = model(x)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            feats.append(f.cpu().numpy())
        labs.extend([lbl for (_, lbl) in pils])
    return np.concatenate(feats), np.array(labs)


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}", flush=True)
    assert os.path.exists(PTH), f"missing {PTH}"
    model = load_dinov2(PTH)

    train_ds = RawPIL("train")
    test_ds = RawPIL("test")
    n_classes = len(train_ds.class_names)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {n_classes}", flush=True)

    print("Extracting DINOv2 features @ 518px (native)...", flush=True)
    tr, tr_y = extract_native(model, train_ds, DEVICE)
    te, te_y = extract_native(model, test_ds, DEVICE)
    print(f"  train {tr.shape} test {te.shape}", flush=True)
    del model
    torch.cuda.empty_cache()

    print("Canonical probe protocol @ 518px...", flush=True)
    proto = run_clip_protocol(tr, tr_y, te, te_y)
    R = {
        "model": "DINOv2 ViT-L/14, 1024-dim CLS, frozen",
        "resolution": "518px native (Resize518 bicubic -> CenterCrop518; 37x37 pos-embed, no interpolation)",
        "protocol": "same canonical protocol as canonical_protocol.py; original image fed to 518px transform (no prior 224-crop); held-out test set (138)",
        "n_train": int(len(tr_y)), "n_test": int(len(te_y)), "n_classes": int(n_classes),
        "full_data": {
            "linear_C1": {k: v for k, v in proto["linear_C1"].items() if k != "preds"},
            "linear_bestC": {k: v for k, v in proto["linear_bestC"].items() if k != "preds"},
            "mlp": {k: v for k, v in proto["mlp"].items() if k != "preds"},
            "knn1": {k: v for k, v in proto["knn1"].items() if k != "preds"},
            "best_C": proto["best_C"], "val_accs": proto["val_accs"],
        },
        "refs": {**canonical_refs(), "dinov2_224px_bestC": 0.7754},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(R, open(os.path.join(REPO, "data", "dinov2_native518.json"), "w"), indent=2)
    print(f"\nDINOv2 @518px: C=1 {proto['linear_C1']['accuracy']*100:.1f}% | "
          f"best-C(C={proto['best_C']}) {proto['linear_bestC']['accuracy']*100:.1f}% | "
          f"MLP {proto['mlp']['accuracy']*100:.1f}% | kNN(1) {proto['knn1']['accuracy']*100:.1f}%", flush=True)
    print(f"  [ref] DINOv2@224px best-C 77.5% | CLIP 85.5% | ResNet-18 86.4%", flush=True)
    print(f"Done in {R['elapsed_sec']}s", flush=True)



def canonical_refs():
    """Cross-backbone reference numbers, read from canonical_fig_artifacts.json.

    Hard-coding these meant every re-run of the referenced experiment left a
    stale number in the shipped artifact, and three of these artifacts are cited
    by the manuscript. Read them instead.
    """
    import json as _json
    from pathlib import Path as _P
    art = _json.loads((_P(__file__).resolve().parent.parent
                       / "data" / "canonical_fig_artifacts.json").read_text())
    return {"clip_dfn_bestC_full": art["clip"]["linear_bestC"]["accuracy"],
            "resnet18_finetuned_full": art["resnet"]["mean"]}


if __name__ == "__main__":
    main()
