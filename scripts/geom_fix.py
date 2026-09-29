"""Focused recompute of feature-geometry metrics with the CORRECTED intra-class
distance (off-diagonal mean only). Reuses canonical_protocol's exact extraction
path so features are identical to the source of truth. Prints + saves the four
(CLIP-DFN / ResNet-frozen) x (train / test) geometry blocks."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import canonical_protocol as cp

train_ds = cp.TPDDDataset(split='train', transform=cp.EVAL_TRANSFORM)
test_ds = cp.TPDDDataset(split='test', transform=cp.EVAL_TRANSFORM)
n_classes = len(train_ds.class_names)
DEV = cp.DEVICE
print("Device:", DEV, "| n_classes:", n_classes, "| train:", len(train_ds), "| test:", len(test_ds), flush=True)

# CLIP DFN (primary)
m, _, pp = cp.open_clip.create_model_and_transforms('ViT-L-14', pretrained=cp.CLIP_CKPTS['dfn_clip'])
m = m.to(DEV).eval()
ct, cty = cp.extract_clip(m, pp, train_ds, DEV)
ce, cey = cp.extract_clip(m, pp, test_ds, DEV)
del m
torch.cuda.empty_cache()
print("CLIP DFN extracted", ct.shape, ce.shape, flush=True)

# ResNet-18 frozen
rn = cp.models.resnet18(weights=cp.models.ResNet18_Weights.IMAGENET1K_V1).to(DEV).eval()
rt, rty = cp.extract_resnet_frozen(rn, train_ds, DEV)
re_, rey = cp.extract_resnet_frozen(rn, test_ds, DEV)
del rn
torch.cuda.empty_cache()
print("ResNet-frozen extracted", rt.shape, re_.shape, flush=True)

out = {
    'clip_dfn_train': cp.compute_geometry(ct, cty, n_classes),
    'clip_dfn_test': cp.compute_geometry(ce, cey, n_classes),
    'resnet_frozen_train': cp.compute_geometry(rt, rty, n_classes),
    'resnet_frozen_test': cp.compute_geometry(re_, rey, n_classes),
}
print(json.dumps(out, indent=2), flush=True)
with open(os.path.join(cp.DATA_DIR, 'geometry_fixed.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("SAVED", os.path.join(cp.DATA_DIR, 'geometry_fixed.json'), flush=True)
