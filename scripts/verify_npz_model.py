"""Definitively identify which CLIP checkpoint produced clip_features_clean.npz
by comparing test-image feature vectors (mean cosine similarity)."""
import os
import sys, os, numpy as np, torch
sys.path.insert(0, ROOT + '/scripts')
from dataset import TPDDataset, EVAL_TRANSFORM
import open_clip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
npz = np.load(os.path.join(BASE, 'data', 'clip_features_clean.npz'))
ref = npz['test_feats'].astype('float32')
ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
ref = torch.from_numpy(ref)
n = len(ref)
print(f'npz test_feats: {ref.shape}')

test_ds = TPDDDataset('test', transform=EVAL_TRANSFORM)
CKPTS = {
    'DFN-CLIP':  'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin',
    'LAION-2B':  'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin',
}
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for name, ckpt in CKPTS.items():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ckpt)
    model = model.eval().to(DEVICE)
    # gather images in dataset order
    feats = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            img = test_ds[i]['image']
            f = model.encode_image(img.to(DEVICE).unsqueeze(0))
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    feats = torch.cat(feats, dim=0)[:n]
    # mean cosine sim (row-aligned)
    cos = (feats * ref).sum(dim=1)
    print(f'{name}: mean cosine = {cos.mean().item():.6f}  (median {cos.median().item():.6f}, min {cos.min().item():.4f})')
    del model
    torch.cuda.empty_cache()
