"""
Frozen-CLIP patch-token extraction for the foreground-anchored patch-attention
head (this paper A+B expansion, EXPERIMENT_PLAN block B1/B2).

The paper's CLS features are the CLIP embedding `encode_image(img)`. This module
returns the SAME CLS embedding (byte-identical, verified) together with the 196
patch tokens in the same 768-d embedding space, so a head trained on patch tokens
is apples-to-apples comparable to the linear probe trained on the CLS token.

Input pipeline is IDENTICAL to the canonical protocol so the CLS features match
the paper exactly: dataset EVAL_TRANSFORM -> tensor_to_pil (de-normalise) ->
open_clip preprocess -> model. Patch tokens then come from the same preprocessed
batch via the ViT's forward_intermediates.

Extraction (open_clip 2.32.0, ViT-L/14, width=1024 -> proj -> 768):
    out = vis.forward_intermediates(
              img, indices=1,                       # last resblock = full depth
              normalize_intermediates=True,         # ln_post applied to full seq
              output_fmt='NLC',                     # (B, N, width) layout
              output_extra_tokens=True)             # keep the CLS prefix token
    out['image_intermediates']        : [(B, 196, width)]  CLS already split off
    out['image_intermediates_prefix'] : [(B,   1, width)]  CLS
    both projected by vis.proj (width -> 768).

CLS consistency: ViT._pool for ViT-L/14 uses final_ln_after_pool, so
encode_image = ln_post(x)[:,0] @ proj -- the SAME as the CLS prefix here.
__main__ asserts they match to < 2e-4 (float32 / PIL round-trip tolerance).
"""
import os, sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import open_clip
from dataset import TPDDDataset, EVAL_TRANSFORM, worker_init_fn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
CKPT = {
    'dfn_clip': 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin',
    'laion2b': 'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin',
}


def tensor_to_pil(img_tensor):
    from PIL import Image
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np_img * IMAGENET_STD + IMAGENET_MEAN
    np_img = np.clip(np_img, 0, 1)
    return Image.fromarray((np_img * 255).astype(np.uint8))


def _preprocess_batch(model_preprocess, imgs):
    return torch.stack([model_preprocess(tensor_to_pil(im)) for im in imgs]).to(DEVICE)


def get_tokens(model, clip_imgs):
    """Return (cls (B,768), patches (B,196,768)) in the CLIP embedding space.

    `clip_imgs` is the open_clip-preprocessed batch (B,3,224,224). The CLS vector
    equals model.encode_image(clip_imgs) (both ln_post(x)[:,0] @ proj)."""
    vis = model.visual
    out = vis.forward_intermediates(
        clip_imgs, indices=1, normalize_intermediates=True,
        output_fmt='NLC', output_extra_tokens=True)
    patches = out['image_intermediates'][0]            # (B, 196, width)
    cls = out['image_intermediates_prefix'][0][:, 0]   # (B, width)
    if vis.proj is not None:
        patches = patches @ vis.proj                   # (B, 196, 768)
        cls = cls @ vis.proj                           # (B, 768)
    return cls, patches


def extract_patch_features(model, preprocess, dataset, bs=32):
    """Full-dataset extraction. Returns (cls (N,768), patches (N,196,768), labels (N,))."""
    cls_list, patch_list, labs = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, bs, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn):
            clip_imgs = _preprocess_batch(preprocess, batch['image'])
            cls, patches = get_tokens(model, clip_imgs)
            cls_list.append(cls.cpu().numpy())
            patch_list.append(patches.cpu().numpy())
            labs.extend(batch['label'].numpy().tolist())
    return np.concatenate(cls_list), np.concatenate(patch_list), np.array(labs)


def sanity_check(n_imgs=32, ckpt_name='dfn_clip'):
    """R001 sanity: shapes + CLS consistency with encode_image + patch-token stats."""
    ckpt = CKPT[ckpt_name]
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ckpt)
    model = model.to(DEVICE).eval()
    ds = TPDDDataset(split='train', transform=EVAL_TRANSFORM)

    idx = list(range(min(n_imgs, len(ds))))
    imgs = torch.stack([ds[i]['image'] for i in idx]).to(DEVICE)
    clip_imgs = _preprocess_batch(preprocess, [ds[i]['image'] for i in idx])

    with torch.no_grad():
        cls, patches = get_tokens(model, clip_imgs)
        emb = model.encode_image(clip_imgs)           # (B,768)  paper's CLS feature

    emb = emb / emb.norm(dim=-1, keepdim=True)
    cls_n = cls / cls.norm(dim=-1, keepdim=True)
    max_diff = float((emb - cls_n).abs().max())
    cos = float((emb * cls_n).sum(1).mean())

    P, d = patches.shape[1], patches.shape[2]
    print(f"ckpt={ckpt_name}")
    print(f"  cls     {tuple(cls.shape)}   emb {tuple(emb.shape)}")
    print(f"  patches {tuple(patches.shape)}   (P={P}, d={d})")
    print(f"  CLS consistency: max|emb-cls|={max_diff:.6f}  mean cos={cos:.6f}")
    print(f"  patch stats: mean|patch|={float(patches.abs().mean()):.4f}  "
          f"mean|patch_i - patch_j|={float((patches - patches.mean(1, keepdim=True)).abs().mean()):.4f}")
    print(f"  vis.proj={'None' if model.visual.proj is None else tuple(model.visual.proj.shape)}")
    # ViT-L/14 on 224px -> 16x16 = 256 patches (patch size 14: 224/14=16).
    ok = max_diff < 2e-4 and P == 256 and d == 768
    print(f"  SANITY {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'dfn_clip'
    ok = sanity_check(ckpt_name=name)
    sys.exit(0 if ok else 1)
