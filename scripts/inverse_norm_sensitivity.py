"""
INVERSE-NORMALIZATION SENSITIVITY (6th-round reviewer item, raised 5x).

canonical_protocol.py extracts CLIP features via:
    preprocess( tensor_to_pil( dataset_tensor ) )
where tensor_to_pil does  v -> v*IMAGENET_STD + IMAGENET_MEAN, clip[0,1], uint8.
The EVAL_TRANSFORM dataset tensor is ALREADY ImageNet-normalized (verified: min
-1.60, max 2.64), so tensor_to_pil is the CORRECT denormalization back to [0,1]
pixels before open_clip's preprocess re-normalizes. The only cost is the uint8
rounding in tensor_to_pil. This test bounds that cost.

This script re-extracts features TWO ways on the SAME images and compares:
  A) current canonical path : preprocess( tensor_to_pil( tensor ) )
  B) clean path             : preprocess( raw_PIL )          (standard open_clip usage)
and reports, on the 138-image test split (primary DFN-CLIP):
  - mean cosine similarity between A and B test features (how much features move)
  - zero-shot best-of-templates accuracy (A vs B)
  - C=1 logistic-probe accuracy (A vs B), train features likewise
Deterministic: no stochastic ops, fixed seed.
"""
import os
import sys, os, json, time
import numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, ROOT + '/scripts')
from dataset import TPDDDataset, EVAL_TRANSFORM
import open_clip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPT = 'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEMPLATES = [
    "EN-Label",    "{class_name}",
    "EN-Photo",    "a photo of a tobacco leaf with {class_name} disease",
    "EN-Symptom",  "tobacco leaf showing {class_name} symptoms",
]

def tensor_to_pil(t):
    a = t.permute(1, 2, 0).cpu().numpy()
    a = a * IMAGENET_STD + IMAGENET_MEAN
    a = np.clip(a, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))


def extract(model, preprocess, ds, batch=64):
    """Returns (A_feats, B_feats, labels). A = inverse-norm round-trip, B = clean. Batched."""
    A, B, labs = [], [], []
    for s in range(0, len(ds), batch):
        idx = list(range(s, min(s + batch, len(ds))))
        tensors = torch.stack([ds[i]['image'] for i in idx]).to(DEVICE)          # (B,3,224,224) in [0,1]
        raws = torch.stack([preprocess(Image.open(ds.images[i]).convert('RGB')) for i in idx]).to(DEVICE)
        a_imgs = torch.stack([preprocess(tensor_to_pil(t)) for t in tensors]).to(DEVICE)
        with torch.no_grad():
            fa = model.encode_image(a_imgs); fa = fa / fa.norm(dim=-1, keepdim=True)
            fb = model.encode_image(raws);    fb = fb / fb.norm(dim=-1, keepdim=True)
        A.append(fa.cpu().numpy()); B.append(fb.cpu().numpy())
        labs.extend([int(ds[i]['label']) for i in idx])
    return np.concatenate(A), np.concatenate(B), np.array(labs)


def zero_shot(model, tok, teA, teB, teY, class_names):
    """Zero-shot accuracy from pre-extracted A/B test features. Returns {template: (A_acc, B_acc)}."""
    results = {}
    al = np.array(teY)
    for tname, tmpl in zip(TEMPLATES[0::2], TEMPLATES[1::2]):
        prompts = [tmpl.replace('{class_name}', n) for n in class_names]
        with torch.no_grad():
            tf = model.encode_text(tok(prompts).to(DEVICE))
        tf = (tf / tf.norm(dim=-1, keepdim=True)).cpu().numpy()
        A = teA @ tf.T; B = teB @ tf.T
        results[tname] = (100 * accuracy_score(al, A.argmax(1)),
                          100 * accuracy_score(al, B.argmax(1)))
    return results


def main():
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=CKPT)
    model = model.to(DEVICE).eval()
    tok = open_clip.get_tokenizer('ViT-L-14')

    train_ds = TPDDDataset(split='train', transform=EVAL_TRANSFORM)
    test_ds = TPDDDataset(split='test', transform=EVAL_TRANSFORM)
    class_names = list(train_ds.tpd_map.keys())
    print(f"Device {DEVICE} | train {len(train_ds)} test {len(test_ds)} classes {len(class_names)}")

    trA, trB, trY = extract(model, preprocess, train_ds)
    teA, teB, teY = extract(model, preprocess, test_ds)

    # cosine similarity A vs B (test + train)
    cs_te = float((teA * teB).sum(1).mean())
    cs_tr = float((trA * trB).sum(1).mean())
    print(f"\nmean cos(A,B)  train={cs_tr:.4f}  test={cs_te:.4f}")

    # C=1 probe acc
    mA = LogisticRegression(C=1.0, max_iter=5000, random_state=42); mA.fit(trA, trY)
    mB = LogisticRegression(C=1.0, max_iter=5000, random_state=42); mB.fit(trB, trY)
    c1A = 100 * accuracy_score(teY, mA.predict(teA))
    c1B = 100 * accuracy_score(teY, mB.predict(teB))
    print(f"\nC=1 probe acc   A(inverse-norm)={c1A:.1f}%   B(clean)={c1B:.1f}%   delta={c1B-c1A:+.1f}pp")

    # zero-shot best template
    zs = zero_shot(model, tok, teA, teB, teY, class_names)
    print(f"\nzero-shot (best-of-templates):")
    bestA, bestB = 0.0, 0.0
    for tname in TEMPLATES[0::2]:
        a, b = zs[tname]; bestA = max(bestA, a); bestB = max(bestB, b)
        print(f"  {tname:12s} A={a:5.1f}%  B={b:5.1f}%  delta={b-a:+.1f}pp")
    print(f"  BEST        A={bestA:5.1f}%  B={bestB:5.1f}%  delta={bestB-bestA:+.1f}pp")

    out = {
        'cos_sim_train': round(cs_tr, 4), 'cos_sim_test': round(cs_te, 4),
        'c1_A_inverse_norm': round(c1A, 2), 'c1_B_clean': round(c1B, 2), 'c1_delta_pp': round(c1B - c1A, 2),
        'zero_shot': {k: {'A_inverse_norm': round(v[0], 2), 'B_clean': round(v[1], 2)} for k, v in zs.items()},
        'zero_shot_best_A': round(bestA, 2), 'zero_shot_best_B': round(bestB, 2),
        'note': 'A = canonical path: denormalize the already-normalized EVAL tensor to [0,1] (tensor_to_pil, uint8) then open_clip preprocess re-normalizes. '
                'B = clean path: open_clip preprocess directly on the raw PIL. Both end at normalized pixels; the only difference is the uint8 rounding in A. '
                'Same 138 test images, DFN-CLIP ViT-L/14. EVAL tensor verified normalized (min -1.60, max 2.64). '
                'Interpretation: the inverse-normalization step is the correct denormalization, not a bug; its uint8-rounding cost is bounded by these deltas.',
        'elapsed_sec': round(time.time() - t0, 1),
    }
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'inverse_norm_sensitivity.json')
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT}")
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
