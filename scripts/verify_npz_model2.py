"""Identify the source model of clip_features_clean.npz by feature cosine,
loading TPDD test images directly (no dependency on the flaky dataset module)."""
import os, json, numpy as np, torch
from PIL import Image
from torchvision import transforms
import open_clip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = ROOT
TEST = ROOT + '/data/tpdd/test'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tf = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
labels = json.load(open(os.path.join(TEST, 'labels.json')))
imgs = [tf(Image.open(os.path.join(TEST, it['image'])).convert('RGB')) for it in labels]
X = torch.stack(imgs).to(DEVICE)          # (138,3,224,224)
print(f'test images: {X.shape}, labels[:6]={[it["label"] for it in labels[:6]]}')

npz = np.load(os.path.join(BASE, 'data', 'clip_features_clean.npz'))
ref = npz['test_feats'].astype('float32')
ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
ref = torch.from_numpy(ref).to(DEVICE)

CKPTS = {
    'DFN-CLIP':  'checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin',
    'LAION-2B':  'checkpoints/clip_vitl14_laion2b/open_clip_pytorch_model.bin',
}
for name, ckpt in CKPTS.items():
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ckpt)
    model = model.eval().to(DEVICE)
    with torch.no_grad():
        f = model.encode_image(X)
        f = f / f.norm(dim=-1, keepdim=True)
    cos = (f * ref).sum(dim=1)
    print(f'{name:10s}: mean cosine={cos.mean().item():.6f}  median={cos.median().item():.6f}  min={cos.min().item():.4f}')
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
