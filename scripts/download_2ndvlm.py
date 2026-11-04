"""
E4: Download a second, generic CLIP ViT-L/14 (LAION-2B, standard CLIP training)
via the hf-mirror endpoint. Same architecture & zero-shot protocol as DFN-CLIP,
but different training -> isolates whether the CLIP-vs-ResNet gap is a
paradigm/systematic effect rather than DFN-specific.

Saves the .bin (+ config) into a self-contained local dir so it can be loaded
with open_clip(..., pretrained=<local .bin>), mirroring the DFN-CLIP setup.
"""
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import shutil
from huggingface_hub import hf_hub_download

CKPT_DIR = 'checkpoints/clip_vitl14_laion2b'
REPO = 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
BIN_LOCAL = os.path.join(CKPT_DIR, 'open_clip_pytorch_model.bin')
CFG_LOCAL = os.path.join(CKPT_DIR, 'open_clip_config.json')
os.makedirs(CKPT_DIR, exist_ok=True)

if not (os.path.exists(BIN_LOCAL) and os.path.getsize(BIN_LOCAL) > 100_000_000):
    print('downloading open_clip_pytorch_model.bin ...')
    src = hf_hub_download(repo_id=REPO, filename='open_clip_pytorch_model.bin')
    shutil.copyfile(src, BIN_LOCAL)
else:
    print('bin already present')
src = hf_hub_download(repo_id=REPO, filename='open_clip_config.json')
shutil.copyfile(src, CFG_LOCAL)
print('DONE ->', BIN_LOCAL, os.path.getsize(BIN_LOCAL)//1e6, 'MB')
