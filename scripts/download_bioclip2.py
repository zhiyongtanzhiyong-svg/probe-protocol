"""
Download BioCLIP 2 (imageomics/bioclip-2, ViT-L/14, open_clip format) via HF mirror.
Mirrors the pattern of download_2ndvlm.py (HF_ENDPOINT = hf-mirror.com).
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

DEST = "checkpoints/bioclip2"
os.makedirs(DEST, exist_ok=True)

FILES = [
    "open_clip_pytorch_model.bin",
    "open_clip_config.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

for f in FILES:
    p = hf_hub_download("imageomics/bioclip-2", f, local_dir=DEST)
    print(f"OK {p}  ({os.path.getsize(p)/1e6:.1f} MB)")
print("DONE: BioCLIP 2 downloaded to", DEST)
