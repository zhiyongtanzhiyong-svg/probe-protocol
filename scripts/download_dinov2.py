"""Download base DINOv2 ViT-L/14 (native .pth) via HF mirror (facebook/ is not proxied)."""
import os, urllib.request

DEST = "checkpoints/dinov2"
os.makedirs(DEST, exist_ok=True)
SRC = "https://hf-mirror.com/hdtech/dinov2_vitl14_pretrain/resolve/main/dinov2_vitl14_pretrain.pth"
DEST_P = os.path.join(DEST, "dinov2_vitl14_pretrain.pth")

_last = [0]
def hook(a, b, t):
    cur = a * b
    if cur - _last[0] > 50_000_000 or (t > 0 and cur >= t):
        _last[0] = cur
        print("\r%.0f/%.0f MB" % (cur / 1e6, max(t, 1) / 1e6), end="", flush=True)

print("downloading", SRC, flush=True)
urllib.request.urlretrieve(SRC, DEST_P, reporthook=hook)
print("\nDONE", DEST_P, round(os.path.getsize(DEST_P) / 1e6, 1), "MB", flush=True)
