#!/usr/bin/env python3
"""Appendix C figure: stage-1 validation surfaces of the two matched 9-config
fine-tuning grids on TPDD at K=5 (mean of the 3 selection seeds).

Reads the stage-1 search results recorded at run time (no re-fitting, no
hard-coded result constants) and renders two 3x3 panels:
  left  = CLIP ViT-L/14 (LAION-2B) vision-encoder fine-tune
  right = ResNet-18
Each cell is a (learning rate, epochs) configuration; the starred cell is the
validation-selected configuration re-run at n=15 in stage 2.

Outputs: <root>/manuscript/paper/figures/fig_grid_plateau.{pdf,png,svg}
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from generate_figures import apply_style, C_DFN, C_LAION, GRID

EPOCHS = [30, 100, 300]


def load_grid(path, lr_key):
    d = json.load(open(path))
    vals = {}
    for e in d["stage1_search"]:
        vals[(e[lr_key], e["epochs"])] = e["mean_best_val"] * 100.0
    sel = d["selected"]
    sel_key = (sel[lr_key], sel["epochs"])
    assert sel_key in vals, f"selected config {sel_key} not in stage1_search"
    return d, vals, sel_key


def draw_panel(ax, vals, lrs, sel_key, title, sel_color):
    lrs = sorted(lrs)
    m = np.full((len(lrs), len(EPOCHS)), np.nan)
    for i, lr in enumerate(lrs):
        for j, ep in enumerate(EPOCHS):
            m[i, j] = vals.get((lr, ep), np.nan)
    im = ax.imshow(m, origin="lower", aspect="auto", cmap="Blues",
                   vmin=40, vmax=100)
    for i, lr in enumerate(lrs):
        for j, ep in enumerate(EPOCHS):
            v = m[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9,
                        color="#666666")
                continue
            is_sel = (lr, ep) == sel_key
            txt = f"{v:.1f}" + ("  *" if is_sel else "")
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10 if is_sel else 9.5,
                    fontweight="bold" if is_sel else "normal",
                    color="#000000")
            lw, ec = (2.5, sel_color) if is_sel else (1.0, GRID)
            ax.add_patch(matplotlib.patches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=False, lw=lw, ec=ec))
    ax.set_xticks(range(len(EPOCHS)))
    ax.set_xticklabels([str(e) for e in EPOCHS])
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([f"{lr:.0e}" if lr < 0.01 else f"{lr:.1e}"
                        for lr in lrs])
    ax.set_xlabel("epochs")
    ax.set_ylabel("learning rate")
    ax.set_title(title, fontsize=10.5)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    outdir = root / "manuscript" / "paper" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    apply_style()

    clip_path = root / "data" / "p3b_clip_finetune_kshot.json"
    rn_path = root / "data" / "p3b_resnet_kshot_matched.json"
    d_clip, vals_clip, sel_clip = load_grid(clip_path, "vis_lr")
    d_rn, vals_rn, sel_rn = load_grid(rn_path, "lr")
    assert d_clip["K"] == d_rn["K"] == 5

    fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=(7.4, 2.9),
                                           dpi=300)
    lrs_clip = sorted({e["vis_lr"] for e in d_clip["stage1_search"]})
    lrs_rn = sorted({e["lr"] for e in d_rn["stage1_search"]})
    im1 = draw_panel(axes[0], vals_clip, lrs_clip, sel_clip,
                     "(a)  CLIP ViT-L/14 (LAION-2B) encoder fine-tune", C_LAION)
    im2 = draw_panel(axes[1], vals_rn, lrs_rn, sel_rn,
                     "(b)  ResNet-18", C_DFN)
    for im in (im1, im2):
        fig.colorbar(im, ax=im.axes, fraction=0.046, pad=0.04)
    fig.text(0.5, 0.005,
             "Cell value = mean stage-1 validation accuracy (%), 3 selection "
             "seeds, 138-image validation split.  * = selected configuration "
             "(thick outline), re-run at n = 15 in stage 2.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    for ext in ("pdf", "png", "svg"):
        fig.savefig(outdir / f"fig_grid_plateau.{ext}", bbox_inches="tight")
    matplotlib.pyplot.close(fig)
    print(f"wrote {outdir}/fig_grid_plateau.{{pdf,png,svg}}")
    print(f"  CLIP selected {sel_clip} -> {vals_clip[sel_clip]:.2f}")
    print(f"  ResNet selected {sel_rn} -> {vals_rn[sel_rn]:.2f}")


if __name__ == "__main__":
    main()
