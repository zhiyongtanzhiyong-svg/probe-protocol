#!/usr/bin/env python3
"""
generate_figures.py — SINGLE source of truth for every figure in probe-protocol.tex,
regenerated from experiment artifacts.

Run once and it (re)generates all four figures into manuscript/paper/figures/:

  fig_fewshot_curve.pdf   Fig 1: few-shot accuracy vs K                              [CPU]
  fig_perclass.pdf        Fig 2: per-class recall, CLIP probe vs ResNet-18          [CPU]
  fig_confusion.pdf       Fig 3: confusion matrices, CLIP probe + ResNet-18 (seed 0) [CPU]
  fig_feature_space.pdf   Fig 4: t-SNE of frozen CLIP test features                  [CPU]

Usage (run from the repository root; the script also self-locates via --root):
  python3 scripts/generate_figures.py                          # all four
  python3 scripts/generate_figures.py --only fewshot_curve perclass
  python3 scripts/generate_figures.py --no-gpu                 # legacy alias: skip confusion figure (no longer GPU-dependent)
  python3 scripts/generate_figures.py --transparent            # transparent PNG output
  python3 scripts/generate_figures.py --formats pdf svg png    # choose output formats
  python3 scripts/generate_figures.py --root /path/to/repo     # locate data/ and output tree

Publication style (see STYLE below): sans-serif >= 8 pt, 1.0-1.4 pt strokes, no
top/right spines, no gradients/shadows/3-D, 300 dpi raster, vector PDF+SVG with
editable text. Every series is redundantly encoded (hue + line style + marker)
so the figures survive greyscale printing; bar plots additionally use hatching.

Required experiment artifacts (expected under data/):
  clip_fewshot.json, laion_fewshot_canonical.json, paired_kshot_a4.json
  p3b_resnet_kshot_matched{_k10}.json                          -> Fig 1a (Table 1: DFN-CLIP, LAION-2B, ResNet-18; K=1 seed-paired, K=5/10 matched-budget)
  lp_plus_plus.json, coop_official.json                       -> Fig 1b (Table 3)
  multiplicity_16_matched_corrected.json                      -> Fig 1a significance (executed 16-contrast, matched-budget family; a shown contrast is starred only if it survives Holm--Bonferroni there)
  canonical_fig_artifacts.json                                -> Fig 2 and Fig 3
  clip_features_clean.npz                                     -> Fig 4

All plotted experiment results are read from the artifacts above. Per-class
recall and Wilson intervals are derived from the saved integer confusion counts.

Notes:
  * Full-data points in Fig 1 are the clean deterministic protocol values (CLIP 76.8%,
    ResNet 87.7%); error bars are 1 SD across the 15 seeds. Fig 1a additionally marks
    the validation-selected-C operating points (DFN 86.2%, LAION 87.7%).
   * Fig 1a significance markers are Holm-Bonferroni corrected within the executed
     sixteen-contrast, matched-budget family (data/multiplicity_16_matched_
     corrected.json); the star tier uses each contrast's two-sided paired-t p (p_two).
     The K=1 LAION-2B frozen contrast does not survive that family (a tie, raw
     p=0.85), so it is left unstarred. K=1 markers are omitted because the plotted
     ResNet point is the legacy baseline while the corrected test uses the
     matched-budget baseline; K=1 inference remains in the table/text.
  * Fig 1b uses the seed-paired support draw (seed+500) of the adaptation controls, whose
    own probe means are 30.5/49.1/57.7 (DFN) and 40.2/60.2/66.5 (LAION).
  * Fig 3's CLIP probe is the C=1 logistic probe on the frozen CLIP features; ResNet-18 is
    the representative seed-0, best-val checkpoint (30 epochs). Class order = sorted names.
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in scripts/)
DATA = os.path.join(ROOT, "data")
OUT = os.path.normpath(os.path.join(ROOT, "manuscript", "paper", "figures"))

# ---------------------------------------------------------------------------
# STYLE — journal-grade, sans-serif, greyscale-safe
# ---------------------------------------------------------------------------
# Hue is never the only channel: every line series also carries a distinct line
# style and marker, and every bar series carries a hatch.
# Sans-serif preference order. Shared by the rcParams below and by the mathtext
# resolver so regular text and math glyphs always use the SAME resolved face.
SANS_CANDIDATES = ["Helvetica Neue", "Helvetica", "Arial",
                   "Liberation Sans", "DejaVu Sans"]

C_RESNET = "#000000"   # CNN baseline: pure black (0% luminance)
C_DFN = "#0072B2"      # Okabe-Ito blue
C_LAION = "#D55E00"    # Okabe-Ito vermillion
GRID = "#BFBFBF"

# 16-class qualitative palette (Okabe-Ito 8 + 8 extended), ordered so that
# neighbouring entries differ strongly in luminance as well as hue.
CAT16 = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
    "#7B3294", "#C77CFF", "#008080", "#8B4513",
    "#A6CEE3", "#FB9A99", "#B2DF8A", "#6A3D9A",
]
MARKERS16 = ["o", "s", "^", "D", "v", "P", "X", "*",
             "<", ">", "h", "d", "8", "p", "H", "1"]


def apply_style():
    """Global rcParams. Sans-serif, >= 8 pt, 1.0-1.4 pt strokes, vector-friendly."""
    # Resolve the sans family to a concrete installed font so math and text use
    # one typeface. Must run against SANS_CANDIDATES (not the pre-update
    # rcParams default list, which starts with DejaVu Sans); newer matplotlib
    # also rejects the generic "sans-serif" alias in findfont/mathtext.
    _sans = "DejaVu Sans"
    for _name in SANS_CANDIDATES:
        try:
            _path = font_manager.findfont(font_manager.FontProperties(family=_name),
                                          fallback_to_default=False)
        except ValueError:
            continue
        _fam = font_manager.get_font(_path).family_name
        if _fam.lower() == _name.lower():
            _sans = _fam
            break
    matplotlib.rcParams.update({
        # --- typography: sans-serif (Helvetica -> Arial -> DejaVu Sans) ---
        "font.family": "sans-serif",
        "font.sans-serif": SANS_CANDIDATES,
        # custom mathtext on the same resolved sans family as the text, so math
        # glyphs ($n$, $C{=}1$) no longer mix a second typeface (DejaVu) into
        # otherwise-Arial figures
        "mathtext.fontset": "custom",
        "mathtext.rm": _sans,
        "mathtext.it": f"{_sans}:italic",
        "mathtext.bf": f"{_sans}:bold",
        "font.size": 8.0,          # hard floor for the whole figure
        "axes.titlesize": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 9.0,
        # --- strokes: 1.0 pt (axes/grid furniture) to 1.4 pt (data) ---
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.4,
        "lines.markersize": 5.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "grid.linewidth": 0.5,
        "grid.color": GRID,
        "grid.alpha": 0.45,
        "axes.axisbelow": True,
        # --- output: 300 dpi raster, vector with editable text ---
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,        # TrueType: text stays selectable/editable
        "ps.fonttype": 42,
        "svg.fonttype": "none",    # keep SVG text as text, not outlines
        "svg.hashsalt": "paper2",  # deterministic SVG output
        "path.simplify": False,
    })


def save(fig, stem, formats=("pdf", "svg", "png"), transparent=False):
    """Write only the requested formats; PNG uses the requested transparency."""
    os.makedirs(OUT, exist_ok=True)
    written = []
    for ext in dict.fromkeys(formats):
        p = os.path.join(OUT, f"{stem}.{ext}")
        options = {"dpi": 300, "transparent": transparent} if ext == "png" else {}
        fig.savefig(p, format=ext, **options)
        written.append(p)
    plt.close(fig)
    return written


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def class_names(short=False):
    with open(os.path.join(DATA, "tpd_to_kg_map.json"), encoding="utf-8") as f:
        m = json.load(f)
    names = [n.replace("_", " ").title() for n in sorted(m.keys())]
    if short:
        names = [SHORT.get(n, n) for n in names]
    return names


SHORT = {
    "Anthracnose": "Anthracnose",
    "Black Shank": "Black shank",
    "Brown Spot": "Brown spot",
    "Cmv": "CMV",
    "Frog Eye": "Frog eye",
    "Genetic Abnormality": "Genetic",
    "Healthy": "Healthy",
    "Nematodes": "Nematodes",
    "Potato Tuber Moth": "PTM",
    "Pvy": "PVY",
    "Sunscald": "Sunscald",
    "Target Spot": "Target spot",
    "Tmv": "TMV",
    "Tswv": "TSWV",
    "Weather Fleck": "Weather",
    "Wildfire": "Wildfire",
}


def _load(name):
    path = os.path.join(DATA, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required experiment artifact not found: {path}. "
            "Set --root to the repository containing the data/ directory.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def canonical_artifacts():
    """Predictions / per-class accuracies behind Figures 2 and 3.

    Built by scripts/canonical_fig_artifacts.py under the paper's canonical
    protocol, which reproduces the reported Table 2 numbers exactly (CLIP C=1
    0.7681). These MUST be used instead of recomputing: refitting on
    data/clip_features_clean.npz yields 0.7609, because that .npz is a different
    feature set from the one behind the canonical results.
    """
    path = os.path.join(DATA, "canonical_fig_artifacts.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\nRun:  python3 scripts/canonical_fig_artifacts.py")
    a = _load("canonical_fig_artifacts.json")
    for k in ("clip", "resnet"):
        if k not in a:
            raise SystemExit(
                f"{path} has no '{k}' block.\n"
                f"Run:  python3 scripts/canonical_fig_artifacts.py{'' if k == 'clip' else '  # drop --clip-only'}")
    return a


def stars(p):
    """Holm-Bonferroni corrected p -> significance marker."""
    if p is None or not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _laion_c1():
    """Default-C (C=1) test accuracy of the LAION-2B probe, in percent."""
    cb = _load("canonical_results.json").get("clip_by_checkpoint", {})
    for key, blk in cb.items():
        if key.startswith("laion") and isinstance(blk, dict):
            acc = blk.get("linear_C1", {}).get("accuracy")
            if acc is not None:
                return acc * 100
    raise KeyError("LAION C=1 accuracy not found in canonical_results.json")


def _laion_bestC():
    """(accuracy in percent, best C) for the LAION-2B checkpoint.

    canonical_fig_artifacts.json holds only the DFN probe (the one the linear
    panels draw), so the LAION number is read from canonical_results.json, which
    carries both checkpoints. Read from data so the figure cannot drift from the
    experiments.
    """
    cb = _load("canonical_results.json").get("clip_by_checkpoint", {})
    for key, blk in cb.items():
        if key.startswith("laion") and isinstance(blk, dict):
            acc = blk.get("linear_bestC", {}).get("accuracy")
            if acc is not None:
                return acc * 100, blk.get("best_C")
    raise KeyError("LAION validation-selected-C accuracy not found in canonical_results.json")


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (better than Wald at small n)."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / d), min(1.0, (centre + half) / d)


def style_bar_errors(container, lw=1.0):
    """ax.bar(..., yerr=) does not accept elinewidth/capthick as kwargs; they must
    be set on the ErrorbarContainer it returns. Best-effort across matplotlib
    versions."""
    eb = getattr(container, "errorbar", None)
    if eb is None:
        return
    for line in getattr(eb, "lines", ()):
        for setter in ("set_linewidth", "set_markeredgewidth"):
            try:
                getattr(line, setter)(lw)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fig 1: few-shot accuracy vs K   (CPU)
# ---------------------------------------------------------------------------
def fig_fewshot_curve(formats=("pdf", "svg"), transparent=False):
    """Two panels: (a) frozen-feature linear probes vs fine-tuned CNN, with
    multiplicity-corrected significance; (b) head-capacity controls (LP++, CoOp)."""
    dfn = _load("clip_fewshot.json")
    laion = _load("laion_fewshot_canonical.json")
    pk = _load("paired_kshot_a4.json")
    p3b_r5 = _load("p3b_resnet_kshot_matched.json")
    p3b_r10 = _load("p3b_resnet_kshot_matched_k10.json")
    lpp = _load("lp_plus_plus.json")["checkpoints"]
    coop = _load("coop_official.json")["checkpoints"]
    mult = {r["name"]: r for r in _load("multiplicity_16_matched_corrected.json")["contrasts"]}
    # Canonical single source of truth for every full-data number. Nothing below
    # is hard-coded: re-running the experiments and regenerating these JSONs
    # changes the figure, so a stale constant cannot silently reappear.
    art = _load("canonical_fig_artifacts.json")
    rn_full_mean = art["resnet"]["mean"] * 100
    rn_full_sd = art["resnet"]["std"] * 100
    rn_full_seed0 = art["resnet"]["seed0"]["accuracy"] * 100

    KLAB = ["K=1", "K=5", "K=10", "Full"]
    x = np.arange(len(KLAB))

    # ---- (a) probes + CNN ------------------------------------------------
    # NOTE: every store read here -- clip_fewshot.json, laion_fewshot_canonical.json,
    # paired_kshot_a4.json, p3b_resnet_kshot_matched{_k10}.json -- stores
    # FRACTIONS (0.3208 == 32.1%), so the `* 100` scaling below is mandatory.
    # The previous version of this figure plotted the CNN baseline at 0.3%.
    # ResNet-18 reference per the executed multiplicity family (p3b): K=1 is the
    # seed-paired point (paired_kshot_a4.json), K=5/K=10 are the matched-budget
    # (hyperparameter-searched, 15-seed) P3b points. This is the same split the
    # tex Table 1 reports (33.0 / 64.9 / 74.6 / 87.7) and the same ResNet
    # reference each p3b contrast is tested against.
    rn_v = [pk["K=1"]["resnet_mean"] * 100,
            p3b_r5["stage2_final"]["mean"] * 100,
            p3b_r10["stage2_final"]["mean"] * 100, rn_full_mean]
    rn_sd = [pk["K=1"]["resnet_std"] * 100,
             p3b_r5["stage2_final"]["std"] * 100,
             p3b_r10["stage2_final"]["std"] * 100, rn_full_sd]
    # Full-data default-probe points are read, not written: the two checkpoints
    # sit at 76.8% and 76.1%, and hard-coding 76.8 for both overstated LAION by
    # 0.7 pp while making the script's own "nothing is hard-coded" claim false.
    dfn_c1 = art["clip"]["linear_C1"]["accuracy"] * 100
    la_c1 = _laion_c1()
    dfn_v = [dfn["K=1"]["mean"] * 100, dfn["K=5"]["mean"] * 100,
             dfn["K=10"]["mean"] * 100, dfn_c1]
    dfn_sd = [dfn["K=1"]["std"] * 100, dfn["K=5"]["std"] * 100,
              dfn["K=10"]["std"] * 100, 0.0]
    la_v = [laion["K=1"]["mean"] * 100, laion["K=5"]["mean"] * 100,
            laion["K=10"]["mean"] * 100, la_c1]
    la_sd = [laion["K=1"]["std"] * 100, laion["K=5"]["std"] * 100,
             laion["K=10"]["std"] * 100, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))

    ax = axes[0]
    series = [
        (rn_v, rn_sd, dict(color=C_RESNET, ls="-", marker="o", ms=5.5,
                           label="ResNet-18 (fine-tuned)", zorder=5), "ResNet"),
        (la_v, la_sd, dict(color=C_LAION, ls=(0, (5, 1.6)), marker="^", ms=5.5,
                           label="CLIP probe (LAION-2B)", zorder=4), "LAION"),
        # open marker: at full data the two default probes sit within 0.8 pp of
        # each other (76.8 / 76.1), so a filled marker would hide one of them
        (dfn_v, dfn_sd, dict(color=C_DFN, ls=(0, (5, 1.6, 1, 1.6)), marker="s", ms=5.5,
                             mfc="none", mew=1.3,
                             label="CLIP probe (DFN-CLIP)", zorder=3), "DFN"),
    ]
    for v, sd, kw, _ in series:
        ax.errorbar(x, v, yerr=sd, elinewidth=1.1, capsize=2.5, capthick=1.0, **kw)
    # validation-selected-C operating points (full data only)
    dfn_bestC = art["clip"]["linear_bestC"]["accuracy"] * 100
    laion_bestC, laion_C = _laion_bestC()
    ax.plot([3], [dfn_bestC], marker="s", ms=7, mfc="none", mew=1.4, ls="none",
            color=C_DFN, zorder=6)
    ax.plot([3], [laion_bestC], marker="^", ms=7.5, mfc="none", mew=1.4, ls="none",
            color=C_LAION, zorder=6)
    lead = dict(arrowstyle="-", lw=0.8, color="0.35", shrinkA=1, shrinkB=4)
    ax.annotate(f"DFN-CLIP, $C{{=}}{art['clip']['best_C']:.0f}$: {dfn_bestC:.1f}%",
                (3, dfn_bestC), xytext=(0.35, rn_full_mean + 6.5),
                textcoords="data", fontsize=8, ha="left", va="center",
                arrowprops=lead, zorder=7)
    ax.annotate(f"LAION-2B, $C{{=}}{laion_C:.0f}$: {laion_bestC:.1f}%",
                (3, laion_bestC), xytext=(0.35, rn_full_mean - 7.0),
                textcoords="data", fontsize=8, ha="left", va="center",
                arrowprops=lead, zorder=7)

    # Holm-Bonferroni-corrected significance for the probe-vs-CNN contrasts.
    # Both the multiplicity gate and the star tier come from the executed
    # sixteen-contrast, matched-budget family (data/multiplicity_16_matched_
    # corrected.json): a shown contrast is starred only if it survives Holm--
    # Bonferroni in that family, and the star tier uses its two-sided paired-t
    # p-value (p_two). The K=1 ResNet point on this panel is the legacy single-
    # config baseline (matching Table 1); the K=1 LAION-2B contrast does NOT
    # survive the matched sixteen-contrast family (a tie, raw p=0.85), so it is
    # deliberately left unstarred.
    _tag_short = {"LAION-2B": "LAION", "DFN-CLIP": "DFN"}
    for i, K in enumerate(["K=1", "K=5", "K=10"]):
        # The plotted K=1 ResNet point is the legacy single-config result,
        # whereas the executed Holm family tests against the matched-budget
        # K=1 ResNet result. Do not attach that test marker to a different point.
        if K == "K=1":
            continue
        for tag, vals, sds in (("LAION-2B", la_v, la_sd), ("DFN-CLIP", dfn_v, dfn_sd)):
            r = mult.get(f"{K} {_tag_short[tag]} frozen")
            if r is None or not r.get("survive"):
                continue
            ax.text(i, vals[i] + sds[i] + 2.2, stars(r["p_two"]),
                    ha="center", va="bottom", fontsize=9, color="black", zorder=8)

    ax.set_title("(a)  Frozen-feature linear probe vs fine-tuned CNN", loc="left")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 104)
    ax.set_xlim(-0.35, 3.35)
    ax.set_xticks(x)
    ax.set_xticklabels(KLAB)
    ax.grid(axis="y")
    despine(ax)
    ax.legend(frameon=False, loc="lower right", handlelength=2.4,
              borderaxespad=0.2, labelspacing=0.35)

    # ---- (b) head-capacity controls -------------------------------------
    ax = axes[1]
    def pull(src, key, ck, K, full):
        node = src[ck][K][key]
        return node["mean"] * 100, (0.0 if full else node["std"] * 100)

    b_series = [
        (rn_v, rn_sd, dict(color=C_RESNET, ls="-", marker="o", ms=4.5, lw=1.1,
                           alpha=0.5, label="ResNet-18 (fine-tuned)"), None),
    ]
    for ck, col, mk, base, lwid in (("dfn_clip", C_DFN, "s", "DFN-CLIP", 1.3),
                                    ("laion2b", C_LAION, "^", "LAION-2B", 2.1)):
        for src, key, lbl, ls, filled in (
                (lpp, "lp_pp", "LP++", (0, (1, 1.5)), False),
                (coop, "coop_official", "CoOp", (0, (6, 1.8)), True)):
            v, sd = [], []
            for K in ["K=1", "K=5", "K=10"]:
                m, s = pull(src, key, ck, K, False)
                v.append(m)
                sd.append(s)
            m, _ = pull(src, key, ck, "full", True)
            v.append(m)
            sd.append(0.0)
            # Three redundant channels separate the two checkpoints in greyscale,
            # where C_DFN (L*~35%) and C_LAION (L*~45%) are too close in luminance:
            # marker shape (square vs triangle), line style (dotted vs long-dashed)
            # and stroke width (1.3 vs 2.1 pt).
            # White marker edge: the four control series converge at K=10/Full
            # (78-87%), so without it a filled marker occludes the one behind it
            # and the panels become unreadable in greyscale.
            kw = dict(color=col, ls=ls, marker=mk, ms=5.6, lw=lwid, zorder=4,
                      mec="white", mew=0.7, label=f"{base} {lbl}")
            if not filled:
                kw.update(mfc="none", mew=1.3)
            b_series.append((v, sd, kw, f"{base} {lbl}"))

    for v, sd, kw, lbl in b_series:
        ax.errorbar(x, v, yerr=sd, elinewidth=1.1, capsize=2.5, capthick=1.0, **kw)

    ax.set_title("(b)  Head-capacity controls on frozen CLIP features", loc="left")
    ax.set_xlabel("Support images per class (K)")
    ax.set_ylim(0, 104)
    ax.set_xlim(-0.35, 3.35)
    ax.set_xticks(x)
    ax.set_xticklabels(KLAB)
    ax.grid(axis="y")
    despine(ax)
    ax.legend(frameon=False, loc="lower right", handlelength=2.4,
              borderaxespad=0.2, labelspacing=0.35, ncol=1)

    fig.tight_layout(w_pad=1.6)
    out = save(fig, "fig_fewshot_curve", formats, transparent)
    print(f"  (a) DFN={[round(v, 1) for v in dfn_v]}  LAION={[round(v, 1) for v in la_v]}"
          f"  ResNet={[round(v, 1) for v in rn_v]}")
    for ck in ("dfn_clip", "laion2b"):
        print(f"  (b) {ck}: LP++ full={lpp[ck]['full']['lp_pp']['mean']*100:.1f}%  "
              f"CoOp full={coop[ck]['full']['coop_official']['mean']*100:.1f}%")
    return out


# ---------------------------------------------------------------------------
# Fig 2: per-class accuracy  (CPU)
# ---------------------------------------------------------------------------
def fig_perclass(formats=("pdf", "svg"), transparent=False):
    """Compare per-class recall with each model's uncertainty source labeled.

    CLIP uses Wilson intervals from exact confusion-matrix counts. ResNet uses
    the per-class mean and SD across 15 training seeds. These summaries describe
    different sources of variation and are not directly comparable intervals.
    """
    art = canonical_artifacts()
    names = class_names(short=True)
    cm_clip = np.asarray(art["clip"]["linear_C1"]["confusion"], dtype=int)
    if cm_clip.shape != (len(names), len(names)):
        raise ValueError("CLIP confusion matrix does not match the 16-class label order")
    counts = cm_clip.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("CLIP confusion matrix contains a class with zero test support")
    clip_correct = np.diag(cm_clip)
    clip = 100.0 * clip_correct / counts
    # ResNet per-class is the 15-seed MEAN -- the statistic Table 4 reports
    # ("a single run is not a stable basis", limitation (2)). Plotting seed0
    # here previously contradicted Table 4 (e.g. class 0: 100% vs 6.7%).
    # perclass_mean / perclass_sd are stored as FRACTIONS (0.067 == 6.7%), so
    # the *100 is mandatory for the percent axis (same unit trap as Fig 1a).
    resnet = np.array(art["resnet"]["perclass_mean"], dtype=float) * 100.0
    resnet_sd = np.array(art["resnet"]["perclass_sd"], dtype=float) * 100.0

    ci = [wilson(int(k), int(n)) for k, n in zip(clip_correct, counts)]
    el_c = clip - np.array([lo for lo, _ in ci]) * 100
    eu_c = np.array([hi for _, hi in ci]) * 100 - clip
    el_r = resnet_sd                 # ResNet 15-seed mean -> symmetric 15-seed SD
    eu_r = resnet_sd.copy()

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ec = dict(ecolor="0.30", elinewidth=0.9, capsize=2.0, capthick=0.9)
    bc = ax.bar(x - w / 2, clip, w, yerr=[el_c, eu_c], color=C_DFN, hatch="///",
                edgecolor="white", linewidth=0.0,
                label="CLIP C=1 (Wilson 95% CI)", error_kw=ec)
    br = ax.bar(x + w / 2, resnet, w, yerr=[el_r, eu_r], color=C_LAION, hatch="\\\\\\",
                edgecolor="white", linewidth=0.0,
                 label="ResNet-18 (15-seed mean ± 1 SD)", error_kw=ec)
    style_bar_errors(bc)
    style_bar_errors(br)

    ax.set_title(f"Per-class recall on the {counts.sum()}-image TPDD test set\n"
                 "CLIP: Wilson 95% CI; ResNet: 15-seed mean ± 1 SD; class support shown below",
                 loc="left")
    ax.set_xticks(x)
    # single-line labels: two-line stacked labels collided between neighbours
    # under 40-degree rotation (verified against the PDF text layer)
    ax.set_xticklabels([f"{n} ($n$={int(c)})" for n, c in zip(names, counts)],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Per-class recall (%)")
    ax.set_ylim(0, 122)
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.grid(axis="y")
    despine(ax)
    ax.legend(frameon=False, loc="upper right", ncol=2, columnspacing=1.2,
              handlelength=1.8)
    fig.tight_layout()
    out = save(fig, "fig_perclass", formats, transparent)
    diff = sum(1 for a, b in zip(clip, resnet) if abs(a - b) > 1e-9)
    print(f"  {len(names)} classes; CLIP mean={np.mean(clip):.1f}%, "
          f"ResNet mean={np.mean(resnet):.1f}%; {diff}/{len(names)} classes differ")
    return out


# ---------------------------------------------------------------------------
# Fig 3: confusion matrices  (CPU; uses saved predictions)
# ---------------------------------------------------------------------------
def fig_confusion(formats=("pdf", "svg"), transparent=False):
    """Confusion matrices from the canonical artifacts (no GPU, no refit).

    Panel (a) is the frozen CLIP linear probe at C=1 and panel (b) the
    fine-tuned ResNet-18 seed-0 run, both produced by
    scripts/canonical_fig_artifacts.py, so the accuracies printed here are the
    same runs that Table 2 reports.
    """
    art = canonical_artifacts()
    cm_clip = np.array(art["clip"]["linear_C1"]["confusion"])
    cm_rn = np.array(art["resnet"]["seed0"]["confusion"])
    acc_clip = art["clip"]["linear_C1"]["accuracy"]
    acc_rn = art["resnet"]["seed0"]["accuracy"]
    NFC = cm_clip.shape[0]
    print(f"  CLIP probe acc: {acc_clip*100:.1f}%   ResNet seed-0 acc: {acc_rn*100:.1f}%")
    print(f"  ResNet 15-seed mean: {art['resnet']['mean']*100:.2f}% "
          f"+/- {art['resnet']['std']*100:.2f}")

    names = class_names(short=True)
    # Greyscale-safe sequential ramp: monotonic in luminance, so a B&W print of
    # this figure loses no information.
    cmap = LinearSegmentedColormap.from_list(
        "seq_blue", ["#FFFFFF", "#9EC5E8", "#4A90C4", "#1F5C8B", "#0B2E4F"])
    cmap.set_bad("#F0F0F0")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.9))
    im = None
    for ax, cm, tag, acc in zip(
            axes, [cm_clip, cm_rn],
            ["(a)  CLIP full-data probe ($C{=}1$)",
             "(b)  ResNet-18 full-data (seed 0)"],
            [acc_clip, acc_rn]):
        row = cm.sum(axis=1, keepdims=True).clip(min=1)
        cmn = cm / row
        im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(NFC))
        ax.set_yticks(range(NFC))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{tag}   {acc*100:.1f}%", loc="left")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.0)
            s.set_color("#000000")
        for i in range(NFC):
            for j in range(NFC):
                if cm[i, j] == 0:
                    continue
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        fontsize=8, fontweight="bold" if cmn[i, j] > 0.5 else "normal",
                        color="white" if cmn[i, j] > 0.5 else "black")
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    pos = axes[0].get_position()
    cbar_ax = fig.add_axes([0.90, pos.y0, 0.025, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Row-normalised rate (recall)", fontsize=9)
    cbar.ax.tick_params(labelsize=8, width=1.0, length=3.0)
    cbar.outline.set_linewidth(1.0)
    out = save(fig, "fig_confusion", formats, transparent)
    return out


# ---------------------------------------------------------------------------
# Fig 4: t-SNE feature space  (CPU)
# ---------------------------------------------------------------------------
def fig_feature_space(formats=("pdf", "svg"), transparent=False):
    from sklearn.manifold import TSNE

    emb = np.load(os.path.join(DATA, "clip_features_clean.npz"))
    ef, el = emb["test_feats"], emb["test_labels"]
    names = class_names(short=True)
    NFC = len(names)

    kwargs = dict(n_components=2, perplexity=30, init="pca", random_state=42)
    try:
        tsne = TSNE(max_iter=1000, **kwargs)      # scikit-learn >= 1.5
    except TypeError:
        tsne = TSNE(n_iter=1000, **kwargs)        # scikit-learn < 1.5
    coords = tsne.fit_transform(ef)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    # Hue alone cannot carry 16 classes in greyscale, so each class also gets its
    # own marker shape; the legend is the authoritative key. Per-point text labels
    # were tried and removed: with 16 classes the centroids collide and the labels
    # obscure the points they annotate.
    for i, name in enumerate(names):
        m = el == i
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=34,
                   c=CAT16[i], marker=MARKERS16[i],
                   edgecolors="white", linewidths=0.5, alpha=1.0, zorder=3)

    ax.set_title("t-SNE of frozen CLIP (DFN-CLIP) test features\n"
                 "138 test images, 16 classes (marker shape and colour both encode class)",
                 loc="left")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)
        s.set_color("#000000")
    handles = [Line2D([], [], color=CAT16[i], marker=MARKERS16[i], ls="none",
                      markersize=5.5, markeredgecolor="white", markeredgewidth=0.5,
                      label=f"{names[i]} ({int((el == i).sum())})")
               for i in range(NFC) if (el == i).any()]
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, labelspacing=0.42, borderaxespad=0.0,
              markerscale=0.9, handletextpad=0.5, title="Class ($n$)",
              title_fontsize=8)
    fig.tight_layout()
    out = save(fig, "fig_feature_space", formats, transparent)
    print(f"  t-SNE on {len(ef)} test features, {NFC} classes")
    return out


JOBS = {
    "fewshot_curve": fig_fewshot_curve,
    "perclass": fig_perclass,
    "confusion": fig_confusion,
    "feature_space": fig_feature_space,
}


def main():
    ap = argparse.ArgumentParser(
        description="Generate all paper figures into manuscript/paper/figures/")
    ap.add_argument("--only", nargs="*", choices=list(JOBS.keys()), default=None)
    ap.add_argument("--root", help="repository root containing data/ (default: script directory)")
    ap.add_argument("--data-dir", help="override the experiment data directory")
    ap.add_argument("--output-dir", help="override the figure output directory")
    ap.add_argument("--no-gpu", action="store_true",
                    help="legacy alias: skips the confusion-matrix figure (kept for compatibility; "
                         "the current pipeline computes it from stored predictions and needs no GPU)")
    ap.add_argument("--transparent", action="store_true",
                    help="PNG with transparent background")
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"],
                    choices=["pdf", "svg", "png"], help="formats to emit")
    args = ap.parse_args()
    global ROOT, DATA, OUT
    ROOT = os.path.abspath(args.root) if args.root else ROOT
    DATA = os.path.abspath(args.data_dir) if args.data_dir else os.path.join(ROOT, "data")
    OUT = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(
        ROOT, "manuscript", "paper", "figures")
    jobs = args.only or list(JOBS.keys())
    if args.no_gpu:
        jobs = [j for j in jobs if j != "confusion"]
    formats = tuple(dict.fromkeys(args.formats))
    os.makedirs(OUT, exist_ok=True)
    apply_style()
    print(f"style: {matplotlib.rcParams['font.family']} "
          f"({matplotlib.rcParams['font.sans-serif'][0]}), "
          f"base {matplotlib.rcParams['font.size']}pt, "
          f"raster {matplotlib.rcParams['savefig.dpi']} dpi, formats {formats}")
    skipped = []
    for j in jobs:
        print(f"=== {j} ===")
        try:
            JOBS[j](formats=formats, transparent=args.transparent)
        except FileNotFoundError as e:
            skipped.append(j)
            print(f"SKIP {j}: missing artifact {e.filename} (large feature stores are not in the"
                  f" public repository; obtainable from the authors). Other figures continue.")
    print(f"\nFigures written to: {OUT}" + (f" (skipped: {', '.join(skipped)})" if skipped else ""))
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".pdf", ".svg", ".png")):
            print(f"  {f}  ({os.path.getsize(os.path.join(OUT, f))//1024} KB)")


if __name__ == "__main__":
    main()
