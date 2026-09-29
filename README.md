# Paper 2 — CLIP vs ResNet-18 on tobacco leaf disease

A frozen large vision-language encoder (CLIP) with a training-light linear head
versus a fully fine-tuned small CNN, on 16-class tobacco leaf disease images.
Full-data accuracy is statistically indistinguishable between the tuned CLIP
probe and fine-tuned ResNet-18 on the held-out test set, but the comparison
reverses by annotation budget: fine-tuned CLIP wins decisively, and at
K=1 with seed-paired support sets the LAION-2B probe leads.

## Layout

| Path | Contents |
|---|---|
| `manuscript/paper/clip_vs_resnet_tobacco.tex` | the manuscript (source of truth for all prose/numbers) |
| `scripts/` | every experiment that produced a number in the paper |
| `data/*.json` | experiment outputs — the single source of truth for all reported numbers |
| `generate_figures.py` | single generator for all four figures |
| `AGENTS.md` | project discipline (read this before writing experiment code) |

## Reproducing

```bash
# regenerate all figures from data/*.json  (no GPU needed)
python3 generate_figures.py

# compile the paper
cd manuscript/paper && pdflatex clip_vs_resnet_tobacco.tex

# verify every number in the .tex against data/*.json  <- must pass before commit
python3 scripts/check_tex_vs_data.py
```

`scripts/check_tex_vs_data.py` derives each headline number from the JSON
artifacts and asserts it appears in the manuscript. It exits non-zero on a stale
or missing value, so a re-run that changes a result cannot silently diverge from
the prose. It also re-derives the Holm–Bonferroni adjusted p-values and fails if
they violate the step-down formula.

## Data provenance

**The image corpora are not in this repository.** This project is
self-contained — it has no runtime dependency on Paper 1 / `anchor-scope`, and
the shared `dataset.py` / `model_zoo.py` modules are vendored under `scripts/`
so the two papers can evolve independently.

| Corpus | Size | How to obtain |
|---|---|---|
| TPDD (`data/tpdd/`) | 52 MB | `TamAko783/TPDD_Honglin_CLS` on the Hugging Face Hub; splits 1295 / 138 / 138, 16 classes |
| TLD (`data/tld_images/`) | 178 MB | `data/tld_*_manifest.json` records the exact file list |
| PlantVillage tomato | 313 MB | public PlantVillage release |

Expected layout: `data/tpdd/{train,validation,test}/<class_name>/*.jpg` plus a
`labels.json` in each split directory (`{"image": ..., "label": "<class>"}`).
`scripts/dataset.py` resolves this via `TPDD_DATA_ROOT`, defaulting to
`<repo>/data/tpdd`.

> **Why this matters.** The original 11 MB low-resolution TPDD images were
> destroyed by an external process on 2026-09-28 and had never been tracked in
> git, so no earlier run could be re-derived exactly. The dataset was re-built
> from the public source at higher resolution, which moved the fine-tuned
> ResNet-18 full-data number from 86.4% to 87.7% (+1.3 pp, 1.8 SEM, Welch
> p=0.15; 86.4% lies inside the current run's 95% interval [86.3, 89.2]) — not
> statistically separable, but no longer exactly reproducible. Paper 1 keeps its
> own copy under git for this reason; keep a dated tarball of any corpus you
> depend on.

## Citation

See `manuscript/paper/clip_vs_resnet_tobacco.tex` for the bibliography.
