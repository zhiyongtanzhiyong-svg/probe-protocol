# CLIP vs ResNet-18 on tobacco leaf disease

Does a frozen CLIP encoder with a training-light head beat a fully fine-tuned
small CNN for classifying fine-grained leaf disease?

Short answer: it depends on the annotation budget, and the crossover is
reproducible. On 16-class tobacco images:

| Regime | Result |
|---|---|
| Zero-shot, no training | CLIP sits at the majority-class baseline. 13.0% is the baseline; the best of 14 templates reaches 14.5%, which is 2 images and not significant (exact McNemar p=0.87; p=0.999 under a selection-aware permutation test across all 14). |
| K=1 (1 image/class) | The LAION-2B probe wins: 40.1% vs ResNet-18's 33.0%, a seed-paired +7.1 pp (p=1.5e-4). The DFN-CLIP probe is not separable from the CNN (31.1%, p=0.14). |
| K=5, K=10 | ResNet-18 wins and the gap widens with K. |
| Full data | A validation-regularized probe on frozen CLIP is statistically indistinguishable from fine-tuned ResNet-18: 86.2% vs the 15-seed mean 87.7% ± 2.8, a 1.5 pp gap (15-seed mean-difference 95% CI [-0.1, +3.1] pp, one-sample t-test p=0.060; 10 of 15 seeds land above the probe and one matches it exactly). The 138-image test set cannot resolve a difference this small. |
| Full data, 5-fold CV | The better-powered comparison separates them: 89.4% vs 92.5%, ResNet ahead in all five folds. |
| Full data, matched tuning budget | Fine-tuning CLIP itself wins by a wide margin: 95.5% (DFN) / 94.9% (LAION) over 5 seeds, +7.2 to +7.8 pp over fine-tuned ResNet-18. |

Two consequences worth stating plainly. First, "CLIP or CNN" is not a property
of the encoders; it is a property of the regime, and the paper's primary
frozen-probe-vs-fine-tuned-CNN contrast is conservative for CLIP. Second, the
138-image test set is too small to settle anything under about 3 pp, so
single-split top-1 differences below that threshold are reported as
indistinguishable rather than as wins.

## Repository layout

```
manuscript/paper/probe-protocol.tex   the paper (all prose and numbers)
data/*.json                                   experiment outputs — source of truth
scripts/                                      every experiment behind a number
scripts/dataset.py                            TPDD dataset (vendored; see below)
scripts/model_zoo.py                          encoders + train/eval loops (vendored)
scripts/generate_figures.py                   all four figures, from data/*.json only
scripts/check_tex_vs_data.py                  consistency gate (run before committing)
RESEARCH_PROTOCOL.md                          experiment working rules
```

`data/*.json` is the single source of truth. The manuscript is written against
those files, not the other way round.

## Reproducing

The figures and the consistency check need no GPU and take seconds:

```bash
python3 scripts/generate_figures.py  # writes manuscript/paper/figures/{pdf,svg,png}
python3 scripts/check_tex_vs_data.py   # must print N/N consistent
cd manuscript/paper && pdflatex probe-protocol.tex
```

`scripts/check_tex_vs_data.py` derives each headline number from the JSON
artifacts and asserts it appears in the manuscript, and it also asserts that
superseded values are *gone* and that table rows match cell by cell. It exits
non-zero on drift, so a re-run that changes a result cannot silently contradict
the prose. It re-derives the Holm-adjusted p-values too, so the multiplicity
correction cannot regress.

Re-running the experiments themselves needs a CUDA GPU (an RTX 4070 Ti was
used) plus the two model families:

```bash
python3 scripts/canonical_protocol.py       # frozen-feature probes, C sweep
python3 scripts/canonical_fig_artifacts.py  # deterministic features + paired preds
python3 scripts/cv_protocol.py              # 5-fold CV, both backbones
python3 scripts/clip_fewshot_probe.py       # K=1/2/5/10
python3 scripts/paired_kshot_a4.py          # seed-paired ResNet few-shot
python3 scripts/zero_shot_clean.py          # 14 templates, saves per-image preds
python3 scripts/clip_finetune.py --out data/clip_finetune.json   # 5 seeds x 2 ckpts
```

`data/_orig_baseline/` holds the outputs of the pre-rerun pipeline, kept so the
diff against current results is auditable. Do not read those numbers as current.

DINOv2 ViT-L/14 fine-tuning (the matched-budget control completing the
backbone axis; results in `data/dinov2_finetune_full_16.json` and
`data/dinov2_kshot_ft_k{1,2,5,10}*.json`, run on A100-80GB/PyTorch 2.9):

```bash
python3 scripts/dinov2_finetune_kshot.py --full   # or --k 1|2|5|10
```

Base weights: official LVD-142M pretrain from HuggingFace
(`facebook/dinov2-large`, `dinov2_vitl14_pretrain.pth`),
SHA256 `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`.
A verified copy is kept at `checkpoints/dinov2/dinov2_vitl14_pretrain.pth`
(1.2 GB, git-ignored); set `DINOV2_PTH` to override the path. The script
asserts the checksum before training.

## Data

**The image corpora are not in this repository.** Obtain them as follows:

| Corpus | Size | Source |
|---|---|---|
| TPDD | 52 MB | `TamAko783/TPDD_Honglin_CLS` (Hugging Face Hub) |
| TLD | 178 MB | see `data/tld_*_manifest.json` for the exact file list |
| PlantVillage tomato | 313 MB | public PlantVillage release |

Expected layout, which `scripts/dataset.py` expects:

```
data/tpdd/
  train/       <class_name>/*.jpg  +  labels.json   (1295 images)
  validation/  <class_name>/*.jpg  +  labels.json   ( 138 images)
  test/        <class_name>/*.jpg  +  labels.json   ( 138 images)
```

`labels.json` is a list of `{"image": "<class>/<file>.jpg", "label": "<class>"}`.
Override the location with the `TPDD_DATA_ROOT` environment variable if you keep
the corpus elsewhere.

The four single-image test classes matter for reading the results: balanced
accuracy is dominated by them, and one of them (TSWV) accounts for most of the
balanced-accuracy gap between the two frozen backbones.

### A provenance problem you should know about

The original 11 MB low-resolution TPDD images were destroyed by an external
process on 2026-09-28, and they had never been tracked in git, so no earlier run
could be re-derived from the bytes it actually used. The corpus was rebuilt from
the public source at higher resolution. That moved the fine-tuned ResNet-18
full-data number from 86.4% to 87.7%: +1.3 pp, 1.8 SEM, Welch p=0.15, with 86.4%
inside the current run's 95% interval [86.3, 89.2]. Not statistically separable,
but no longer exactly reproducible either. The manuscript says so in the Fig. 4
caption. Paper 1 now versions its own copy of the corpus in git for this reason.

If you depend on a corpus for a paper, keep a dated tarball outside the repo
and in git, or learn this the same way.

### Citation direction (companion readout paper)

The readout-geometry section reports a domain-internal characterization and cites only external work (e.g. `shi2026registers`); it deliberately does **not** forward-cite the companion readout-hygiene paper (the `protocol-swing` project). Citation is one-way: the later-submitted paper cites the earlier one, so `protocol-swing` cites *this* paper, not the reverse. This keeps this paper free of a dangling reference to an unpublished work and keeps the two readout sections from sharing one body of evidence (anti-salami-slicing guard).

## Dependencies

Python 3.8 with `torch`, `torchvision`, `open_clip_torch` (2.32.0), `transformers`,
`huggingface_hub`, `scikit-learn`, `numpy`, `scipy`, `matplotlib`. Exact versions
from the runs are recorded in the `date`/`gpu`/`torch`/`open_clip` fields of the
JSON outputs.

## Citation

See the bibliography in `manuscript/paper/probe-protocol.tex`.
