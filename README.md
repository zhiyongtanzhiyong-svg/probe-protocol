# Fine-Tuning or Frozen Probes?

Reproducibility release for the manuscript *"Fine-Tuning or Frozen Probes? Empirical
Applicability Boundaries for Off-the-Shelf CLIP in Tobacco Leaf-Disorder Diagnosis"*
(Long, Tan, Deng, Luo, Zhang).

**Core finding.** For fine-grained tobacco leaf-disorder classification, the
CLIP-versus-ResNet-18 ordering is set by the adaptation protocol, not by the data
regime. Under a matched-configuration search (same size and selection procedure,
not equal compute), fine-tuned CLIP leads at every few-shot K; frozen probes at
default regularization tie or trail. The boundary is within-crop and the field
check is directional, not an external validation.

Persistent code/data release: <https://doi.org/10.5281/zenodo.24557268>.

---

## Repository layout

```
paper/          manuscript source + compiled PDF, cover letter, abstract, highlights, figures
data/           result JSONs — the single source of truth for every reported number
                (per-seed results, multiplicity, CV, manifests in data/tpdd/*/labels.json)
scripts/        experiment and analysis code; every number has a script here
protocols/      pre-registration documents (timestamps for the pre-specified contrasts)
tests/          consistency gate: re-derives numbers from data/ and checks the manuscript
```

`data/*.json` is authoritative: the manuscript is written against those files,
never the other way round.

---

## Quick start (no GPU, seconds)

```bash
python3 scripts/generate_figures.py   # regenerates paper/figures/{pdf,svg,png} from data/
python3 tests/check_tex_vs_data.py    # must print "N/N consistent"
cd paper && pdflatex probe-protocol.tex
```

`tests/check_tex_vs_data.py` re-derives each headline number (including the
Holm-adjusted p-values), asserts it appears in the manuscript, and fails on
superseded values or table drift.

## Reproducing the experiments (CUDA GPU)

RTX 4070 Ti was used for the primary results; DINOv2 controls used an A100-80GB.

```bash
python3 scripts/canonical_protocol.py       # frozen probes + C sweep
python3 scripts/canonical_fig_artifacts.py  # deterministic features + paired predictions
python3 scripts/cv_protocol.py              # 5-fold CV
python3 scripts/clip_fewshot_probe.py       # K=1/2/5/10 probes
python3 scripts/paired_kshot_a4.py          # seed-paired ResNet few-shot
python3 scripts/zero_shot_clean.py          # 14-template zero-shot
python3 scripts/clip_finetune.py --out data/clip_finetune.json
python3 scripts/dinov2_finetune_kshot.py --full   # or --k 1|2|5|10
```

DINOv2 ViT-L/14 weights: official LVD-142M (`facebook/dinov2-large`),
SHA256 `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`.
The script verifies the checksum before training.

## Data provenance

Image corpora are **not** in this repository (and not in the Zenodo release).

| Corpus | Size | Source |
|---|---|---|
| TPDD | 52 MB | `TamAko783/TPDD_Honglin_CLS` (Hugging Face Hub) |
| TLD | 178 MB | file list in `data/tld_*_manifest.json` |
| PlantVillage tomato | 313 MB | public PlantVillage release |
| Field images (246) | 78 MB | from the authors on request |

`scripts/dataset.py` expects:

```
data/tpdd/
  train/       <class_name>/*.jpg + labels.json   (1295)
  validation/  <class_name>/*.jpg + labels.json   ( 138)
  test/        <class_name>/*.jpg + labels.json   ( 138)
```

`labels.json` entries: `{"image": "<class>/<file>.jpg", "label": "<class>"}`.
Override with `TPDD_DATA_ROOT` if the corpus lives elsewhere.

The original low-resolution TPDD images were destroyed by an external process on
2026-09-28; the corpus was rebuilt from the public source at higher resolution.
This shifted the fine-tuned ResNet full-data mean from 86.4% to 87.7% (+1.3 pp,
Welch p=0.15, not significant). The manuscript discloses this in Appendix A.

## Dependencies

Python 3.8. Exact versions of the run environment are in
`requirements-lock.txt`; MD5 checksums of all tracked `data/` files are in
`MD5SUMS` (verify with `md5sum -c MD5SUMS`).

## Citation

See the bibliography in `paper/probe-protocol.tex`.
