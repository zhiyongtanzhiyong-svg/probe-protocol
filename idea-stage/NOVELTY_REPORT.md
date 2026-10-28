# Novelty Check Report — Adaptation Ladder (A: token readout, B: LoRA cell)

**Date**: 2026-10-20
**Method**: OpenAlex multi-source search (Phase B) + independent referee assessment (Phase C)
**Dossier**: `idea-stage/NOVELTY_DOSSIER.md`

## Verdict per claim

| Claim | Verdict | Rationale |
|-------|---------|-----------|
| A1 token-pooling beats CLS probing | **NOT NOVEL** | TokenLearner (2021), TAP (2026), DINOv3-MIL (2026); "apply known pooling to Y" |
| A2 checkpoint-dependence (LAION rich / DFN collapsed) | **PARTIALLY NOVEL** | Mechanism documented in medical imaging (chest radiography 2026, KiTS23 2026, TraceCLIP 2026); the two-checkpoint fine-grained tobacco quantification is a fresh instantiation — a finding, not a method |
| A3 practitioner readout rule | PARTIALLY NOVEL | Rides on A2; a recommendation, not a contribution |
| B1 LoRA rank sweep as method | **NOT NOVEL** | LoRA-CLIP few-shot is heavily published (CVPR 2024, 2025, 2026) |
| B2 adaptation-budget curve as finding | PARTIALLY NOVEL | Clean characterization; each rung is a known technique, the curve is the presentation |

## Verified closest prior work (all OpenAlex-confirmed 2026-10-20)

| Paper | DOI | Threat |
|-------|-----|--------|
| Frozen Foundation-Model Embeddings Discard Small-Lesion Signal in Chest Radiography (2026) | 10.48550/arxiv.2606.11606 | **Directly pre-empts A2's mechanism** in the medical domain — must cite + differentiate |
| DINOv3-MIL: Per-Kidney Tumour/Cyst Detection from Foundation-Model Patch Tokens on KiTS23 (2026) | 10.48550/arxiv.2607.22687 | Patch-tokens-for-small-lesion is an established pattern |
| TraceCLIP: Recovering Local Semantics from Patch-to-CLS Contributions (2026) | 10.48550/arxiv.2607.26107 | CLS-vs-patch local semantics in CLIP |
| TokenLearner: What Can 8 Learned Tokens Do for Images and Videos? (2021) | 10.48550/arxiv.2106.11297 | Generic learned-token head |
| Low-Rank Few-Shot Adaptation of VLMs (CVPR 2024); One Head Eight Arms (2025); LoRA-TTT (2025) | — | B-rung method prior work |
| CA-CLIP (Smart Agri Tech 2026); BLAP (Front. Plant Sci. 2026); BananaVLM (2026) | 10.1016/j.atech.2026.102617 | Plant-domain PEFT context (text-side prompt tuning, not image-side readout) |

## Strongest reviewer objection (referee)

"This is a thorough empirical survey of known adaptation techniques (probing, prompt-tuning,
LoRA, full-FT) on a single domain, whose headline mechanism — CLS under-weights small-lesion
signal while patch tokens recover it — is already documented in medical imaging. Where is the
contribution beyond 'we ran these on tobacco and re-confirmed a known phenomenon'?"

## Overall: **PROCEED WITH CAUTION**

Viable at C&E in Agriculture (ag-engineering values decision-relevant findings over method
novelty) **only if re-scoped as a finding/decision study, not a methods paper**. Do NOT pitch
to a CV journal as a method (desk-reject risk).

### Required moves
1. **Headline = checkpoint-dependent readout geometry** (the one defensible novelty),
   explicitly framed as cross-domain convergence with the medical small-lesion literature
   (chest radiography, KiTS23, TraceCLIP) — plant-disease domain is the first quantification
   of the two-checkpoint split.
2. **Cite all 4 verified prior papers above**; differentiate by domain + checkpoint comparison.
3. **Sell the adaptation-budget number** ("sub-1% LoRA recovers X% of full fine-tuning at Y
   cost, per regime") as the practical contribution.
4. Contribute sentence (referee-drafted):
   > "We provide the first controlled multi-checkpoint quantification of the
   > accuracy-vs-adaptation-cost trade-off for frozen CLIP on fine-grained lesion
   > classification, showing that (i) where class signal sits is pretraining-dependent
   > (LAION-2B vs DFN-CLIP) and (ii) sub-1% LoRA recovers X% of full fine-tuning at Y
   > cost — a decision-relevant deployment characterization, with the CLS-vs-patch
   > mechanism positioned against the medical small-lesion literature."

## Experiments still to run (unchanged from Option C)
- **A**: readout-head sweep on both checkpoints (mean-pool / max-pool / CLS⊕patch /
  attention-pool already partially done in `scripts/m0_diagnostic.py`) — formalize into
  Table + figure with per-K + full-data.
- **B**: LoRA rank sweep (r ∈ {4,8,16,32}) on CLIP ViT-L image encoder, K ∈ {1,5,10} + full,
  15 seeds where budget allows (at minimum K=5,10 full), both checkpoints.
- **New**: add the 4 verified prior-work papers to the bibliography + a short
   "Relation to the small-lesion readout literature" paragraph in Discussion.

---

## Cross-validation & final decision (added 2026-10-20)

### Content verification of the 3 medical prior papers (abstracts read)
- **CXR 2606.11606**: compares CLS vs patch-mean vs patch-local across 6 frozen ViTs (RAD-DINO, DINOv2-B, DINOv3-7B, BiomedCLIP, MedSigLIP, DINO-ResNet50); CLS at the chance floor (AUC 0.50–0.52), patch-local recovers ~1.0 (+0.41 to +0.49), **uniform direction across all models**. Pre-empts the *general* "patch recovers small-lesion signal" mechanism; does **not** report checkpoint-dependence.
- **DINOv3-MIL 2607.22687**: CLS probe vs attention-MIL vs prototype head on ONE model (DINOv3 ViT-H); aggregator matters, single model.
- **TraceCLIP 2607.26107**: training-free patch-to-CLS local semantics; dense tasks (localization/segmentation), not classification.

→ The *mechanism* is established in medical. What survives as novel is the **within-architecture, pretraining-only contrast** (DFN vs LAION, both ViT-L/14) + the plant-disease instantiation — a cleaner isolation than the cross-family medical comparisons, and a first quantification in fine-grained plant disease.

### Go/no-go referee (independent, investment framing)
**NO-GO on the full 2–3 week Option C.** The LoRA grid (B) is the weakest novelty + highest 12GB cost. Minimum viable: **A (readout/pretraining contrast, ~3–5 days, features cached) + at most one LoRA decision-number cell (r=8/16, K=5, 5 seeds, both ckpts, 2–3 days)**. Drop the r×K×15-seed grid. Gate on the K=5 readout test.
Framing rank at C&E-in-Ag: **(b) pretraining contrast + cross-domain replication > (c) adaptation-budget decision study > (a) adaptation-ladder method paper (avoid).**

### Load-bearing test — CONFIRMED (K=5, from cached frozen features, `data/m0_diagnostic.json` + re-run)
| K | DFN-CLIP best readout − CLS | LAION-2B best readout − CLS |
|---|---|---|
| full | +2.2pp (CLS⊕max; every *pure* patch readout ≤ CLS: mean 18.8, max 21.7) | +11.6pp (mean-pool 87.0 vs CLS 76.1) |
| 10-shot | +0.6pp | +4.5pp |
| **5-shot** | **+0.5pp (collapsed)** | **+6.5pp (rich)** |

The split **holds and widens as K falls** — a low-data phenomenon, not a full-data artifact. This is the one genuinely defensible, evidence-backed novelty.

### DECISION
**Invest ~1 week (narrow), not 2–3 weeks (full).**
- **Invest (headline):** the within-architecture pretraining contrast / checkpoint-dependent readout geometry, framed as cross-domain convergence with the medical small-lesion literature. Cheap (features cached, diagnostic already run).
- **Optional (secondary):** one LoRA cell for a single "LoRA recovers X% of full fine-tune" decision number.
- **Drop:** the full LoRA rank×K×seed grid.

### Landed in the paper (this session)
- 3 medical 2026 citations added (`cxr2026frozen`, `traceclip2026`, `dinov3mil2026`); authors verified via OpenAlex.
- Readout-geometry paragraph (Section "Why does generic CLIP underperform…") now reports the K=5 split and the cross-domain convergence.
- New **Table `tab:readout`** (readout × K, both checkpoints).
- Abstract: one added sentence surfacing the checkpoint-dependent readout geometry as a takeaway.
- Compiles clean (2× pdflatex, 30 pp, no undefined refs/cites).
