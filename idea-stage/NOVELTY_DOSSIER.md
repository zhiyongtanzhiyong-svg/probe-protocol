# Novelty Dossier — Adaptation Ladder (A: token readout, B: LoRA cell)

## Context

Existing paper (target: Computers and Electronics in Agriculture): rigorous boundary study of
frozen CLIP (DFN-CLIP, LAION-2B, both ViT-L/14) vs fine-tuned ResNet-18 for 16-class tobacco
leaf-disorder classification (TPDD, 1,572 images; test 138), with tomato+corn (PlantVillage)
cross-crop replication, LP++/CoOp head controls, DINOv2 control, ResNet-50 scale control,
15-seed paired statistics, Holm/BH multiplicity, 5-fold CV. Known internal finding (currently
reported only as a "characterization, not a method"):
- DFN-CLIP: CLS token carries essentially all class signal (patch-token readouts add <=0.6pp;
  256 patch tokens nearly identical, mean pairwise l1 0.011; patch-only probe near noise).
- LAION-2B: patch tokens carry +10.9pp that the CLS readout misses (mean-pool 87.0% vs CLS
  76.1% full data); best readout +4.5pp at K=10.
- Attribution: pretraining (not architecture) determines where class signal sits.

Proposed upgrade: "adaptation ladder" paper — frozen probe -> head capacity (LP++/CoOp, done)
-> **A: token-aware frozen readout** -> **B: LoRA on CLIP image encoder** -> full fine-tuning
(done). Each rung = training-cost level; the paper maps accuracy-vs-adaptation-cost and
explains rung-to-rung gains via the readout-geometry mechanism.

## Claim A (token-aware frozen readout)

A1. A token-aware frozen head (e.g., mean-pool / adaptive pooling over patch tokens) can beat
    CLS-only linear probing on fine-grained plant-disease classification, training-free.
A2. The benefit is CHECKPOINT-DEPENDENT: rich on LAION-2B (+10.9pp full data), absent on
    DFN-CLIP (patch tokens collapsed) — pretraining determines where the signal sits.
A3. Practical rule for practitioners: check patch tokens before concluding CLS-only probing
    has hit the representation ceiling, in fine-grained low-data domains.

Closest prior work found (OpenAlex/arXiv, 2024-2026):
- "TAP into the Patch Tokens: Leveraging Vision Foundation Model Features for AI-Generated
  Image Detection", arXiv 2604.26772 (2026) — benchmarks VFMs incl. CLIP-ViT for AIGC
  detection; different task, not plant disease, no checkpoint-dependence claim.
- "Unmute the Patch Tokens: Rethinking Probing in Multi-Label Audio Classification" (2025) —
  audio domain, CLAP-style audio models.
- "Which Tokens to Use? Investigating Token Reduction in Vision Transformers" (WACV 2023) —
  token reduction for efficiency, not readout signal recovery.
- "Vision Transformers Need Registers" (Darcet et al., ICLR 2024) — attention sinks; already
  cited in the paper (as Shi et al. 2026 analog).
- "CA-CLIP" (Smart Agricultural Technology 2026) — plant disease, but TEXT-side structured
  prompts for multi-label co-occurring diseases, not image-side token readout.
- "BLAP" (Frontiers in Plant Science 2026) — lesion-aware visual prompt TUNING (trainable),
  few-shot crop disease.
- General ViT token-pooling/CLS-vs-patch literature exists (token selection, TokenLearner,
  mean-pool heads in probing studies) — the generic idea is NOT new.

## Claim B (LoRA cell)

B1. LoRA (rank sweep) on the CLIP ViT-L image encoder, few-shot K in {1,5,10} + full data on
    tobacco, answers: what fraction of the full fine-tuning gain does 0.1-1% of trainable
    parameters capture, per data regime, on fine-grained lesions?
B2. The adaptation-cost curve (probe -> LoRA -> full FT) is checkpoint- and regime-dependent.

Closest prior work found:
- "Low-Rank Few-Shot Adaptation of Vision-Language Models" (CVPR 2024) — general LoRA-VLM
  few-shot method.
- "One Head Eight Arms: Block Matrix based Low Rank Adaptation for CLIP-based Few-Shot
  Learning" (2025) — CLIP few-shot LoRA variant.
- "LoRA-TTT" (2025), "ACE-LoRA" (2026, medical VLM), DoRA/Flat-LoRA (2024) — LoRA methods.
- Plant-disease PEFT is mostly prompt-tuning (BLAP, CA-CLIP, BananaVLM 2026); no direct
  "LoRA-CLIP tobacco/leaf-disorder ladder" found in OpenAlex top hits, but coverage of this
  niche may be incomplete.

## Questions for the referee

1. Is Claim A (esp. A2 checkpoint-dependence in fine-grained plant disease) novel enough to
   carry a methods contribution at an ag-engineering Q1 journal, or is it "apply known
   token-pooling to Y"?
2. Is Claim B novel as a method, and as a FINDING (adaptation-budget curve on fine-grained
   lesions)? What would a reviewer cite as prior work?
3. As a combined "adaptation ladder" framing, what is the strongest novelty objection, and
   how should the contribution sentence be scoped to survive it?
4. Any obvious 2025-2026 paper you know (beyond the list) that already does CLS-vs-patch
   readout on CLIP for plant/medical fine-grained domains? Flag uncertainty honestly.
