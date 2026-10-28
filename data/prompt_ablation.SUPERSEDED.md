# ⚠️ SUPERSEDED — DO NOT CITE

**File:** `data/prompt_ablation.json`
**Status:** superseded as a measurement; retained only as provenance and as a live
input to three scripts.
**Superseded by:** `data/zero_shot_clean.json` (the authoritative zero-shot build).

## What this file is

The 14-template zero-shot sweep from the **pre-cleanup prompt build**, written by
`scripts/prompt_ablation.py`. Its KG templates reproduced the earlier
"knowledge-graph prompting helps" ordering.

## Why it must not be used for any claim

1. **It is a different prompt build, not a competing estimate of the same quantity.**
   Here KG-Symptom is the *best* template at 21/138 = 15.2% and EN-Symptom sits at
   only 8.7%. In the authoritative `zero_shot_clean.json` the ordering reverses:
   EN-Symptom 14.5% is best and KG-Symptom is 19/138 = 13.8%.

2. **It contains an implausible value indicating label leakage.**
   `EN-Label` (the bare class-name prompt) is recorded as
   `0.7246376811594203` — **exactly 100/138 = 72.46%**. A bare class-name
   zero-shot prompt cannot plausibly reach 72% on a 16-class problem whose
   majority-class baseline is 13.0% and whose uniform baseline is 6.25%. This is
   the signature of label leakage in that build, not a real accuracy.

3. **The paper reports the clean value, not this one.** Table 1 of the manuscript
   reports EN-Label at **0.7% (1/138)**, taken from `zero_shot_clean.json`.

## Why the file is still here

Three scripts read it and must keep working:

| Reader | Use |
|---|---|
| `scripts/zero_shot_clean.py:196` | records the prior prompt build's best accuracy as a *reconciliation* field in `zero_shot_clean.json` |
| `scripts/rigorous_analysis.py:168` | reporting (`prompt_ablation.json (paper)` row in the comparison dump) |
| `scripts/zpe_experiment.py:36` | static per-template accuracies used as ZPE ensemble weights |

Deleting or moving the file would break all three. Do not.

## If you are reviewing or re-deriving

Do not read this file as evidence about zero-shot CLIP. Read
`data/zero_shot_clean.json`.