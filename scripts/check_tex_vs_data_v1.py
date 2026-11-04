"""Cross-check the manuscript's reported numbers against the experiment JSONs.

The paper's claim is that data/ is the single source of truth. This script
enforces that mechanically: it derives every headline number from the JSON
artifacts and asserts that the same number appears in the .tex. It fails loudly
rather than warning, so a stale constant cannot survive an edit unnoticed.

Run:  python3 scripts/check_tex_vs_data.py
"""
import json
import re
import statistics as st
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "manuscript" / "paper" / "probe-protocol.tex"
DATA = ROOT / "data"


def load(name):
    with open(DATA / name) as fh:
        return json.load(fh)


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}"


def norm(s):
    """Normalise LaTeX math spacing so '10.1\\,pp', '$10.1$ pp' and '10.1 pp' match.

    Numbers in the manuscript are written with inconsistent (all valid) spacing
    around units, so a literal substring search produces false negatives. Strip
    the math-mode delimiters and every thin/normal space before comparing.
    """
    for tok in ("\\,", "\\;", "\\ ", "\\!", "~", " ", "$", "\\%"):
        s = s.replace(tok, "" if tok != "\\%" else "\\%")
    return re.sub(r"\s+", "", s)


def main():
    tex = TEX.read_text(encoding="utf-8")
    # collapse whitespace so a number split across a line break still matches,
    # then normalise math spacing (see norm())
    flat = norm(re.sub(r"\s+", " ", tex))

    art = load("canonical_fig_artifacts.json")
    can = load("canonical_results.json")
    zs = load("zero_shot_full_20261025.json")
    cv = load("cv_protocol.json")
    dino = load("dinov2_protocol.json")
    dino518 = load("dinov2_native518.json")
    mult16 = load("multiplicity_16_matched_corrected.json")
    ver = load("claim_verification.json")
    r50 = load("resnet50_full.json")
    cal = load("calibration_20261025.json")
    # Legacy files still loaded for backward-compat checks that reference them
    try:
        fs = load("clip_fewshot.json")
    except FileNotFoundError:
        fs = None
    try:
        la = load("laion_fewshot_canonical.json")
    except FileNotFoundError:
        la = None

    checks = []

    def chk(label, needle, ok=None):
        present = norm(needle) in flat if ok is None else ok
        checks.append((label, needle, present))

    # ---- full-data CLIP probes -------------------------------------------
    chk("CLIP DFN C=1", f"{pct(art['clip']['linear_C1']['accuracy'])}\\%")
    chk("CLIP DFN bestC", f"{pct(art['clip']['linear_bestC']['accuracy'])}\\%")
    chk("CLIP best C", f"$C{{=}}{art['clip']['best_C']:.0f}$")
    for ck, label in (("dfn_clip", "DFN"), ("laion2b", "LAION")):
        chk(f"CLIP {label} bestC",
            f"{pct(can['clip_by_checkpoint'][ck]['linear_bestC']['accuracy'])}\\%")
        chk(f"CLIP {label} MLP",
            f"{pct(can['clip_by_checkpoint'][ck]['mlp']['accuracy'])}\\%")
    chk("CLIP DFN kNN", f"{pct(can['clip_by_checkpoint']['dfn_clip']['knn1']['accuracy'])}\\%")

    # ---- ResNet ------------------------------------------------------------
    chk("ResNet FT 15-seed mean", f"{pct(art['resnet']['mean'])}\\%")
    chk("ResNet FT 15-seed sd", f"{pct(art['resnet']['std'])}")
    chk("ResNet seed0", f"{pct(art['resnet']['seed0']['accuracy'])}\\%")
    chk("ResNet frozen C=1", f"{pct(can['matrix']['resnet_frozen_linear']['accuracy'])}\\%")
    chk("ResNet frozen bestC", f"{pct(can['matrix']['resnet_frozen_bestC']['accuracy'])}\\%")
    chk("ResNet frozen MLP", f"{pct(can['matrix']['resnet_frozen_mlp']['accuracy'])}\\%")

    # ---- few-shot ----------------------------------------------------------
    for k in ("K=1", "K=5", "K=10"):
        chk(f"DFN {k}", f"{fs[k]['mean'] * 100:.1f}\\%")
        chk(f"LAION {k}", f"{la[k]['mean'] * 100:.1f}\\%")
    # ResNet K-shot reference differs by regime: K=1 is the seed-paired re-run
    # (paired_kshot_a4.json); K=5/10 are the matched-budget (P3b) re-runs, which is
    # what Table 1 and the executed ten-contrast family quote (resnet_budget =
    # "matched"). The older seed-paired K=5/10 values (65.5/75.7) are superseded.
    chk("ResNet K=1 (seed-paired)", f"{pk['K=1']['resnet_mean'] * 100:.1f}\\%")
    chk("ResNet K=5 (matched P3b)", f"{p3b_r5['stage2_final']['mean'] * 100:.1f}\\%")
    chk("ResNet K=10 (matched P3b)", f"{p3b_r10['stage2_final']['mean'] * 100:.1f}\\%")

    # ---- fine-tuned CLIP (matched-budget control) --------------------------
    import statistics as _st
    for ck, label in (("dfn_clip", "DFN"), ("laion2b", "LAION")):
        t = [r["test"] for r in ft["results"][ck]["seeds"]]
        chk(f"CLIP FT {label} mean", f"{_st.mean(t) * 100:.1f}\\%")
        chk(f"CLIP FT {label} sd", f"{_st.stdev(t) * 100:.1f}")
        d_pp = _st.mean(t) * 100 - art["resnet"]["mean"] * 100
        chk(f"CLIP FT {label} margin", f"{d_pp:.1f}")
    chk("CLIP FT margin range",
        f"{min(_st.mean([r['test'] for r in ft['results'][c]['seeds']]) * 100 for c in ('dfn_clip', 'laion2b')) - art['resnet']['mean'] * 100:.1f}")
    # the ResNet reference embedded in the fine-tune artifact must be current
    chk("clip_finetune ResNet ref is canonical", "n/a",
        abs(ft["resnet_finetuned_ref"]["accuracy"] - art["resnet"]["mean"]) < 1e-9)

    # ---- DFN-CLIP few-shot fine-tuning (P3c: the two new matched-budget cells) --
    dfn5 = load("p3c_clip_finetune_kshot_dfn.json")
    dfn10 = load("p3c_clip_finetune_kshot_dfn_k10.json")
    chk("DFN FT K=5 cell (Table 1)",
        f"${dfn5['stage2_final']['mean'] * 100:.1f} \\pm {dfn5['stage2_final']['std'] * 100:.1f}$")
    chk("DFN FT K=10 cell (Table 1)",
        f"${dfn10['stage2_final']['mean'] * 100:.1f} \\pm {dfn10['stage2_final']['std'] * 100:.1f}$")

    # ---- ResNet-50 scale control: per-seed accuracies must be stored, otherwise
    #      the reported mean +/- SD cannot be checked against the cited artifact --
    chk("ResNet-50 per-seed accs stored", f"{len(r50.get('accs', []))} values",
       len(r50.get("accs", [])) == r50.get("n_seeds"))
    import statistics as _st50
    if r50.get("accs"):
        chk("ResNet-50 mean", f"{_st50.mean(r50['accs']) * 100:.1f}\\%",
           abs(_st50.mean(r50["accs"]) - r50["mean"]) < 1e-9)
        chk("ResNet-50 SD", f"{_st50.stdev(r50['accs']) * 100:.1f}",
           abs(_st50.stdev(r50["accs"]) - r50["std"]) < 1e-9)
    chk("ResNet-50 refs are current", "n/a",
       abs(r50["refs"]["resnet18_finetuned"] - art["resnet"]["mean"]) < 1e-9
       and abs(r50["refs"]["clip_finetuned_dfn"]
               - _st.mean([x["test"] for x in ft["results"]["dfn_clip"]["seeds"]])) < 1e-9)

    # ---- zero-shot ---------------------------------------------------------
    best_name, best = max(zs["templates"].items(), key=lambda kv: kv[1]["acc"])
    chk("ZS best template acc", f"{best['acc'] * 100:.1f}\\%")
    chk("ZS majority baseline",
        f"{sum(zs['majority_baseline_correct']) / zs['n_test'] * 100:.1f}\\%")
    chk("ZS uniform ensemble", f"{zs['ensemble_uniform']['acc'] * 100:.1f}\\%")
    chk("ZS ZPE ensemble", f"{zs['ensemble_zpe_signal']['acc'] * 100:.1f}\\%")

    # ---- cross-validation --------------------------------------------------
    chk("CV CLIP", f"{cv['summary']['clip']['acc_mean'] * 100:.1f}\\%")
    chk("CV ResNet", f"{cv['summary']['resnet']['acc_mean'] * 100:.1f}\\%")
    # the per-fold gap breakdown -- not just the aggregate mean -- must match the
    # cited artifact; a stale per-fold list from a superseded CV run would
    # otherwise slip past the two aggregate checks above
    _cv_gap = cv["summary"]["gap_clip_minus_resnet"]
    _gap_list = ",".join(f"{g:.1f}" for g in _cv_gap["per_fold_pp"])
    chk("CV per-fold gap list", _gap_list, _gap_list in flat)
    chk("CV gap SD", r"\text{pp}\pm1.0", r"\text{pp}\pm1.0" in flat)
    chk("CV per-fold list not stale baseline", "absent",
        "-5.4,-5.0,-4.6,-1.2,-5.4" not in flat)

    # ---- DINOv2 ------------------------------------------------------------
    chk("DINO224 C=1", f"{dino['full_data']['linear_C1']['accuracy'] * 100:.1f}\\%")
    chk("DINO224 bestC", f"{dino['full_data']['linear_bestC']['accuracy'] * 100:.1f}\\%")
    chk("DINO518 C=1", f"{dino518['full_data']['linear_C1']['accuracy'] * 100:.1f}\\%")
    chk("DINO518 bestC", f"{dino518['full_data']['linear_bestC']['accuracy'] * 100:.1f}\\%")

    # ---- multiplicity ------------------------------------------------------
    # The executed family as of P3c is the TWELVE-contrast family
    # (data/multiplicity_p3c.json): the eight frozen-probe contrasts (K in {1,2,5,10}
    # x 2 checkpoints; K=5/10 against the matched-budget ResNet, K=1/2 against the
    # seed-paired re-run) plus four matched-budget fine-tuned-CLIP contrasts
    # (K=5/10 x {LAION-2B, DFN-CLIP}). The P3b ten-contrast family
    # (data/multiplicity_p3b.json) is now a subset of it; the k2 8-family
    # (data/multiplicity_k2.json) is the frozen-probe subset.
    chk("Holm survivors (executed 12-family)",
        f"{mult3c['n_holm_surviving']} of {mult3c['n_tests']}")
    chk("executed family has 12 tests", "n_tests=12", mult3c["n_tests"] == 12)
    chk("12-family: all contrasts survive Holm", "12 of 12",
        mult3c["n_holm_surviving"] == 12)
    chk("P3b 10-family subset store intact", "n_tests=10", mult3["n_tests"] == 10)
    chk("k2 8-family subset store intact", "n_tests=8", mult["n_tests"] == 8)
    chk("legacy 6-test multiplicity.json superseded, not cited",
        "absent", norm("data/multiplicity.json") not in flat
        or norm("data/multiplicity.json}") in flat)
    # Holm adjusted p must follow the step-down formula AND stay monotone in the
    # raw p, so the generator bug cannot come back silently:
    #   adj(r) = max_{j<=r} min(1, (m - j + 1) * p_(j))
    for name, fam in (("12-family", mult3c), ("10-family", mult3), ("k2 8-family", mult)):
        rows = sorted(fam["rows"], key=lambda r: r["paired_t_p"])
        m = len(rows)
        ok_adj, run = True, 0.0
        for r, row in enumerate(rows, start=1):
            run = max(run, min(1.0, (m - r + 1) * row["paired_t_p"]))
            if abs(row["holm_adjusted_p"] - run) > 1e-12:
                ok_adj = False
        chk(f"Holm adjusted p = step-down running max ({name})", "n/a", ok_adj)
        ok_rej = all(row["holm_significant"] == (row["holm_adjusted_p"] <= fam["alpha"])
                      for row in fam["rows"])
        chk(f"Holm flags agree with adjusted p ({name})", "n/a", ok_rej)

    # ---- paired claims -----------------------------------------------------
    c1 = ver["Q2_full_data"]["clip_C1"]
    bc = ver["Q2_full_data"]["clip_bestC"]
    # The gap may be written as a pp delta or as the accuracy pair; both are
    # valid prose, so accept either rather than forcing one phrasing.
    _pair = (f"{art['clip']['linear_C1']['accuracy'] * 100:.1f}"
             f"\\% vs.\\ {art['resnet']['seed0']['accuracy'] * 100:.1f}\\%")
    # The McNemar p is computed against seed 0, so the seed-0 pair must be
    # stated; the 15-seed mean is a separate quantity.
    chk("Q2 C=1 delta (pp or accuracy pair)",
        f"{c1['delta_pp']:.1f}pp / {_pair}",
        norm(f"{c1['delta_pp']:.1f}\\,pp") in flat
        or norm(_pair) in flat
        # also accept "76.8% vs. the 15-seed mean 87.7% ... seed at 87.0%"
        or (norm(f"{art['clip']['linear_C1']['accuracy'] * 100:.1f}\\% vs.")
            in flat and norm(f"{art['resnet']['seed0']['accuracy'] * 100:.1f}\\%")
            in flat))
    chk("Q2 C=1 p", f"p={c1['mcnemar_p_exact']:.3f}")
    chk("Q2 C=1 CI low", f"{c1['ci95_delta_pp'][0]:.1f}")
    chk("Q3 bestC delta", f"{abs(bc['delta_pp']):.1f}\\,pp")
    chk("Q3 bestC p", f"p={bc['mcnemar_p_exact']:.2f}")
    chk("Q3 bestC n_seeds", f"{bc['n_seeds_above_clip']} of {bc['n_resnet_seeds']}")
    # Guard against the specific confusion this repo already hit once: seed 0
    # (86.96%) is the *lowest-but-one* ResNet run and sits 0.72 pp above the
    # 86.23% probe, so quoting "87.0% vs 86.2% = 0.7 pp" is arithmetically
    # correct but is a single draw, not the 15-seed mean (87.7%, gap 1.5 pp).
    # Assert the .tex quotes the seed-MEAN gap, and that it never presents the
    # seed-0 gap as if it were the headline difference.
    _rn = np.array(art["resnet"]["accs"]) * 100
    _seedmean_gap = _rn.mean() - bc["clip_acc"] * 100
    _mean_tok = norm(f"{_seedmean_gap:.1f}\\,pp")
    _seed0_gap_tok = norm(f"{abs(bc['delta_pp']):.1f}\\,pp")
    chk("Q3 bestC 15-seed mean gap quoted", _mean_tok, _mean_tok in flat)
    # Document-wide presence is not enough: other passages legitimately cite both
    # numbers (the per-image McNemar paragraph is explicitly about seed 0). The
    # failure mode is the *pairing*, so require that no sentence quotes the
    # probe next to seed 0's accuracy as the headline full-data difference.
    _probe_tok = norm(f"{bc['clip_acc'] * 100:.1f}\\%")
    _seed0_acc_tok = norm(f"{art['resnet']['seed0']['accuracy'] * 100:.1f}\\%")
    # Do NOT try to split into sentences here. norm() collapses whitespace and the
    # prose is full of LaTeX em-dashes (---), which join clauses with no
    # terminator, so any sentence split both glues real violations together and
    # matches the MDPI preamble. Instead inspect a local window around every
    # mention of the probe accuracy: if seed 0's accuracy and seed 0's gap show
    # up in the same window while the seed-mean gap does not, that is the
    # confusion being guarded against.
    _win = 260
    _bad = []
    for _m in re.finditer(re.escape(_probe_tok), flat):
        _w = flat[max(0, _m.start() - _win): _m.end() + _win]
        if not (_seed0_acc_tok in _w and _seed0_gap_tok in _w):
            continue
        if _mean_tok in _w:
            continue
        # Co-occurrence alone is not the confusion. The MLP robustness note
        # legitimately mentions the probe (86.2%), an MLP head 0.7 pp below it,
        # and separately the RBF-SVM's own 87.0% -- three different models. The
        # error is specifically attributing seed 0's gap TO the probe, i.e. the
        # seed-0 gap must be the one that is compared against the probe. Anchor on
        # the actual comparison: probe accuracy, seed-0 accuracy and seed-0 gap
        # must appear as one claim, detected via the fixed phrase that introduced
        # the mix-up.
        if re.search(re.escape(_probe_tok) + r".{0,120}?" + re.escape(_seed0_gap_tok), _w) or \
           re.search(re.escape(_seed0_gap_tok) + r".{0,120}?" + re.escape(_seed0_acc_tok), _w):
            _bad.append(_w)
    chk("Q3 bestC probe-vs-seed0 never paired as headline",
        f"{len(_bad)} offending window(s)",
        not _bad)
    # The seed-mean CI must contain zero for "indistinguishable" to be honest.
    _sem = _rn.std(ddof=1) / np.sqrt(len(_rn))
    _tcrit = 2.145  # df=14, two-sided 95%
    _lo = _seedmean_gap - _tcrit * _sem
    chk("Q3 bestC 15-seed CI brackets 0 (justifies 'indistinguishable')",
        f"CI low {_lo:.1f}",
        _lo <= 0 <= _seedmean_gap + _tcrit * _sem)
    chk("Q2 C=1 all seeds exceed probe",
        f"all {c1['n_seeds_above_clip']} of {c1['n_resnet_seeds']} ResNet seeds")

    # ------------------------------------------------------------------
    # SINGLE-CALIBER GUARD (added 2026-10-05).
    #
    # Two defects let one nominal method carry two different accuracy rows.
    # (a) The few-shot probe was fitted to support features encoded with the
    #     stochastic training transform and drawn once per seed. For a single
    #     convex sklearn fit that is noise injection, not augmentation, and it
    #     made the full-data column non-deterministic (seed sd was 1.79) while
    #     the caption asserted zero seed variance.
    # (b) The LP++/CoOp artifacts carry their own probe_C1 baseline, computed
    #     on a separately cached encoding, so the tex could quote either one as
    #     "the DFN-CLIP probe".
    #
    # These checks make the failure mechanical rather than a matter of reading
    # carefully: the canonical probe row must come from clip_fewshot.json, the
    # superseded numbers must be gone, the full-data column must actually be
    # deterministic, and the disclosed encoding delta must match a recomputation.
    if mult["n_tests"] == 8:
        # (1) Canonical probe row must be the current single-caliber artifact.
        for k in (1, 2, 5, 10):
            want = f"{fs[f'K={k}']['mean'] * 100:.1f}"
            chk(f"Table 2 DFN probe K={k} = clip_fewshot.json", want,
                norm(f"${want} \\pm") in flat or norm(want) in flat)

        # (2) The superseded augmented-support values must not survive anywhere.
        #     51.3 and 49.1 in particular were the two competing K=5 numbers.
        for stale, why in (("31.1", "aug K=1"), ("51.3", "aug K=5"),
                           ("58.3", "aug K=10")):
            chk(f"superseded probe value {stale} ({why}) absent", "absent",
                norm(stale) not in flat)

        # (3) The caption's zero-seed-variance claim must be true of the artifact.
        chk("full-data probe is deterministic (sd == 0)",
            "std == 0", fs["full"]["std"] == 0.0)
        chk("support transform recorded as EVAL in artifact",
            "support_transform=EVAL",
            all(fs[f"K={k}"].get("support_transform") == "EVAL" for k in (1, 2, 5, 10)))

        # (4) K=2 must be reported: the paired contrast, and the crossover bracket.
        rn2 = pk2["K=2"]["resnet_mean"] * 100
        chk("Table 2 ResNet K=2 = paired run", f"{rn2:.1f}",
            norm(f"${rn2:.1f} \\pm") in flat)
        la2 = la["K=2"]["mean"] * 100
        chk("LAION probe K=2 quoted", f"{la2:.1f}",
            norm(f"${la2:.1f} \\pm") in flat)
        chk("crossover bracketed to (2,5)", "open interval $(2,5)$",
            norm("open interval $(2,5)$") in flat)
        # The bracket is only legitimate if LAION leads at K=2 and trails at K=5.
        k2row = next(r for r in mult["rows"]
                     if r["contrast"].startswith("K=2: CLIP (LAION"))
        k5row = next(r for r in mult["rows"]
                     if r["contrast"].startswith("K=5: CLIP (LAION"))
        chk("bracket is sound (K=2 leads, K=5 trails)",
            f"K2 {k2row['delta_pp']:+.2f} / K5 {k5row['delta_pp']:+.2f}",
            k2row["delta_pp"] > 0 > k5row["delta_pp"])

        # (5) The corrected K=1 DFN contrast must be reported as significant,
        #     since it changes status against the superseded value.
        df1 = next(r for r in mult["rows"]
                   if r["contrast"].startswith("K=1: CLIP (DFN"))
        chk("corrected K=1 DFN contrast is significant", f"p={df1['paired_t_p']:.3f}",
            df1["holm_significant"])
        chk("corrected K=1 DFN p-value quoted", "0.031",
            norm("p=0.031") in flat)

        # (6) Feature-store unification must be complete for LP++ and explicitly
        #     scoped for CoOp, and the disclosed numbers must recompute.
        lp2 = load("lp_plus_plus.json")
        laic = load("laion_fewshot_canonical.json")
        canon_d = [fs[f"K={k}"]["mean"] * 100 for k in (1, 5, 10)]
        canon_l = [laic[f"K={k}"]["mean"] * 100 for k in (1, 5, 10)]
        d_dfn = [lp2["checkpoints"]["dfn_clip"][f"K={k}"]["probe_C1"]["mean"] * 100 - c
                 for k, c in zip((1, 5, 10), canon_d)]
        d_lai = [lp2["checkpoints"]["laion2b"][f"K={k}"]["probe_C1"]["mean"] * 100 - c
                 for k, c in zip((1, 5, 10), canon_l)]
        worst = max(max(abs(x) for x in d_dfn), max(abs(x) for x in d_lai))
        chk("LP++ probe reference is now the canonical row", f"max |delta|={worst:.3f} pp",
            worst <= 0.02)
        chk("tex states the LP++/canonical coincidence", "within $0.01$ pp",
            norm("within $0.01$ pp") in flat)

        # CoOp is deliberately NOT unified; the tex must say so and must quote
        # CoOp's own probe triple rather than the canonical one.
        co = load("coop_official.json")
        coop_triple = [co["checkpoints"]["dfn_clip"][f"K={k}"]["probe_C1"]["mean"] * 100
                       for k in (1, 5, 10)]
        want = "/".join(f"{v:.1f}" for v in coop_triple)
        chk("CoOp probe reference quoted from its own artifact", want,
            norm(want) in flat)
        chk("CoOp non-unification is disclosed", "CoOp was not re-run",
            norm("CoOp was not re-run") in flat)

        # The LAION few-shot correction must be stated with both endpoints.
        two = load("two_model_e4.json")["few_shot_probe"]["full"]["mean"] * 100
        chk("LAION full-data disagreement is disclosed",
            f"{two:.1f} vs canonical {laic['full']['mean']*100:.1f}",
            norm(f"{two:.1f}\\%") in flat and norm("76.1\%") in flat)
        chk("LAION correction declared sign-preserving", "no change of sign",
            norm("no change of sign") in flat)

        # The stale store must still be flagged as drifted, never cited as canonical.
        legacy_cache = DATA / "clip_features_clean.npz"
        canon_cache = DATA / "canonical_clip_features_dfn.npz"
        if legacy_cache.exists() and canon_cache.exists():
            import numpy as _np
            _a = _np.load(canon_cache)["train"]
            _b = _np.load(legacy_cache)["train_feats"]
            drift = float(_np.abs(_a - _b).max())
            chk("legacy feature cache is drifted (must not be cited as canonical)",
                f"max|diff|={drift:.3f}", drift > 1e-5)
            chk("tex discloses the feature-store unification", "Feature-store unification",
                norm("Feature-store unification") in flat)

    bad = [(l, n) for l, n, ok in checks if not ok]
    for label, needle, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:32s} {needle}")

    # ---- producer/consumer filename agreement -------------------------------
    # The producer used to write clip_fewshot_fixed.json while six consumers read
    # clip_fewshot.json; they were equal only because someone copied them by
    # hand. If both exist they must agree, and both must exist.
    alias = DATA / "clip_fewshot_fixed.json"
    canon = DATA / "clip_fewshot.json"
    chk("clip_fewshot.json exists (consumed by 6 scripts)", "present", canon.exists())
    if canon.exists() and alias.exists():
        c_, a_ = load("clip_fewshot.json"), load("clip_fewshot_fixed.json")
        agree = all(abs(c_[k]["mean"] - a_[k]["mean"]) < 1e-12
                    for k in ("K=1", "K=5", "K=10") if k in c_ and k in a_)
        chk("clip_fewshot.json == _fixed alias", "identical" if agree else "DIVERGED", agree)

    # ---- the Conclusions must quote the current numbers too ----------------
    # A stale "best of 14 templates 13.0%" survived in the Conclusions after the
    # Results section had been corrected, because every other check only asked
    # whether a value appeared SOMEWHERE in the file. Conclusions are the most
    # read part of a paper, so they are checked explicitly.
    ci = tex.find(r"\section{Conclusions}")
    # Conclusions is the LAST section, so there is no following \section; bound
    # the slice at the bibliography (or end of file) instead.
    cb = tex.find(r"\begin{thebibliography}", ci)
    cend = cb if cb > ci else len(tex)
    concl = norm(tex[ci:cend]) if ci > 0 else ""
    best_acc = max(v["acc"] for v in zs["templates"].values()) * 100
    ft_lo = min(st.mean([x["test"] for x in ft["results"][c]["seeds"]])
                for c in ft["results"]) * 100
    ft_hi = max(st.mean([x["test"] for x in ft["results"][c]["seeds"]])
                for c in ft["results"]) * 100
    for label, needle in (("conclusion: best template", f"{best_acc:.1f}"),
                          ("conclusion: fine-tuned CLIP low", f"{ft_lo:.1f}"),
                          ("conclusion: fine-tuned CLIP high", f"{ft_hi:.1f}"),
                          ("conclusion: CV ResNet",
                           f"{cv['summary']['resnet']['acc_mean'] * 100:.1f}"),
                          ("conclusion: balanced-accuracy ResNet",
                           f"{art['resnet']['balanced_accuracy_mean'] * 100:.1f}")):
        chk(label, needle, bool(concl) and norm(needle) in concl)

    # ---- zero-shot: best template, its binomial test, and split selection ----
    # The Results once quoted a stale EN-Photo/KG-Symptom "match the baseline"
    # pairing (18/138) and a 7.3% validation-split figure. Every zero-shot claim
    # is now re-derived from zero_shot_clean.json / r4b_results.json so it
    # cannot drift again. Section-scoped slices mirror the Conclusions block:
    # presence ANYWHERE in the file is not enough when two sections quote the
    # same statistic (the intro says "best of 14 templates (14.5%)" and the
    # discussion says "best of 14 templates: 14.5%").
    def scoped(start, end, needle):
        """True if norm(needle) occurs in tex[start ... next end marker)."""
        i = tex.find(start)
        if i < 0:
            return False
        j = tex.find(end, i + len(start))
        seg = tex[i:j] if j > 0 else tex[i:]
        return norm(needle) in norm(re.sub(r"\s+", " ", seg))

    n_test = zs["n_test"]
    k_best = round(best_acc / 100 * n_test)
    p0 = sum(zs["majority_baseline_correct"]) / n_test
    p_sup = sum(comb(n_test, i) * p0 ** i * (1 - p0) ** (n_test - i)
                for i in range(k_best, n_test + 1))
    chk("zero-shot best template count", f"{k_best}/{n_test}")
    chk("zero-shot superiority binomial p", f"p={p_sup:.2f}")
    top = max(zs["templates"], key=lambda t: zs["templates"][t]["acc"])
    top_pct = zs["templates"][top]["acc"] * 100
    kg_pct = zs["templates"]["KG-Symptom"]["acc"] * 100
    chk("zero-shot top template named", f"({top}, {top_pct:.1f}\\%)")
    chk("zero-shot KG best template", f"KG-Symptom, {kg_pct:.1f}\\%")
    chk("intro: best-of-14 value",
        f"best of 14 templates ({best_acc:.1f}\\%)",
        scoped(r"\section{Introduction}", r"\section",
               f"best of 14 templates ({best_acc:.1f}\\%)"))
    chk("discussion: best-of-14 value",
        f"best of 14 templates: {best_acc:.1f}\\%",
        scoped(r"\subsection{Zero-Shot Sits at the Baseline", r"\subsection",
               f"best of 14 templates: {best_acc:.1f}\\%"))
    # RQ3 is a two-part question (external zero-shot check (i), cross-crop
    # check (ii)). The subsection titles, the two answer markers, the intro
    # bullet and the closing recommendation must all carry the same labels, so
    # a half-finished relabel -- e.g. one marker renamed and the other left as
    # a bare "Answer to RQ3:" -- fails here.
    chk("RQ3(i): top template value",
        f"({top}, {top_pct:.1f}\\%)",
        scoped(r"\textbf{Answer to RQ3 (i):}", r"\subsection",
               f"({top}, {top_pct:.1f}\\%)"))
    chk("RQ3 intro is two-part (i)+(ii)",
        "(i) Does the at-baseline zero-shot failure replicate on an external "
        "dataset (TLD)? (ii) Do the zero-shot / few-shot / full-data model "
        "rankings generalize across crops")
    chk("RQ3 subsection (i) titled",
        r"\subsection{RQ3 (i): External Zero-Shot Consistency Check}")
    chk("RQ3 subsection (ii) titled",
        r"\subsection{RQ3 (ii): Cross-Crop Consistency on Tomato and Corn (PlantVillage)}")
    chk("RQ3(ii) answer marker", r"\textbf{Answer to RQ3 (ii, cross-crop):}")
    # corn cross-crop table (tab:corn) -- guard the second crop against corn_results.json
    c = load("corn_results.json")
    czs = c["zero_shot"]
    cbest = max(v["accuracy"] for k, v in czs.items() if not k.startswith("__"))
    chk("corn table label", r"\label{tab:corn}")
    chk("corn table caption", "Cross-crop full-protocol replication on corn")
    # table cells are bare numbers (column header carries the %), so match pct() bare
    chk("corn zero-shot best", pct(cbest))
    chk("corn zero-shot majority", pct(czs['__majority_baseline']['accuracy']))
    for kk in ("K=1", "K=5", "K=10"):
        chk(f"corn {kk} CLIP", pct(c['few_shot'][kk]['clip_probe_C1']['mean']))
        chk(f"corn {kk} ResNet", pct(c['few_shot'][kk]['resnet_finetuned']['mean']))
    chk("corn full-data CLIP", pct(c['full_data']['clip']['clip_bestC']))
    chk("corn full-data ResNet", pct(c['full_data']['resnet']['mean']))
    chk("corn CV CLIP", pct(c['cv']['summary']['clip_bestC_mean']))
    chk("corn CV ResNet", pct(c['cv']['summary']['resnet_mean']))
    chk("corn near-dup screen disclosed", "no test image above cosine 0.99")
    chk("no bare Answer-to-RQ3 marker left", "bare marker absent",
        r"\textbf{Answer to RQ3:}" not in tex)
    chk("recommendation cites RQ3(i)", "(RQ3(i))")

    # validation-split prompt selection (Results, zero-shot section)
    ps = r4b["prompt_selection"]
    chk("val-split test acc", f"achieves only {pct(ps['test_acc'])}\\%")
    chk("val-split template named", f"validation-best template ({ps['best_prompt']})")
    chk("val-split winner is val-argmax", "best_prompt == argmax(val_accs)",
        ps["best_prompt"] == max(ps["val_accs"], key=ps["val_accs"].get))
    chk("r4b test acc agrees with zero-shot table",
        f"{pct(ps['test_acc'])}\\% == {ps['best_prompt']} table cell",
        pct(ps["test_acc"]) == pct(zs["templates"][ps["best_prompt"]]["acc"]))

    # calibration prose (Discussion); bins are recomputed from the artifact so
    # the quoted confidence range tracks the data, not the manuscript.
    zsc, prc = cal["zero_shot"], cal["probe_full"]
    zconfs = [b["conf"] for b in zsc["bins"] if b["size"] > 0]
    chk("zero-shot confidence range", f"({min(zconfs):.2f}--{max(zconfs):.2f})")
    chk("zero-shot ECE text", f"(ECE) is {zsc['ece']:.2f}")
    chk("probe ECE text", f"ECE is {prc['ece']:.2f}")
    chk("zero-shot correct conf", f"correct predictions ({zsc['correct_conf']:.2f})")
    chk("zero-shot incorrect conf", f"incorrect ones ({zsc['incorrect_conf']:.2f})")
    chk("probe correct conf", f"correct predictions ({prc['correct_conf']:.2f})")
    chk("probe incorrect conf", f"incorrect ones ({prc['incorrect_conf']:.2f})")
    pgaps = [abs(b["gap"]) for b in prc["bins"] if b["size"] > 0]
    chk("probe bin gap range", f"{min(pgaps):.2f}--{max(pgaps):.2f}")
    # The zero-shot calibration run was produced on a pre-cleanup prompt build;
    # the manuscript must disclose both that provenance and the two numbers it
    # implies (stale build accuracy vs. the authoritative clean one).
    stale_n = round(zsc["accuracy"] / 100 * zs["n_test"])
    kg = zs["templates"]["KG-Symptom"]
    chk("calibration provenance caveat",
        f"pre-cleanup prompt build that scores {zsc['accuracy']:.1f}\\% "
        f"({stale_n}/{zs['n_test']})")
    chk("authoritative KG fraction",
        f"{kg['acc'] * 100:.1f}\\% ({kg['n_correct']}/{zs['n_test']})")
    chk("no stray TAB characters", "0 tabs", "\t" not in tex)

    # ---- negative checks: superseded values must be ABSENT -------------------
    # The positive checks above cannot catch a leftover from an earlier run --
    # they only confirm the new number is present somewhere. These assert the
    # numbers we replaced are gone, except where the paper deliberately cites an
    # old figure as provenance (86.4% in the TPDD image-provenance paragraph).
    negatives = [
        ("stale fine-tuned CLIP (96.7)", "96.7", 0),
        ("stale fine-tuned CLIP (94.8)", "94.8", 0),
        ("stale C-plateau (90.4)", "90.4", 0),
        ("stale CV ResNet (92.1)", "92.1", 0),
        ("stale CV CLIP (87.8)", "87.8", 0),
        # 86.4% is deliberately retained once: the Methods paragraph
        # \textbf{TPDD image provenance} cites it as the pre-loss value that
        # explains the 86.4 -> 87.7 shift. It must appear exactly once, inside
        # that paragraph, and nowhere else.
        ("86.4 only as TPDD provenance", "86.4", 1),
        # 2026 data-consistency pass: superseded zero-shot wording must stay
        # gone. The (?<![\d.]) boundary keeps "56.5\%" from satisfying "6.5\%".
        ("stale best-of-14 (13.0, intro)", "best of 14 templates (13.0", 0),
        ("stale best-of-14 (13.0, discussion)", "best of 14 templates: 13.0", 0),
        ("stale matches-baseline intro", "matches always-predicting-the-common-class (13.0", 0),
        ("stale binomial pairing", "EN-Photo and KG-Symptom", 0),
        ("stale below-baseline 4.3%", "4.3\\%", 0),
        ("stale below-baseline 6.5%", "6.5\\%", 0),
        ("stale validation-split 7.3%", "7.3\\%", 0),
        ("stale KG ties EN-Photo", "ties the plain EN-Photo caption", 0),
        ("stale KG tie 13.0%", "KG-Symptom, 13.0", 0),
        ("stale calibration range", "(0.20--0.45)", 0),
        ("stale binomial wording", "match the baseline exactly", 0),
        ("stale unpopulated-bins claim", "overconfident across all confidence bins", 0),
    ]
    # Sweep only PROSE. Per-class recall cells legitimately repeat values like
    # 96.7, so scanning tables would flag correct numbers as stale. norm() strips
    # spaces so multi-word needles match regardless of tex line wrapping.
    prose_only = norm(re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", tex, flags=re.S))
    for label, needle, allowed in negatives:
        hits = len(re.findall(r"(?<![\d.])" + re.escape(norm(needle)) + r"(?![\d])", prose_only))
        prov = tex.find(r"TPDD image provenance")
        in_provenance = ("86.4" not in needle) or (
            prov != -1
            and "86.4" in re.sub(r"\s+", " ", tex[prov:prov + 2600])
        )
        ok = hits == allowed and in_provenance
        checks.append((label, f"{hits} occurrence(s), allowed {allowed}",
                       ok))

    # ---- table cells must match the artifacts, cell by cell -----------------
    # The string-presence checks above are structurally unable to catch a wrong
    # TABLE: they only ask whether a number appears anywhere in the .tex. That is
    # how 11 of 14 rows of the zero-shot table and the whole ResNet column of the
    # per-class table were wrong while this gate still reported all-pass. Every
    # results table is therefore compared cell by cell.
    lines = tex.split("\n")

    def numeric(cells):
        return [float(re.findall(r"-?\d+\.?\d*", c)[0]) for c in cells
                if re.search(r"-?\d+\.?\d*", c)]

    def table_rows_between(start_marker, end_marker):
        """Rows of the ONE table delimited by the given literal markers.

        Scoping matters: template names like EN-Symptom also appear in the TLD
        and BioCLIP tables, so a global prefix search silently compares cells
        from different experiments.
        """
        i = tex.find(start_marker)
        if i < 0:
            return []
        j = tex.find(end_marker, i)
        if j < 0:
            return []
        blk = tex[i:j]
        out = []
        for ln in blk.split("\n"):
            # Skip only LaTeX STRUCTURAL lines. Skipping every line starting with
            # a backslash also drops \textbf{10} rows -- i.e. the two most
            # important classes in the per-class table.
            if "&" not in ln:
                continue
            if re.match(r"\s*\\(toprule|midrule|bottomrule|hline|cmidrule)", ln):
                continue
            out.append([c.strip() for c in ln.rstrip("\\").split("&")])
        return out

    # -- zero-shot template table (tab:prompts) --
    # delimiters: the "Lang." column header, up to the uniform-baseline row
    zs_rows = table_rows_between(r"\textbf{Lang.}", r"Uniform (1/16) baseline")
    name_map = {}
    for cells in zs_rows:
        m = re.match(r"([A-Za-z\-]+)", cells[0].strip())
        if m and m.group(1) in zs["templates"]:
            name_map[m.group(1)] = cells
    bad_zs = []
    for tname, cells in name_map.items():
        want = zs["templates"][tname]
        lo, hi = want["ci95"]
        got = numeric(cells[2:3])
        got_ci = [float(x) for x in re.findall(r"-?\d+\.?\d*", cells[3])] if len(cells) > 3 else []
        if (not got or abs(got[0] - want["acc"] * 100) > 0.05
                or len(got_ci) < 2
                or abs(got_ci[0] - lo * 100) > 0.05
                or abs(got_ci[1] - hi * 100) > 0.05):
            bad_zs.append(tname)
    checks.append((f"zero-shot table cells ({len(name_map)}/14 templates)",
                   f"{len(bad_zs)} wrong" + (f": {bad_zs}" if bad_zs else ""),
                   len(name_map) == 14 and not bad_zs))
    missing_zs = [t for t in zs["templates"] if t not in name_map]
    checks.append(("all 14 templates present in table",
                   f"missing {missing_zs}" if missing_zs else "none missing",
                   not missing_zs))

    # -- C-grid table --
    grid = sorted(float(x) for x in can["C_grid"])

    def table_row(prefix):
        for ln in lines:
            if ln.lstrip().startswith(prefix) and "&" in ln:
                return re.findall(r"\d+\.\d", re.sub(r"\\textbf\{|\}", "", ln))
        return None

    got_c = table_row("CLIP val.")
    got_r = table_row("Frozen ResNet val.")
    want_c = [f"{can['val_accs'][str(g)] * 100:.1f}" for g in grid]
    want_r = [f"{can['resnet_frozen_val_accs'][str(g)] * 100:.1f}" for g in grid]
    checks.append(("C-grid CLIP row cells", " ".join(got_c or ["<missing>"]),
                   got_c == want_c))
    checks.append(("C-grid ResNet row cells", " ".join(got_r or ["<missing>"]),
                   got_r == want_r))
    best = [g for g in grid
            if abs(can["val_accs"][str(g)] * 100
                   - can["val_accs"][str(can["best_C"])] * 100) < 0.05]
    clip_lines = [ln for ln in lines
                  if ln.lstrip().startswith("CLIP val.") and "&" in ln]
    nbold = sum(ln.count("\\textbf") for ln in clip_lines)
    checks.append(("C-grid bolds exactly the tied maxima",
                   f"{len(best)} tied, {nbold} bolded", nbold == len(best)))

    # -- per-class table: the ResNet column must come from the 15-seed
    #    statistics, not a single arbitrarily-drawn run --
    pc_rows = []
    for cells in table_rows_between("Class & $n$ & CLIP", "Balanced accuracy"):
        first = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cells[0]).strip()
        if re.fullmatch(r"\d{1,2}", first) and len(cells) >= 5:
            pc_rows.append(cells)
    n_pc = len(pc_rows)
    checks.append(("per-class table has 16 class rows", f"{n_pc} rows",
                   n_pc == 16))
    if n_pc == 16 and "perclass_mean" in art["resnet"]:
        want_pc = [v * 100 for v in art["resnet"]["perclass_mean"]]
        mism = []
        for i, cells in enumerate(pc_rows):
            vals = numeric(cells[1:5])          # n, clipC1, clipBestC, resnet
            if len(vals) >= 4 and abs(vals[3] - want_pc[i]) > 0.6:
                mism.append((i, round(vals[3], 1), round(want_pc[i], 1)))
        checks.append(("per-class ResNet column = 15-seed mean",
                       f"{len(mism)} rows differ" + (f" {mism[:3]}" if mism else ""),
                       not mism))
        # CLIP columns against canonical_results
        want_c1 = [v["acc"] * 100 for v in
                   (can["per_class"]["clip_linear_C1"].values()
                    if isinstance(can["per_class"]["clip_linear_C1"], dict) else [])]
        if len(want_c1) == 16:
            bad_c1 = [i for i, cells in enumerate(pc_rows)
                      if abs(numeric(cells[1:5])[1] - want_c1[i]) > 0.6]
            checks.append(("per-class CLIP C=1 column",
                           f"{len(bad_c1)} rows differ", not bad_c1))

    # -- confusion cells quoted in the Results prose -------------------------
    # A 2026-10-02 five-reviewer audit found two of the five off-diagonal
    # counts printed in the "Confusion structure" paragraph wrong (nematodes->
    # sunscald and weather_fleck->wildfire). Neither the per-class table check
    # above nor any other check covered confusion cells, so they drifted
    # silently. Derive every quoted cell from the canonical matrix and assert
    # both that the correct string is present and that the stale one is gone.
    cm = art["clip"]["linear_C1"].get("confusion")
    cn = zs.get("class_names")
    if cm and cn and len(cm) == len(cn) == 16:
        ci = {name: k for k, name in enumerate(cn)}

        def cell(t, p):
            return cm[ci[t]][ci[p]]

        def row(t):
            return sum(cm[ci[t]])

        for t, p in (("nematodes", "sunscald"), ("pvy", "cmv"),
                     ("target_spot", "brown_spot")):
            n, tot = cell(t, p), row(t)
            # Prefix form: the prose quotes these cells as "(N of T)" or
            # "(N of T, P%)" depending on the cell, so match the count pair.
            chk(f"confusion cell {t}->{p}", f"({n} of {tot}")
        # The two wildfire cells are quoted jointly as "(a of b and c of d)".
        w1, w1t = cell("weather_fleck", "wildfire"), row("weather_fleck")
        w2, w2t = cell("sunscald", "wildfire"), row("sunscald")
        chk("confusion cells -> wildfire (joint)",
            f"({w1} of {w1t} and {w2} of {w2t})")
        chk("no stale confusion count 5 of 8",
            "absent", "(5 of 8, 62\\%)" not in tex)
        chk("no stale confusion count 2 of 12 and 2 of 14",
            "absent", "(2 of 12 and 2 of 14)" not in tex)

    # -- default-probe gap is a hand-computed difference -------------------
    # Reported as 9.7 pp at one site while the abstract and Discussion said
    # 10.9 pp. Re-derive it and pin both the number and its absence.
    gap_df = (art["resnet"]["mean"] - art["clip"]["linear_C1"]["accuracy"]) * 100
    chk("default-probe gap (ResNet mean - CLIP C=1)", f"{gap_df:.1f}\\,pp")
    chk("no stale 9.7 pp gap", "absent", "gap of 9.7\\,pp" not in tex)

    # -- RBF-SVM must not be described as beating ResNet-18 -----------------
    # heads_a7.json gives 0.8696 vs the 15-seed ResNet mean 0.8773: the SVM is
    # ~0.8 pp BELOW the CNN. The prose previously claimed it was "modestly
    # above ... fine-tuned ResNet-18", inverting the comparison.
    ha = DATA / "heads_a7.json"
    if ha.exists():
        h = json.loads(ha.read_text())
        rbf = h["rbf_svm"]["test_acc"] * 100
        chk("RBF-SVM below ResNet-18 in the artifact",
            f"{rbf:.1f}\\% < {art['resnet']['mean'] * 100:.1f}\\%",
            rbf < art["resnet"]["mean"] * 100)
        chk("no claim the RBF-SVM beats ResNet-18", "absent",
            not re.search(r"RBF-SVM[^.]{0,120}?(exceeds|above)[^.]{0,60}?ResNet-18",
                          tex))
        # Semantic guard, not a magic phrase: the prose must say the 87.0% SVM
        # sits BELOW the 87.7% ResNet mean wherever it is compared to it.
        chk("RBF-SVM reported below the ResNet mean", "87.0\\% below the 87.7\\% mean",
            bool(re.search(r"RBF-SVM.{0,120}?below.{0,40}?87\.7", norm(tex))))

    # -- wildfire is bacterial, not abiotic ---------------------------------
    chk("wildfire classified as bacterial",
        "wildfire, a bacterial disease caused by \\textit{Pseudomonas syringae}")
    chk("no stale 'abiotic damage (sunscald, weather fleck, wildfire)'",
        "absent", "abiotic damage (sunscald, weather fleck, wildfire)" not in tex)

    # -- the misleading zero-shot artifact stays quarantined ----------------
    sup = DATA / "prompt_ablation.SUPERSEDED.md"
    chk("prompt_ablation.json carries a SUPERSEDED notice", "exists",
        sup.exists())
    if (DATA / "prompt_ablation.json").exists():
        pa = load("prompt_ablation.json")
        leaked = [x for x in pa if abs(x.get("accuracy", 0) - 100 / 138) < 1e-9]
        chk("prompt_ablation.json leak flagged in the notice",
            f"{len(leaked)} implausible value(s)", len(leaked) > 0)

    # ------------------------------------------------------------------
    # Adaptation-control rows (LP++ / CoOp) are computed on a SEPARATELY
    # cached feature set whose pixel encoding differs from the fresh
    # deterministic extraction behind the probe rows (feature cosine 0.83-0.97;
    # 76.1 vs 76.8 at full data for DFN-CLIP). The head-capacity claims are
    # therefore within-artifact, and the .tex must say so. These checks encode
    # the two ways that disclosure previously went wrong: quoting the canonical
    # LAION numbers as if they were the seed+500 draw, and describing the draw
    # as uniformly higher when it is lower for DFN and higher for LAION.
    lp_path = Path(__file__).resolve().parent.parent / "data" / "lp_plus_plus.json"
    if lp_path.exists():
        lp = json.loads(lp_path.read_text())
        draw = {ck: [lp["checkpoints"][ck][f"K={k}"]["probe_C1"]["mean"] * 100
                     for k in (1, 5, 10)]
                for ck in ("dfn_clip", "laion2b")}
        for ck in ("dfn_clip", "laion2b"):
            want = "/".join(f"{v:.1f}" for v in draw[ck])
            chk(f"LP++ {ck} seed+500 draw probe quoted",
                want, norm(want) in flat)
        # The direction claim must not be stated as uniform.
        _uniform = norm("higher at every $K$ except") in flat
        chk("LP++ draw-vs-canonical direction is not claimed uniform",
            f"{_uniform}", not _uniform)
        # Full-data gains must be within-artifact: LP++ minus its OWN probe.
        for ck in ("dfn_clip", "laion2b"):
            n = lp["checkpoints"][ck]["full"]
            gain = (n["lp_pp"]["mean"] - n["probe_C1"]["mean"]) * 100
            chk(f"LP++ {ck} full gain is within-artifact",
                f"+{gain:.1f} pp",
                norm(f"+{gain:.1f}$\\\\,pp") in flat
                or norm(f"+{gain:.1f}$ pp") in flat)
        chk("feature-encoding difference is disclosed",
            "pixel encoding differs",
            norm("pixel encoding differs") in flat)

    # ------------------------------------------------------------------
    # Two independent ResNet K-shot artifacts exist (different support draws).
    # paired_kshot_a4.json is the seed-paired store: the frozen-probe reference
    # at K=1 (and K=2 via paired_kshot_a4_k2.json). The matched-budget (P3b)
    # K=5/10 reference (64.9/74.6) is checked in the few-shot block above and is
    # what Table 1 quotes at K=5/10. This block only verifies the seed-paired
    # store still carries its expected values (store integrity, not the tex).
    paired_path = Path(__file__).resolve().parent.parent / "data" / "paired_kshot_a4.json"
    if paired_path.exists():
        pj = json.loads(paired_path.read_text())
        _pairs = {"K=1": 33.0, "K=5": 65.5, "K=10": 75.7}
        for k, claim in _pairs.items():
            chk(f"seed-paired ResNet store carries {k} reference",
                f"{pj[k]['resnet_mean'] * 100:.1f}",
                abs(pj[k]["resnet_mean"] * 100 - claim) <= 0.06)
        for k in (1, 5, 10):
            rp = Path(__file__).resolve().parent.parent / "data" / f"resnet_k{k}_n15.json"
            if rp.exists():
                rj = json.loads(rp.read_text())
                chk(f"resnet_k{k}_n15.json labelled non-authoritative",
                    "_provenance",
                    rj.get("_provenance", {}).get("role")
                    == "reference_only_non_paired_run")

    bad = [(l, n) for l, n, ok in checks if not ok]
    for label, needle, ok in checks[len(checks) - len(negatives) - 30:]:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:32s} {needle}")
    print(f"\n{len(checks) - len(bad)}/{len(checks)} consistent")
    if bad:
        print("\nSTALE / MISSING NUMBERS:")
        for label, needle in bad:
            print(f"  - {label}: expected {needle!r} not found in the .tex")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
