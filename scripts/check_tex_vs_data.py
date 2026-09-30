"""Cross-check the manuscript's reported numbers against the experiment JSONs.

The paper's claim is that data/ is the single source of truth. This script
enforces that mechanically: it derives every headline number from the JSON
artifacts and asserts that the same number appears in the .tex. It fails loudly
rather than warning, so a stale constant cannot survive an edit unnoticed.

Run:  python3 scripts/check_tex_vs_data.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "manuscript" / "paper" / "clip_vs_resnet_tobacco.tex"
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
    zs = load("zero_shot_clean.json")
    cv = load("cv_protocol.json")
    dino = load("dinov2_protocol.json")
    dino518 = load("dinov2_native518.json")
    mult = load("multiplicity.json")
    ver = load("claim_verification.json")
    fs = load("clip_fewshot.json")
    la = load("two_model_e4.json")["few_shot_probe"]
    pk = load("paired_kshot_a4.json")
    ft = load("clip_finetune.json")

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
        chk(f"ResNet {k}", f"{pk[k]['resnet_mean'] * 100:.1f}\\%")

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

    # ---- DINOv2 ------------------------------------------------------------
    chk("DINO224 C=1", f"{dino['full_data']['linear_C1']['accuracy'] * 100:.1f}\\%")
    chk("DINO224 bestC", f"{dino['full_data']['linear_bestC']['accuracy'] * 100:.1f}\\%")
    chk("DINO518 C=1", f"{dino518['full_data']['linear_C1']['accuracy'] * 100:.1f}\\%")
    chk("DINO518 bestC", f"{dino518['full_data']['linear_bestC']['accuracy'] * 100:.1f}\\%")

    # ---- multiplicity ------------------------------------------------------
    chk("Holm survivors", f"{mult['n_holm_surviving']} of {mult['n_tests']}")
    # Holm adjusted p must follow the step-down formula AND stay monotone in the
    # raw p, so the generator bug cannot come back silently:
    #   adj(r) = max_{j<=r} min(1, (m - j + 1) * p_(j))
    rows = sorted(mult["rows"], key=lambda r: r["paired_t_p"])
    m = len(rows)
    ok_adj, run = True, 0.0
    for r, row in enumerate(rows, start=1):
        run = max(run, min(1.0, (m - r + 1) * row["paired_t_p"]))
        if abs(row["holm_adjusted_p"] - run) > 1e-12:
            ok_adj = False
    chk("Holm adjusted p = step-down running max", "n/a", ok_adj)
    # rejection flags must agree with the adjusted p at alpha
    ok_rej = all(r["holm_significant"] == (r["holm_adjusted_p"] <= mult["alpha"])
                 for r in mult["rows"])
    chk("Holm flags agree with adjusted p", "n/a", ok_rej)

    # ---- paired claims -----------------------------------------------------
    c1 = ver["Q2_full_data"]["clip_C1"]
    bc = ver["Q2_full_data"]["clip_bestC"]
    chk("Q2 C=1 delta", f"{c1['delta_pp']:.1f}\\,pp")
    chk("Q2 C=1 p", f"p={c1['mcnemar_p_exact']:.3f}")
    chk("Q2 C=1 CI low", f"{c1['ci95_delta_pp'][0]:.1f}")
    chk("Q3 bestC delta", f"{abs(bc['delta_pp']):.1f}\\,pp")
    chk("Q3 bestC p", f"p={bc['mcnemar_p_exact']:.2f}")
    chk("Q3 bestC n_seeds", f"{bc['n_seeds_ge_clip']} of {bc['n_resnet_seeds']}")
    chk("Q2 C=1 all seeds exceed probe",
        f"all {c1['n_seeds_ge_clip']} of {c1['n_resnet_seeds']} ResNet seeds")

    bad = [(l, n) for l, n, ok in checks if not ok]
    for label, needle, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:32s} {needle}")

    # ---- negative checks: superseded values must be ABSENT -------------------
    # The positive checks above cannot catch a leftover from an earlier run --
    # they only confirm the new number is present somewhere. These assert the
    # numbers we replaced are gone, except where the paper deliberately cites an
    # old figure as provenance (86.4% in the Fig. 4 caption).
    negatives = [
        ("stale fine-tuned CLIP (96.7)", "96.7", 0),
        ("stale fine-tuned CLIP (94.8)", "94.8", 0),
        ("stale C-plateau (90.4)", "90.4", 0),
        ("stale CV ResNet (92.1)", "92.1", 0),
        ("stale CV CLIP (87.8)", "87.8", 0),
        # 86.4% is deliberately retained: the Fig. 4 caption cites it as the
        # pre-rerun value to explain the 86.4 -> 87.7 shift. It must appear
        # exactly once, in that caption, and nowhere else.
        ("86.4 only as Fig.4 provenance", "86.4", 1),
    ]
    for label, needle, allowed in negatives:
        hits = len(re.findall(r"(?<![\d.])" + re.escape(needle) + r"(?![\d])", flat))
        in_caption = ("86.4" not in needle) or (
            "86.4" in re.sub(r"\s+", " ", tex[tex.find(r"\caption{Confusion"):])
        )
        ok = hits == allowed and in_caption
        checks.append((label, f"{hits} occurrence(s), allowed {allowed}",
                       ok))

    # ---- table cells must match the artifacts, cell by cell -----------------
    # A stale \begin{tabular} row can silently contradict corrected prose, so the
    # C-grid table is regenerated from the JSON and compared exactly.
    grid = sorted(float(x) for x in can["C_grid"])
    lines = tex.split("\n")

    def table_row(prefix):
        """The tabular row starting with `prefix` (not the prose that mentions it)."""
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
    # the bolded maximum must sit on an actually-tied-optimal C
    best = [g for g in grid
            if abs(can["val_accs"][str(g)] * 100
                   - can["val_accs"][str(can["best_C"])] * 100) < 0.05]
    clip_lines = [ln for ln in lines
                  if ln.lstrip().startswith("CLIP val.") and "&" in ln]
    nbold = sum(ln.count("\\textbf") for ln in clip_lines)
    checks.append(("C-grid bolds exactly the tied maxima",
                   f"{len(best)} tied, {nbold} bolded", nbold == len(best)))

    bad = [(l, n) for l, n, ok in checks if not ok]
    for label, needle, ok in checks[len(checks) - len(negatives) - 4:]:
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
