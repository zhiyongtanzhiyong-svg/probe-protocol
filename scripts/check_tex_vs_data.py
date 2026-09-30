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

    bad = [(l, n) for l, n, ok in checks if not ok]
    for label, needle, ok in checks[len(checks) - len(negatives) - 8:]:
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
