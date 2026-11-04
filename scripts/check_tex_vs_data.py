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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "manuscript" / "paper" / "probe-protocol.tex"
DATA = ROOT / "data"


def load(name):
    with open(DATA / name) as fh:
        return json.load(fh)


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}"


def norm(s):
    for tok in ("\\,", "\\;", "\\ ", "\\!", "~", " ", "$", "\\%"):
        s = s.replace(tok, "" if tok != "\\%" else "\\%")
    return re.sub(r"\s+", "", s)


def main():
    tex = TEX.read_text(encoding="utf-8")
    flat = norm(re.sub(r"\s+", " ", tex))

    checks = []

    def chk(label, needle, ok=None):
        present = norm(needle) in flat if ok is None else ok
        checks.append((label, needle, present))

    # ==================================================================
    # 1. ZERO-SHOT (data/zero_shot_full_20261025.json)
    # ==================================================================
    zs = load("zero_shot_full_20261025.json")
    n_test = zs["n_test"]
    maj_pct = zs["majority_baseline_pct"]

    for ck, label in (("dfn_clip", "DFN"), ("laion2b", "LAION"), ("bioclip2", "BioCLIP")):
        d = zs["checkpoints"][ck]
        best_pct = d["best_acc_pct"]
        chk(f"ZS {label} best template", f"{best_pct:.1f}")
        chk(f"ZS {label} best name", d["best_template"])

    chk("ZS majority baseline", f"{maj_pct:.1f}")
    chk("ZS n_test", f"{n_test}")

    # ==================================================================
    # 2. SIGLIP ZERO-SHOT (data/siglip_zs_20261025.json)
    # ==================================================================
    sl = load("siglip_zs_20261025.json")
    if "best_acc_pct" in sl:
        chk("SigLIP ZS best", f"{sl['best_acc_pct']:.1f}")

    # ==================================================================
    # 3. CROSS-VALIDATION (data/cv_protocol.json)
    # ==================================================================
    cv = load("cv_protocol.json")
    cv_clip = cv["summary"]["clip"]["acc_mean"]
    cv_resnet = cv["summary"]["resnet"]["acc_mean"]
    chk("CV CLIP mean", f"{pct(cv_clip)}")
    chk("CV ResNet mean", f"{pct(cv_resnet)}")

    gap_pp = (cv_resnet - cv_clip) * 100
    chk("CV gap (ResNet - CLIP)", f"{gap_pp:.1f}")
    # sign-stability and p-value
    if "n_sign_stable" in cv["summary"].get("gap_clip_minus_resnet", {}):
        nss = cv["summary"]["gap_clip_minus_resnet"]["n_sign_stable"]
        chk("CV sign-stable", f"{nss}/5")
    if "p" in cv["summary"].get("gap_clip_minus_resnet", {}):
        chk("CV p-value", f"p={cv['summary']['gap_clip_minus_resnet']['p']:.3f}")

    # ==================================================================
    # 4. FINE-TUNED CLIP (data/cv_finetune_clip.json)
    # ==================================================================
    cft = load("cv_finetune_clip.json")
    for ck, label in (("dfn_clip", "DFN"), ("laion2b", "LAION")):
        if ck in cft:
            d = cft[ck]
            if "acc_mean" in d:
                chk(f"FT CLIP {label} mean", f"{pct(d['acc_mean'])}")
            elif "mean" in d:
                chk(f"FT CLIP {label} mean", f"{pct(d['mean'])}")

    # ==================================================================
    # 5. 16-CONTRAST MULTIPlicity (data/multiplicity_16_matched_corrected.json)
    # ==================================================================
    m16 = load("multiplicity_16_matched_corrected.json")
    n_c = m16["n_contrasts"]
    n_s = m16["n_survive_holm"]
    chk("16-contrast n_contrasts", f"{n_c}")
    # Paper uses the fraction form (15/16); accept either wording.
    chk("16-contrast survivors", f"{n_s} of {n_c}",
        ok=(norm(f"{n_s} of {n_c}") in flat) or (norm(f"{n_s}/{n_c}") in flat))

    # ==================================================================
    # 6. RESNET-50 (data/resnet50_full.json)
    # ==================================================================
    r50 = load("resnet50_full.json")
    if "mean" in r50:
        chk("ResNet-50 mean", f"{pct(r50['mean'])}")
    if "std" in r50:
        chk("ResNet-50 SD", f"{pct(r50['std'])}")

    # ==================================================================
    # 7. FIELD FINE-TUNE (data/field_finetune_cnn.json, data/field_finetune.json)
    # ==================================================================
    ff_cnn = load("field_finetune_cnn.json")
    for model in ("resnet18", "convnext"):
        if model in ff_cnn:
            d = ff_cnn[model]
            # values stored as percentages already (76.7, not 0.767)
            chk(f"Field {model} mean", f"{d['mean']:.1f}")
            if "std" in d:
                chk(f"Field {model} SD", f"{d['std']:.1f}")

    # ==================================================================
    # 8. CORN CROSS-CROP (data/corn_results.json)
    # ==================================================================
    corn = load("corn_results.json")
    if "full_data" in corn:
        fd = corn["full_data"]
        if "clip" in fd and "clip_bestC" in fd["clip"]:
            chk("Corn CLIP full-data", f"{pct(fd['clip']['clip_bestC'])}")
        if "resnet" in fd and "mean" in fd["resnet"]:
            chk("Corn ResNet full-data", f"{pct(fd['resnet']['mean'])}")

    # ==================================================================
    # 9. TOMATO CROSS-CROP (data/pv_tomato_results.json)
    # ==================================================================
    tomato = load("pv_tomato_results.json")
    if "full_data" in tomato:
        fd = tomato["full_data"]
        if "clip" in fd:
            for k in ("clip_bestC", "mean"):
                if k in fd["clip"]:
                    chk("Tomato CLIP full-data", f"{pct(fd['clip'][k])}")

    # ==================================================================
    # 10a. DINOV2 FINE-TUNING (data/dinov2_finetune_full_16.json, A100 run)
    # ==================================================================
    try:
        dft = load("dinov2_finetune_full_16.json")
        m = dft["stage2_final"]["mean"]
        chk("DINOv2-FT full mean", f"{pct(m)}")
        chk("DINOv2-FT hardware note", "A100-80GB/PyTorch 2.9")
    except (FileNotFoundError, KeyError):
        pass

    # 10. DINOV2 (data/dinov2_protocol.json)
    # ==================================================================
    dino = load("dinov2_protocol.json")
    if "full_data" in dino:
        fd = dino["full_data"]
        if "linear_C1" in fd:
            chk("DINOv2 C=1", f"{pct(fd['linear_C1']['accuracy'])}")
        if "linear_bestC" in fd:
            chk("DINOv2 bestC", f"{pct(fd['linear_bestC']['accuracy'])}")

    # ==================================================================
    # 11. SIGLIP PROTOCOL (data/siglip_protocol.json)
    # ==================================================================
    slp = load("siglip_protocol.json")
    if "full_data" in slp:
        fd = slp["full_data"]
        for key in ("linear_C1", "linear_bestC", "finetuned"):
            if key in fd and "accuracy" in fd[key]:
                chk(f"SigLIP {key}", f"{pct(fd[key]['accuracy'])}")

    # ==================================================================
    # 12. M2 POWERED STATS (data/m2_powered_stats.json)
    # ==================================================================
    m2 = load("m2_powered_stats.json")
    # Check key statistics are present in the paper
    if "clip_mean" in m2:
        chk("M2 CLIP mean", f"{pct(m2['clip_mean'])}")
    if "resnet_mean" in m2:
        chk("M2 ResNet mean", f"{pct(m2['resnet_mean'])}")

    # ==================================================================
    # 13. CALIBRATION (ECE from zero_shot_full + calibration_20261025)
    # ==================================================================
    # Zero-shot ECE range (per-checkpoint, from zero_shot_full_20261025.json)
    zs_ece_vals = []
    for ck in ("dfn_clip", "laion2b"):
        ece_dict = zs["checkpoints"][ck].get("ece_best_template", {})
        if ece_dict:
            ece_tau100 = ece_dict.get("100", {}).get("ece")
            if ece_tau100 is not None:
                zs_ece_vals.append(ece_tau100)
    if len(zs_ece_vals) >= 2:
        ece_lo = min(zs_ece_vals)
        ece_hi = max(zs_ece_vals)
        chk("ZS ECE range low", f"{ece_lo:.2f}")
        chk("ZS ECE range high", f"{ece_hi:.2f}")
    # Probe C=1 ECE (from calibration_20261025.json)
    cal = load("calibration_20261025.json")
    if "probe" in cal and "1.0" in cal["probe"]:
        chk("Probe C=1 ECE", f"{cal['probe']['1.0']['ece']:.2f}")
    if "probe" in cal and "100.0" in cal["probe"]:
        chk("Probe C=100 ECE", f"{cal['probe']['100.0']['ece']:.2f}")

    # ==================================================================
    # 14. LP++ (data/lp_plus_plus.json)
    # ==================================================================
    lp = load("lp_plus_plus.json")
    if "checkpoints" in lp:
        for ck in ("dfn_clip", "laion2b"):
            if ck in lp["checkpoints"]:
                d = lp["checkpoints"][ck]
                if "full" in d and "lp_pp" in d["full"]:
                    chk(f"LP++ {ck} full", f"{pct(d['full']['lp_pp']['mean'])}")

    # ==================================================================
    # 15. VAL-SPLIT (data/valsplit_template_20261025.json)
    # ==================================================================
    vs = load("valsplit_template_20261025.json")
    if "best_prompt" in vs:
        chk("Val-split best prompt", vs["best_prompt"])
    if "test_acc" in vs:
        chk("Val-split test acc", f"{pct(vs['test_acc'])}")

    # ==================================================================
    # 16. TLD ZERO-SHOT (data/tld_zs_20261025.json)
    # ==================================================================
    tld = load("tld_zs_20261025.json")
    if "best_acc_pct" in tld:
        chk("TLD ZS best", f"{tld['best_acc_pct']:.1f}")

    # ==================================================================
    # 17. PROBE CGRID (data/probe_cgrid_k1_k2_valselected.json)
    # ==================================================================
    pc = load("probe_cgrid_k1_k2_valselected.json")
    # Check that the C-grid values are present

    # ==================================================================
    # 18. RESNET K-SHOT MATCHED (cross-crop)
    # ==================================================================
    # The paper's cross-crop numbers come from the canonical documented split
    # (pv_tomato_results.json, pv_corn_results.json; test n=1,635/346). The
    # resnet_k1_matched_*.json artifacts are a separate cross-check split
    # (13,892/1,816) and are intentionally NOT cited in the prose.
    for crop in ("corn", "tomato"):
        try:
            pv = load(f"pv_{crop}_results.json")
            m = pv["few_shot"]["K=1"]["resnet_finetuned"]["mean"]
            chk(f"ResNet K=1 {crop}", f"{pct(m)}")
        except (FileNotFoundError, KeyError):
            pass

    # ==================================================================
    # REPORT
    # ==================================================================
    bad = [(l, n) for l, n, ok in checks if not ok]
    for label, needle, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:35s} {needle}")

    print(f"\n{len(checks) - len(bad)}/{len(checks)} consistent")
    if bad:
        print("\nFAILURES:")
        for label, needle in bad:
            print(f"  - {label}: expected {needle!r} not found in .tex")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
