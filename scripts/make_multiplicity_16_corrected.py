"""Generator for data/multiplicity_16_matched_corrected.json.

Builds the 16-contrast multiplicity family (paired, matched-budget) from the raw
per-seed stores and recomputes, for each contrast:
  - delta_pp  : mean of per-seed (CLIP - ResNet) accuracy differences, in pp
  - p_two     : two-sided paired t-test p-value (n=15 seeds)
  - wtl       : [wins, ties, losses] for CLIP across seeds
then applies Holm-Bonferroni across the whole family of 16 and flags survival
at alpha = 0.05.

The 8 frozen-probe contrasts pair clip_fewshot.json / laion_fewshot_canonical.json
against p3b_resnet_kshot_matched_k{1,2,5,10}.json; the 8 fine-tuned contrasts pair
the p3b/p3c matched fine-tuning stores against the same ResNet stores.

Sign convention: delta_pp > 0 means CLIP leads; < 0 means ResNet leads.

Pairing precondition (asserted at run time in every source store): CLIP and
ResNet use the identical per-seed support sets
(kshot_subset_idx(y, K, seed+500), offset 500), so per-seed differences are
within-subject.

Default mode verifies the on-disk file against a fresh recomputation (fails
loudly on any drift); --write regenerates the file.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

K_SHOTS = [1, 2, 5, 10]
OUTFILE = DATA / "multiplicity_16_matched_corrected.json"
DESCRIPTION = "Corrected 16-contrast multiplicity family with two-sided paired t-test p-values"

DFN_FROZEN_STORE = "clip_fewshot.json"
LAION_FROZEN_STORE = "laion_fewshot_canonical.json"
CLIP_FT_STORES = {
    (1, "dfn"): "p3b_clip_finetune_kshot_k1_dfn.json",
    (1, "laion"): "p3b_clip_finetune_kshot_k1_laion.json",
    (2, "dfn"): "p3b_clip_finetune_kshot_k2_dfn.json",
    (2, "laion"): "p3b_clip_finetune_kshot_k2_laion.json",
    (5, "dfn"): "p3c_clip_finetune_kshot_dfn.json",
    (5, "laion"): "p3b_clip_finetune_kshot.json",
    (10, "dfn"): "p3c_clip_finetune_kshot_dfn_k10.json",
    (10, "laion"): "p3b_clip_finetune_kshot_k10.json",
}
RESNET_STORES = {
    1: "p3b_resnet_kshot_matched_k1.json",
    2: "p3b_resnet_kshot_matched_k2.json",
    5: "p3b_resnet_kshot_matched.json",
    10: "p3b_resnet_kshot_matched_k10.json",
}


def check_paired_precondition(store_a, store_b, label):
    assert len(store_a["stage2_final"]["accs"]) == len(store_b["stage2_final"]["accs"]) == 15, label
    assert store_a["seed_offset"] == store_b["seed_offset"] == 500, label
    assert store_a["support_draw"] == store_b["support_draw"], label


def load_accs(path, is_stage2):
    d = json.loads(path.read_text())
    if is_stage2:
        return d["stage2_final"]["accs"], d
    return [d[f"K={k}"]["accs"] for k in K_SHOTS], d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="regenerate multiplicity_16_matched_corrected.json (default: verify only)")
    args = ap.parse_args()

    resnet_j = {k: json.loads((DATA / f).read_text()) for k, f in RESNET_STORES.items()}
    dfn_frozen_j = json.loads((DATA / DFN_FROZEN_STORE).read_text())
    laion_frozen_j = json.loads((DATA / LAION_FROZEN_STORE).read_text())
    clip_ft_j = {kk: json.loads((DATA / f).read_text()) for kk, f in CLIP_FT_STORES.items()}

    contrasts = []

    for k in K_SHOTS:
        resnet_accs = np.asarray(resnet_j[k]["stage2_final"]["accs"], dtype=float)

        for ckpt, frozen_j in (("DFN", dfn_frozen_j), ("LAION", laion_frozen_j)):
            frozen_accs = np.asarray(frozen_j[f"K={k}"]["accs"], dtype=float)
            label = f"K={k} {ckpt} frozen"
            assert len(frozen_accs) == 15, label
            assert isinstance(frozen_j.get("_protocol") or frozen_j.get("protocol"), str), label
            contrasts.append((label, frozen_accs - resnet_accs))

        for ckpt in ("DFN", "LAION"):
            ft_store = clip_ft_j[(k, ckpt.lower())]
            ft_accs = np.asarray(ft_store["stage2_final"]["accs"], dtype=float)
            label = f"K={k} {ckpt} FT"
            check_paired_precondition(ft_store, resnet_j[k], label)
            contrasts.append((label, ft_accs - resnet_accs))

    assert len(contrasts) == 16

    systematics = []
    for label, diffs in contrasts:
        n = len(diffs)
        t, p_two = stats.ttest_rel(diffs, np.zeros(n))
        delta_pp = 100.0 * float(np.mean(diffs))
        wtl = [int(np.sum(diffs > 0)), int(np.sum(diffs == 0)), int(np.sum(diffs < 0))]
        systematics.append({
            "name": label,
            "p_two": float(p_two),
            "delta_pp": delta_pp,
            "wtl": wtl,
        })

    # Holm-Bonferroni over the full family of 16, family-wide alpha = 0.05.
    m = len(systematics)
    order = sorted(range(m), key=lambda i: systematics[i]["p_two"])
    running_max = 0.0
    p_holm = [0.0] * m
    for rank, idx in enumerate(order):
        adj = (m - rank) * systematics[idx]["p_two"]
        running_max = max(running_max, adj)
        p_holm[idx] = running_max

    for i, sys_item in enumerate(systematics):
        sys_item["p_holm"] = p_holm[i]
        sys_item["survive"] = bool(sys_item["p_holm"] <= 0.05)

    systematics.sort(key=lambda s: s["p_two"])
    regen = {
        "description": DESCRIPTION,
        "n_contrasts": m,
        "n_survive_holm": sum(s["survive"] for s in systematics),
        "contrasts": systematics,
    }

    print(f"{'contrast':<20} {'delta_pp':>10} {'p_two':>12} {'p_holm':>12} survive")
    for s in systematics:
        print(f"{s['name']:<20} {s['delta_pp']:+10.4f} {s['p_two']:12.3e} {s['p_holm']:12.3e} "
              f"{'yes' if s['survive'] else 'NO'}")
    print(f"\nn_survive_holm = {regen['n_survive_holm']} / {m}")

    if not OUTFILE.exists():
        if not args.write:
            print(f"\n{OUTFILE.relative_to(ROOT)} does not exist; rerun with --write to create it.", file=sys.stderr)
            return 1
        OUTFILE.write_text(json.dumps(regen, indent=2) + "\n")
        print(f"\nWrote {OUTFILE.relative_to(ROOT)}.")
        return 0

    existing = json.loads(OUTFILE.read_text())
    ok = True
    if existing.get("description") != regen["description"]:
        print(f"MISMATCH description:\n  disk: {existing.get('description')!r}\n  recomp: {regen['description']!r}")
        ok = False
    for key, err in (("n_contrasts", 0), ("n_survive_holm", 0)):
        if existing.get(key) != regen[key]:
            print(f"MISMATCH {key}: disk={existing.get(key)} recomp={regen[key]}")
            ok = False
    disk_names = [s["name"] for s in existing["contrasts"]]
    recomp_names = [s["name"] for s in regen["contrasts"]]
    if disk_names != recomp_names:
        print(f"MISMATCH contrast names/order:\n  disk:   {disk_names}\n  recomp: {recomp_names}")
        ok = False
    for ds, rs in zip(existing["contrasts"], regen["contrasts"]):
        for key in ("delta_pp", "p_two", "p_holm"):
            tol = 1e-9 * max(1.0, abs(rs[key]))
            if abs(ds[key] - rs[key]) > tol:
                print(f"MISMATCH {ds['name']}.{key}: disk={ds[key]!r} recomp={rs[key]!r}")
                ok = False
        if ds["wtl"] != rs["wtl"] or ds["survive"] != rs["survive"]:
            print(f"MISMATCH {ds['name']} wtl/survive: disk={ds['wtl']}/{ds['survive']} "
                  f"recomp={rs['wtl']}/{rs['survive']}")
            ok = False

    if ok:
        print(f"\nVERIFIED: fresh recomputation matches {OUTFILE.relative_to(ROOT)} "
              f"(tolerance 1e-9).")
        if args.write:
            text = json.dumps(regen, indent=2) + "\n"
            if text == OUTFILE.read_text():
                print("Byte-identical; file left untouched.")
            else:
                OUTFILE.write_text(text)
                print("Rewrote file (values matched, formatting refreshed).")
        return 0

    if args.write:
        OUTFILE.write_text(json.dumps(regen, indent=2) + "\n")
        print(f"\nMismatch detected; {OUTFILE.relative_to(ROOT)} overwritten with recomputation.", file=sys.stderr)
        return 2
    print("\nFAILED verification:", file=sys.stderr)
    print("  - if the recomputation is wrong, fix this script;", file=sys.stderr)
    print("  - if the recomputation is right, rerun with --write and re-run the consistency gate.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
