"""
Systematic multiple-comparisons correction (reviewer #8 fix).

Defines ONE explicit family -- the six few-shot CLIP-probe vs fine-tuned ResNet-18
seed-paired contrasts (K in {1,5,10} x {DFN-CLIP, LAION-2B}) -- and reports how many
survive (a) Holm-Bonferroni (FWER) and (b) Benjamini-Hochberg (FDR) at alpha=0.05.

The six paired-t p-values are read from data/paired_kshot_a4_both.json (the single source
of truth for the seed-paired K-shot contrasts already cited in the manuscript). This makes
the already-present Bonferroni x2 (K=1 only) into one corrected family over all headline
few-shot contrasts, and states clearly which survive and which do not.

Outputs data/multiplicity.json. CPU-only.
"""
import os, json, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "data", "paired_kshot_a4_both.json")
OUT = os.path.join(REPO, "data", "multiplicity.json")
ALPHA = 0.05

# family order (as cited in the manuscript); nested keys in the source json
FAMILY = [
    ("K=1",  "laion2b_vs_resnet", "LAION-2B"),
    ("K=1",  "dfn_vs_resnet",     "DFN-CLIP"),
    ("K=5",  "laion2b_vs_resnet", "LAION-2B"),
    ("K=5",  "dfn_vs_resnet",     "DFN-CLIP"),
    ("K=10", "laion2b_vs_resnet", "LAION-2B"),
    ("K=10", "dfn_vs_resnet",     "DFN-CLIP"),
]


def holm(pvals):
    """Return (rejected, adjusted_p) using Holm step-down on the given list."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    rejected = [False] * m
    running = True
    for r, i in enumerate(order, start=1):
        thr = ALPHA / (m - r + 1)
        if running:
            if pvals[i] <= thr:
                rejected[i] = True
            else:
                # once a non-rejection occurs, all remaining (larger) are non-rejected
                running = False
    # Holm adjusted p: the max of the step-down scaled p-values up to and
    # including rank r, i.e. adj(r) = max_{j <= r} min(1, (m - j + 1) * p_(j)).
    # Enforced as a running max from the SMALLEST p upward, so adjusted p stays
    # monotone non-decreasing in p. (Running max from the largest would collapse
    # every entry onto the largest p-value.)
    adj2 = [0.0] * m
    running_max = 0.0
    for r, i in enumerate(order, start=1):
        running_max = max(running_max, min(1.0, (m - r + 1) * pvals[i]))
        adj2[i] = running_max
    return rejected, adj2


def benjamini_hochberg(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    for r, i in enumerate(order, start=1):
        q[i] = pvals[i] * m / r
    for r in range(len(order) - 2, -1, -1):
        q[order[r]] = min(q[order[r]], q[order[r + 1]])
    rejected = [q[i] <= ALPHA for i in range(m)]
    return rejected, q


def main():
    d = json.load(open(SRC))
    rows = []
    for kval, sub, ckpt in FAMILY:
        blk = d[kval][sub]
        rows.append({
            "contrast": f"{kval}: CLIP ({ckpt}) vs ResNet-18",
            "clip_acc_pct": round(blk["clip"], 2),
            "resnet_acc_pct": round(blk["resnet"], 2),
            "delta_pp": round(blk["delta"], 2),
            "paired_t_p": blk["t_p"],
        })
    pvals = [r["paired_t_p"] for r in rows]
    holm_rej, holm_adj = holm(pvals)
    bh_rej, bh_q = benjamini_hochberg(pvals)
    for i, r in enumerate(rows):
        r["holm_significant"] = bool(holm_rej[i])
        r["holm_adjusted_p"] = holm_adj[i]
        r["bh_fdr_significant"] = bool(bh_rej[i])
        r["bh_qvalue"] = bh_q[i]

    survivors = [r["contrast"] for r, ok in zip(rows, holm_rej) if ok]
    out = {
        "family": "six few-shot CLIP-probe vs fine-tuned ResNet-18 seed-paired contrasts "
                  "(K in {1,5,10} x {DFN-CLIP, LAION-2B}); paired-t p-values from "
                  "data/paired_kshot_a4_both.json",
        "alpha": ALPHA,
        "n_tests": len(rows),
        "n_holm_surviving": int(sum(holm_rej)),
        "n_bh_surviving": int(sum(bh_rej)),
        "rows": rows,
        "note": (f"Holm-Bonferroni controls FWER; Benjamini-Hochberg controls FDR. "
                 f"{int(sum(holm_rej))}/{len(rows)} contrasts survive Holm and "
                 f"{int(sum(bh_rej))}/{len(rows)} survive BH; the survivors are "
                 f"{survivors}. Generated from the current p-values, so this note "
                 f"cannot drift from the table above it."),
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"family of {len(rows)} contrasts; alpha={ALPHA}")
    print(f"Holm-Bonferroni: {int(sum(holm_rej))}/{len(rows)} survive")
    print(f"BH-FDR:          {int(sum(bh_rej))}/{len(rows)} survive")
    for r in rows:
        print(f"  {r['contrast']:45s} p={r['paired_t_p']:<12.3e} "
              f"holm={'YES' if r['holm_significant'] else 'no '} (adj {r['holm_adjusted_p']:.3g}) "
              f"bh={'YES' if r['bh_fdr_significant'] else 'no '} (q {r['bh_qvalue']:.3g})")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
