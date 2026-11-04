"""Consolidated seed-paired few-shot contrasts, now including K=2.

Replaces data/multiplicity.json (K in {1,5,10}, 6 contrasts) with
data/multiplicity_k2.json (K in {1,2,5,10}, 8 contrasts).

Inputs and their provenances
----------------------------
  DFN CLIP per-seed   data/clip_fewshot.json        single caliber as of 2026-10-05
                                                   (EVAL support, fresh features)
  LAION CLIP per-seed data/two_model_e4.json        encodes fresh, no cache reuse
  ResNet per-seed K in {1,5,10}
                     data/paired_kshot_a4.json      original paired run
  ResNet per-seed K=2
                     data/paired_kshot_a4_k2.json   added 2026-10-05

All four sources draw the same support indices per seed (seed+500, stratified,
K per class). Verified because data/tpdd/*/labels.json is class-sorted, so the
dict-insertion order used by the KShotSubset classes equals the sorted() order
used by coop_experiment.kshot_subset_idx and two_model_e4.kshot_idx.

Reports per-contrast paired t, Wilcoxon (Pratt, ties-inclusive), bootstrap 95% CI
on the mean difference, and seed-level win/tie/loss, plus Holm-Bonferroni FWER
and Benjamini-Hochberg FDR across the family of eight.
"""
import os
import sys, json
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO = ROOT
ALPHA = 0.05
B = 10000
BOOT_SEED = 0


def load(p):
    return json.load(open(f'{REPO}/data/{p}'))


def holm(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def bh(pvals):
    n = len(pvals)
    order = np.argsort(pvals)[::-1]
    q = np.empty(n)
    running = 1.0
    for rank, i in enumerate(order):
        val = n * pvals[i] / (n - rank)
        running = min(running, val)
        q[i] = min(running, 1.0)
    return q


def contrast(clip_accs, rn_accs):
    c, r = np.asarray(clip_accs), np.asarray(rn_accs)
    d = c - r
    tt = stats.ttest_rel(c, r)
    try:
        ww = stats.wilcoxon(c, r, zero_method='pratt')
        w_p = float(ww[1])
    except Exception:
        w_p = float('nan')
    rng = np.random.RandomState(BOOT_SEED)
    boots = d[rng.randint(0, len(d), (B, len(d)))].mean(axis=1)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return {
        'clip_acc_pct': round(float(c.mean()) * 100, 2),
        'resnet_acc_pct': round(float(r.mean()) * 100, 2),
        'resnet_std_pct': round(float(r.std(ddof=1)) * 100, 2),
        'delta_pp': round(float(d.mean()) * 100, 2),
        'delta_ci95_bootstrap_pp': [round(ci[0] * 100, 2), round(ci[1] * 100, 2)],
        'paired_t': round(float(tt.statistic), 3),
        'paired_t_p': float(tt.pvalue),
        'wilcoxon_p': w_p,
        'wins_ties_losses': [int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())],
    }


def main():
    dfn = load('clip_fewshot.json')
    laion = load('laion_fewshot_canonical.json')
    rn_prior = load('paired_kshot_a4.json')
    rn_k2 = load('paired_kshot_a4_k2.json')

    rn = {}
    for K in (1, 5, 10):
        rn[K] = rn_prior[f'K={K}']['resnet_accs']
    rn[2] = rn_k2['K=2']['resnet_accs']

    rows = []
    for K in (1, 2, 5, 10):
        for name, accs in (
            ('LAION-2B', laion[f'K={K}']['accs']),
            ('DFN-CLIP', dfn[f'K={K}']['accs']),
        ):
            row = contrast(accs, rn[K])
            row['contrast'] = f'K={K}: CLIP ({name}) vs ResNet-18'
            rows.append(row)
            print(f"K={K:<3d} {name:9s} CLIP {row['clip_acc_pct']:6.2f} | "
                  f"ResNet {row['resnet_acc_pct']:6.2f} | d={row['delta_pp']:+7.2f} pp "
                  f"CI{row['delta_ci95_bootstrap_pp']} p={row['paired_t_p']:.3g} "
                  f"w/t/l={row['wins_ties_losses']}", flush=True)

    p = [r['paired_t_p'] for r in rows]
    h, q = holm(p), bh(p)
    for i, r in enumerate(rows):
        r['holm_adjusted_p'] = float(h[i])
        r['holm_significant'] = bool(h[i] < ALPHA)
        r['bh_qvalue'] = float(q[i])
        r['bh_fdr_significant'] = bool(q[i] < ALPHA)

    out = {
        'family': ('eight few-shot CLIP-probe vs fine-tuned ResNet-18 seed-paired '
                   'contrasts (K in {1,2,5,10} x {DFN-CLIP, LAION-2B})'),
        'alpha': ALPHA, 'n_tests': len(rows),
        'n_holm_surviving': int(sum(r['holm_significant'] for r in rows)),
        'n_bh_surviving': int(sum(r['bh_fdr_significant'] for r in rows)),
        'bootstrap_resamples': B, 'bootstrap_seed': BOOT_SEED,
        'rows': rows,
        'crossover': {
            'LAION-2B': ('CLIP leads at K=1,2 and trails at K=5,10, so the '
                         'crossover is bracketed to the open interval (2,5).'),
            'DFN-CLIP': ('CLIP trails at every measured K, including K=2; no '
                         'crossover is bracketed because none is crossed.'),
        },
        'provenance': {
            'clip_dfn': 'data/clip_fewshot.json (single caliber, EVAL support, fresh features)',
            'clip_laion': ('data/laion_fewshot_canonical.json (frozen canonical '
                           'store; supersedes two_model_e4.json, whose store gives '
                           '75.36% at full data vs the canonical 76.09%)'),
            'resnet_K_1_5_10': 'data/paired_kshot_a4.json',
            'resnet_K_2': 'data/paired_kshot_a4_k2.json',
        },
    }
    json.dump(out, open(f'{REPO}/data/multiplicity_k2.json', 'w'), indent=2)
    print(f"\nn_tests={out['n_tests']} holm_surviving={out['n_holm_surviving']} "
          f"bh_surviving={out['n_bh_surviving']}")
    print(f'Saved {REPO}/data/multiplicity_k2.json')


if __name__ == '__main__':
    main()
