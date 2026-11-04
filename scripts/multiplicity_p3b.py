"""Expanded multiplicity family for the P3b restructure (executed-as-ten).

Pre-registration section 7 pre-commits that the family of 8 few-shot contrasts
(K in {1,2,5,10} x {DFN-CLIP, LAION-2B}) is expanded by the four P3b cells
(K in {5,10} x {DFN-CLIP, LAION-2B}) and that Holm-Bonferroni and
Benjamini-Hochberg are recomputed over the expanded family, reporting both the
old-8 and the expanded results. This script computes the expanded family.

Family as executed (10 contrasts, all seed-paired):
  - 8 frozen-probe contrasts: CLIP frozen linear probe vs fine-tuned ResNet-18,
    K in {1,2,5,10} x {LAION-2B, DFN-CLIP}.
      * K=1, K=2: ResNet comparator at the legacy single-config budget
        (byte-identical inputs to data/multiplicity_k2.json, which is validated
        below).
      * K=5, K=10: ResNet comparator at the matched budget (amendment A4
        re-base; data/p3b_resnet_kshot_matched{,_k10}.json).
  - 2 fine-tuned-CLIP contrasts: LAION-2B CLIP fine-tuned vs matched-budget
    ResNet-18 at K in {5,10} (the P3b decisive cells).
The two DFN-CLIP fine-tune cells of the pre-registered grid were not run
(Branch A was decided on the LAION-2B decisive cell), so the executed family is
10, not 12; this is documented in the output.

The old family of 8 remains in data/multiplicity_k2.json; both are reported.

Inputs and their provenances
----------------------------
  DFN CLIP per-seed     data/clip_fewshot.json              single caliber (fresh features)
  LAION CLIP per-seed   data/laion_fewshot_canonical.json   frozen canonical store
  ResNet K=1            data/paired_kshot_a4.json           legacy paired run
  ResNet K=2            data/paired_kshot_a4_k2.json        legacy paired run
  ResNet K=5 matched    data/p3b_resnet_kshot_matched.json  P3b stage-2 finals
  ResNet K=10 matched   data/p3b_resnet_kshot_matched_k10.json
  CLIP FT K=5           data/p3b_clip_finetune_kshot.json   P3b stage-2 finals
  CLIP FT K=10          data/p3b_clip_finetune_kshot_k10.json

All sources draw the same support indices per seed (seed+500, stratified,
K per class). Verified because data/tpdd/*/labels.json is class-sorted, so the
dict-insertion order used by the KShotSubset classes equals the sorted() order
used by coop_experiment.kshot_subset_idx and two_model_e4.kshot_idx.

Reports per-contrast paired t, Wilcoxon (Pratt, ties-inclusive), bootstrap 95% CI
on the mean difference, and seed-level win/tie/loss, plus Holm-Bonferroni FWER
and Benjamini-Hochberg FDR across the executed family of ten.

Validation (hard assertions, aborts on mismatch):
  - K=1/K=2 probe rows reproduce data/multiplicity_k2.json exactly.
  - K=5/K=10 FT rows reproduce data/p3b_branch_decision{,_k10}.json paired t/p
    and win/tie/loss.
  - K=5/K=10 probe rows use the matched-budget ResNet means (64.93/74.64).
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
    rn_k5m = load('p3b_resnet_kshot_matched.json')
    rn_k10m = load('p3b_resnet_kshot_matched_k10.json')
    ft_k5 = load('p3b_clip_finetune_kshot.json')
    ft_k10 = load('p3b_clip_finetune_kshot_k10.json')

    resnet = {
        1: ('legacy', rn_prior['K=1']['resnet_accs']),
        2: ('legacy', rn_k2['K=2']['resnet_accs']),
        5: ('matched', rn_k5m['stage2_final']['accs']),
        10: ('matched', rn_k10m['stage2_final']['accs']),
    }

    rows = []
    for K in (1, 2, 5, 10):
        for name, accs in (
            ('LAION-2B', laion[f'K={K}']['accs']),
            ('DFN-CLIP', dfn[f'K={K}']['accs']),
        ):
            budget, rn_accs = resnet[K]
            row = contrast(accs, rn_accs)
            row['contrast'] = f'K={K}: CLIP ({name}) vs ResNet-18'
            row['clip_kind'] = 'frozen_probe'
            row['resnet_budget'] = budget
            rows.append(row)
            print(f"K={K:<3d} {name:9s} probe   CLIP {row['clip_acc_pct']:6.2f} | "
                  f"ResNet {row['resnet_acc_pct']:6.2f} | d={row['delta_pp']:+7.2f} pp "
                  f"CI{row['delta_ci95_bootstrap_pp']} p={row['paired_t_p']:.3g} "
                  f"w/t/l={row['wins_ties_losses']}", flush=True)

    for K, ft in ((5, ft_k5), (10, ft_k10)):
        budget, rn_accs = resnet[K]
        assert budget == 'matched'
        row = contrast(ft['stage2_final']['accs'], rn_accs)
        row['contrast'] = f'K={K}: CLIP fine-tuned (LAION-2B) vs ResNet-18 (matched budget)'
        row['clip_kind'] = 'finetuned'
        row['resnet_budget'] = budget
        rows.append(row)
        print(f"K={K:<3d} {'LAION-2B':9s} FT-CLIP CLIP {row['clip_acc_pct']:6.2f} | "
              f"ResNet {row['resnet_acc_pct']:6.2f} | d={row['delta_pp']:+7.2f} pp "
              f"CI{row['delta_ci95_bootstrap_pp']} p={row['paired_t_p']:.3g} "
              f"w/t/l={row['wins_ties_losses']}", flush=True)

    # ---- validation -------------------------------------------------------
    old = load('multiplicity_k2.json')
    old_by = {r['contrast']: r for r in old['rows']}
    for r in rows:
        if r['clip_kind'] == 'frozen_probe' and r['resnet_budget'] == 'legacy':
            o = old_by[r['contrast']]
            assert abs(r['paired_t'] - o['paired_t']) < 1e-9, r['contrast']
            assert abs(r['delta_pp'] - o['delta_pp']) < 1e-9, r['contrast']
            assert np.allclose(r['delta_ci95_bootstrap_pp'],
                               o['delta_ci95_bootstrap_pp'], atol=1e-9), r['contrast']
            assert np.isclose(r['paired_t_p'], o['paired_t_p'], rtol=1e-9), r['contrast']
            assert np.isclose(r['wilcoxon_p'], o['wilcoxon_p'], rtol=1e-9), r['contrast']
    print('validation: K=1/K=2 probe rows reproduce multiplicity_k2.json', flush=True)

    bd = {5: load('p3b_branch_decision.json'), 10: load('p3b_branch_decision_k10.json')}
    for r in rows:
        if r['clip_kind'] == 'finetuned':
            K = int(r['contrast'].split(':')[0].split('=')[1])
            s = bd[K]['seed_paired']
            assert abs(r['delta_pp'] - s['mean_diff_pp']) < 1e-9, r['contrast']
            assert abs(r['paired_t'] - s['paired_t']) < 1e-3, (r['paired_t'], s['paired_t'])
            assert np.isclose(r['paired_t_p'], s['paired_t_p'], rtol=1e-6), r['contrast']
            assert np.isclose(r['wilcoxon_p'], s['wilcoxon_p'], rtol=1e-6), r['contrast']
            assert r['wins_ties_losses'] == s['wins_ties_losses'], r['contrast']
            assert np.allclose(r['delta_ci95_bootstrap_pp'],
                               s['bootstrap_ci95_pp'], atol=0.02), r['contrast']
    print('validation: K=5/K=10 FT rows reproduce p3b_branch_decision{,_k10}.json', flush=True)

    for r in rows:
        if r['clip_kind'] == 'frozen_probe' and r['resnet_budget'] == 'matched':
            K = int(r['contrast'].split(':')[0].split('=')[1])
            exp = round(bd[K]['resnet_finetuned_matched_budget']['mean'] * 100, 2)
            assert abs(r['resnet_acc_pct'] - exp) < 1e-9, r['contrast']
    print('validation: K=5/K=10 probe rows use matched-budget ResNet means', flush=True)

    # ---- multiplicity correction over the executed family of ten ----------
    p = [r['paired_t_p'] for r in rows]
    h, q = holm(p), bh(p)
    for i, r in enumerate(rows):
        r['holm_adjusted_p'] = float(h[i])
        r['holm_significant'] = bool(h[i] < ALPHA)
        r['bh_qvalue'] = float(q[i])
        r['bh_fdr_significant'] = bool(q[i] < ALPHA)

    out = {
        'family': ('expanded multiplicity family as executed: ten seed-paired '
                   'contrasts = eight frozen-probe CLIP vs fine-tuned ResNet-18 '
                   '(K in {1,2,5,10} x {DFN-CLIP, LAION-2B}; K=1,2 ResNet '
                   'comparators at the legacy budget, K=5,10 at the matched '
                   'budget per amendment A4) plus two LAION-2B fine-tuned-CLIP '
                   'vs matched-budget ResNet-18 contrasts (K in {5,10})'),
        'family_note': ('Pre-registration section 7 pre-commits an expanded family '
                        'of 12 (four P3b cells: K in {5,10} x {DFN-CLIP, LAION-2B}). '
                        'The two DFN-CLIP fine-tune cells were not run because Branch A '
                        'was decided on the LAION-2B decisive cell alone; the executed '
                        'family is therefore 10. The old family of eight remains in '
                        'data/multiplicity_k2.json and both are reported.'),
        'alpha': ALPHA, 'n_tests': len(rows),
        'n_holm_surviving': int(sum(r['holm_significant'] for r in rows)),
        'n_bh_surviving': int(sum(r['bh_fdr_significant'] for r in rows)),
        'bootstrap_resamples': B, 'bootstrap_seed': BOOT_SEED,
        'rows': rows,
        'crossover': {
            'LAION-2B': ('CLIP leads at K=1,2 and trails at K=5,10 under the '
                         'matched-budget ResNet comparator, so the crossover is '
                         'bracketed to the open interval (2,5).'),
            'DFN-CLIP': ('CLIP trails at every measured K, including K=2; no '
                         'crossover is bracketed because none is crossed.'),
        },
        'provenance': {
            'clip_dfn': 'data/clip_fewshot.json',
            'clip_laion': 'data/laion_fewshot_canonical.json',
            'resnet_K_1': 'data/paired_kshot_a4.json (legacy budget)',
            'resnet_K_2': 'data/paired_kshot_a4_k2.json (legacy budget)',
            'resnet_K_5_matched': 'data/p3b_resnet_kshot_matched.json',
            'resnet_K_10_matched': 'data/p3b_resnet_kshot_matched_k10.json',
            'clip_ft_K_5': 'data/p3b_clip_finetune_kshot.json',
            'clip_ft_K_10': 'data/p3b_clip_finetune_kshot_k10.json',
            'old_family_of_8': 'data/multiplicity_k2.json',
        },
        'validation': {
            'k1_k2_probe_rows_reproduce_k2': True,
            'ft_rows_reproduce_branch_decision': [5, 10],
            'k5_k10_probe_rows_use_matched_resnet': True,
        },
    }
    json.dump(out, open(f'{REPO}/data/multiplicity_p3b.json', 'w'), indent=2)
    print(f"\nn_tests={out['n_tests']} holm_surviving={out['n_holm_surviving']} "
          f"bh_surviving={out['n_bh_surviving']}")
    print(f'Saved {REPO}/data/multiplicity_p3b.json')


if __name__ == '__main__':
    main()
