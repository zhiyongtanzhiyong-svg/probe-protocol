"""Checkpoint-complete multiplicity family for P3c (executed-as-twelve).

Supersedes data/multiplicity_p3b.json (executed-as-ten). P3c (P3C_DFN_FEW_SHOT_FT_
PREREGISTRATION.md section 5) re-opens the two DFN-CLIP few-shot fine-tune cells
(K=5, K=10) that P3b deferred, so the matched-budget fine-tune comparison is now
checkpoint-complete: BOTH CLIP checkpoints have fine-tune cells at K=5/10.

Family as executed (12 contrasts, all seed-paired):
  - 8 frozen-probe contrasts: CLIP frozen linear probe vs fine-tuned ResNet-18,
    K in {1,2,5,10} x {LAION-2B, DFN-CLIP}.
      * K=1, K=2: ResNet comparator at the legacy single-config budget.
      * K=5, K=10: ResNet comparator at the matched budget.
  - 2 LAION-2B fine-tuned-CLIP contrasts vs matched-budget ResNet-18 at K in {5,10}
    (the P3b decisive cells).
  - 2 DFN-CLIP fine-tuned-CLIP contrasts vs matched-budget ResNet-18 at K in {5,10}
    (the P3c cells).

Validation (hard assertions, aborts on mismatch):
  - K=1/K=2 probe rows reproduce data/multiplicity_k2.json exactly.
  - K=5/K=10 LAION FT rows reproduce data/p3b_branch_decision{,_k10}.json.
  - K=5/K=10 probe rows use the matched-budget ResNet means (64.93/74.64).
  - K=5/K=10 DFN FT clip means equal the P3c stage-2 store means.
The two old families remain: data/multiplicity_k2.json (8) and
data/multiplicity_p3b.json (10); the 12-family is the executed one.
"""
import os
import json
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
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def bh(pvals):
    n = len(pvals)
    order = np.argsort(pvals)[::-1]
    q = np.empty(n)
    running = 1.0
    for rank, i in enumerate(order):
        running = min(running, n * pvals[i] / (n - rank))
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
    ft_laion_k5 = load('p3b_clip_finetune_kshot.json')
    ft_laion_k10 = load('p3b_clip_finetune_kshot_k10.json')
    ft_dfn_k5 = load('p3c_clip_finetune_kshot_dfn.json')
    ft_dfn_k10 = load('p3c_clip_finetune_kshot_dfn_k10.json')

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
            row.update(contrast=f'K={K}: CLIP ({name}) vs ResNet-18',
                       clip_kind='frozen_probe', resnet_budget=budget)
            rows.append(row)
            print(f"K={K:<3d} {name:9s} probe   CLIP {row['clip_acc_pct']:6.2f} | "
                  f"ResNet {row['resnet_acc_pct']:6.2f} | d={row['delta_pp']:+7.2f} pp "
                  f"CI{row['delta_ci95_bootstrap_pp']} p={row['paired_t_p']:.3g} "
                  f"w/t/l={row['wins_ties_losses']}", flush=True)

    ft_cells = []
    for K, ft, tag in ((5, ft_laion_k5, 'LAION-2B'), (10, ft_laion_k10, 'LAION-2B'),
                       (5, ft_dfn_k5, 'DFN-CLIP'), (10, ft_dfn_k10, 'DFN-CLIP')):
        budget, rn_accs = resnet[K]
        assert budget == 'matched'
        accs = ft['stage2_final']['accs']
        row = contrast(accs, rn_accs)
        row.update(contrast=f'K={K}: CLIP fine-tuned ({tag}) vs ResNet-18 (matched budget)',
                   clip_kind='finetuned', checkpoint=tag, resnet_budget=budget)
        rows.append(row)
        ft_cells.append(row)
        print(f"K={K:<3d} {tag:9s} FT-CLIP CLIP {row['clip_acc_pct']:6.2f} | "
              f"ResNet {row['resnet_acc_pct']:6.2f} | d={row['delta_pp']:+7.2f} pp "
              f"CI{row['delta_ci95_bootstrap_pp']} p={row['paired_t_p']:.3g} "
              f"w/t/l={row['wins_ties_losses']}", flush=True)

    # ---- validation (unchanged P3b checks for the 10 non-DFN-FT rows) -------
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
        if r['clip_kind'] == 'finetuned' and r.get('checkpoint') == 'LAION-2B':
            K = int(r['contrast'].split(':')[0].split('=')[1])
            s = bd[K]['seed_paired']
            assert abs(r['delta_pp'] - s['mean_diff_pp']) < 1e-9, r['contrast']
            assert abs(r['paired_t'] - s['paired_t']) < 1e-3, (r['paired_t'], s['paired_t'])
            assert np.isclose(r['paired_t_p'], s['paired_t_p'], rtol=1e-6), r['contrast']
            assert np.isclose(r['wilcoxon_p'], s['wilcoxon_p'], rtol=1e-6), r['contrast']
            assert r['wins_ties_losses'] == s['wins_ties_losses'], r['contrast']
            assert np.allclose(r['delta_ci95_bootstrap_pp'],
                               s['bootstrap_ci95_pp'], atol=0.02), r['contrast']
    print('validation: K=5/K=10 LAION FT rows reproduce p3b_branch_decision{,_k10}.json', flush=True)

    for r in rows:
        if r['clip_kind'] == 'frozen_probe' and r['resnet_budget'] == 'matched':
            K = int(r['contrast'].split(':')[0].split('=')[1])
            exp = round(bd[K]['resnet_finetuned_matched_budget']['mean'] * 100, 2)
            assert abs(r['resnet_acc_pct'] - exp) < 1e-9, r['contrast']
    print('validation: K=5/K=10 probe rows use matched-budget ResNet means', flush=True)

    for r in rows:
        if r['clip_kind'] == 'finetuned' and r.get('checkpoint') == 'DFN-CLIP':
            K = int(r['contrast'].split(':')[0].split('=')[1])
            src = ft_dfn_k5 if K == 5 else ft_dfn_k10
            exp = round(src['stage2_final']['mean'] * 100, 2)
            assert abs(r['clip_acc_pct'] - exp) < 1e-9, r['contrast']
    print('validation: K=5/K=10 DFN FT clip means equal the P3c stage-2 stores', flush=True)

    # ---- multiplicity correction over the executed family of twelve ---------
    p = [r['paired_t_p'] for r in rows]
    h, q = holm(p), bh(p)
    for i, r in enumerate(rows):
        r['holm_adjusted_p'] = float(h[i])
        r['holm_significant'] = bool(h[i] < ALPHA)
        r['bh_qvalue'] = float(q[i])
        r['bh_fdr_significant'] = bool(q[i] < ALPHA)

    def branch(r):
        lo, hi = r['delta_ci95_bootstrap_pp']
        if lo > 0:
            return 'R (reversal)'
        if hi < 0:
            return 'S (scope)'
        return 'T (tie)'

    for r in rows:
        if r['clip_kind'] == 'finetuned':
            r['p3c_branch'] = branch(r)

    out = {
        'family': ('checkpoint-complete multiplicity family as executed: twelve '
                   'seed-paired contrasts = eight frozen-probe CLIP vs fine-tuned '
                   'ResNet-18 (K in {1,2,5,10} x {DFN-CLIP, LAION-2B}; K=1,2 ResNet '
                   'comparators at the legacy budget, K=5,10 at the matched budget) '
                   'plus four fine-tuned-CLIP vs matched-budget ResNet-18 contrasts '
                   '(K in {5,10} x {LAION-2B, DFN-CLIP})'),
        'family_note': ('P3c section 5 pre-commits the executed family of 12: the '
                        'P3b family of 10 (data/multiplicity_p3b.json) plus the two '
                        'DFN-CLIP few-shot fine-tune cells that P3b deferred. Both '
                        'checkpoints now have fine-tune cells at K=5/10.'),
        'alpha': ALPHA, 'n_tests': len(rows),
        'n_holm_surviving': int(sum(r['holm_significant'] for r in rows)),
        'n_bh_surviving': int(sum(r['bh_fdr_significant'] for r in rows)),
        'bootstrap_resamples': B, 'bootstrap_seed': BOOT_SEED,
        'rows': rows,
        'dfn_ft_branches': {
            'K=5': next((r['p3c_branch'] for r in rows
                         if r['clip_kind'] == 'finetuned' and r.get('checkpoint') == 'DFN-CLIP'
                         and 'K=5:' in r['contrast']), None),
            'K=10': next((r['p3c_branch'] for r in rows
                          if r['clip_kind'] == 'finetuned' and r.get('checkpoint') == 'DFN-CLIP'
                          and 'K=10:' in r['contrast']), None),
        },
        'provenance': {
            'clip_dfn': 'data/clip_fewshot.json',
            'clip_laion': 'data/laion_fewshot_canonical.json',
            'resnet_K_1': 'data/paired_kshot_a4.json (legacy budget)',
            'resnet_K_2': 'data/paired_kshot_a4_k2.json (legacy budget)',
            'resnet_K_5_matched': 'data/p3b_resnet_kshot_matched.json',
            'resnet_K_10_matched': 'data/p3b_resnet_kshot_matched_k10.json',
            'clip_ft_laion_K_5': 'data/p3b_clip_finetune_kshot.json',
            'clip_ft_laion_K_10': 'data/p3b_clip_finetune_kshot_k10.json',
            'clip_ft_dfn_K_5': 'data/p3c_clip_finetune_kshot_dfn.json',
            'clip_ft_dfn_K_10': 'data/p3c_clip_finetune_kshot_dfn_k10.json',
            'old_family_of_8': 'data/multiplicity_k2.json',
            'old_family_of_10': 'data/multiplicity_p3b.json',
        },
    }
    json.dump(out, open(f'{REPO}/data/multiplicity_p3c.json', 'w'), indent=2)
    print(f"\nn_tests={out['n_tests']} holm_surviving={out['n_holm_surviving']} "
          f"bh_surviving={out['n_bh_surviving']}")
    print(f"DFN-CLIP FT branches: {out['dfn_ft_branches']}")
    print(f'Saved {REPO}/data/multiplicity_p3c.json')


if __name__ == '__main__':
    main()
