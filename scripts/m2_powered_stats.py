"""
M2 powered statistics (lever A, from EXISTING on-disk per-seed arrays; no new training).

The paper already runs the full 15-seed protocol on three crops (tobacco TPDD,
tomato + corn PlantVillage). It currently frames the tomato/corn check as
"qualitative / near-ceiling". The few-shot regime split (CLIP leads at K=1,
ResNet-18 leads at K>=5) is, however, on powered test sets (138/1635/346) with
SEED-PAIRED runs (identical kshot_subset_idx(y,K,seed+500) support for CLIP and
ResNet, so seed i is paired). This computes the paired significance of that split.

Honest scope: the full-DATA comparison is near-ceiling (both ~98-99%) and is NOT
claimed as significant here; only the few-shot regime split is tested.
"""
import os, json
import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, 'data')
OUT = os.path.join(D, 'm2_powered_stats.json')


def paired(clip_accs, res_accs):
    c = np.asarray(clip_accs, float); r = np.asarray(res_accs, float)
    assert len(c) == len(r) and len(c) >= 3, f"paired arrays mismatch {len(c)} vs {len(r)}"
    gap = c - r                       # CLIP minus ResNet, per seed
    n = len(gap)
    m = float(gap.mean()); sd = float(gap.std(ddof=1)); se = sd / np.sqrt(n)
    t, p = stats.ttest_1samp(gap, 0.0)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    ci = [m - tcrit * se, m + tcrit * se]
    return {'clip_mean': round(float(c.mean()), 4), 'resnet_mean': round(float(r.mean()), 4),
            'gap_pp': round(m * 100, 2), 'ci95_pp': [round(ci[0] * 100, 2), round(ci[1] * 100, 2)],
            't': round(float(t), 3), 'df': n - 1, 'p_two_sided': round(float(p), 5),
            'sig_05': bool(p < 0.05), 'n_seeds': int(n)}


def main():
    R = {'protocol': 'seed-paired (same kshot_subset_idx(y,K,seed+500) support) one-sample t on per-seed gap (CLIP - ResNet); 3 crops x K=1/5/10; full-data excluded (near-ceiling).'}
    crops = {}

    # ---- TPDD (tobacco) ----
    clip = json.load(open(os.path.join(D, 'clip_fewshot.json')))
    res = {K: json.load(open(os.path.join(D, f'resnet_k{K}_n15.json'))) for K in (1, 5, 10)}
    crops['tobacco_TPDD'] = {f'K={K}': paired(clip[f'K={K}']['accs'], res[K]['accs']) for K in (1, 5, 10)}

    # ---- tomato + corn (PlantVillage) ----
    for tag, f in [('tomato_PlantVillage', 'pv_tomato_results.json'), ('corn_PlantVillage', 'corn_results.json')]:
        d = json.load(open(os.path.join(D, f)))
        fs = d['few_shot']
        crops[tag] = {f'K={K}': paired(fs[f'K={K}']['clip_probe_C1']['per_seed'],
                                       fs[f'K={K}']['resnet_finetuned']['per_seed']) for K in (1, 5, 10)}

    R['crops'] = crops

    # summary: regime-split consistency
    R['regime_split'] = {}
    for crop, ks in crops.items():
        R['regime_split'][crop] = {
            'K1_clip_leads': ks['K=1']['gap_pp'] > 0,
            'K5_resnet_leads': ks['K=5']['gap_pp'] < 0,
            'K10_resnet_leads': ks['K=10']['gap_pp'] < 0,
        }

    print(f"{'crop':24s} {'K':4s} {'CLIP':>7s} {'ResNet':>7s} {'gap_pp':>8s} {'95%CI(pp)':>16s} {'p':>9s} sig")
    for crop, ks in crops.items():
        for K in ('K=1', 'K=5', 'K=10'):
            r = ks[K]
            print(f"{crop:24s} {K:4s} {r['clip_mean']*100:6.1f}% {r['resnet_mean']*100:6.1f}% "
                  f"{r['gap_pp']:+7.2f} [{r['ci95_pp'][0]:+.2f},{r['ci95_pp'][1]:+.2f}] "
                  f"{r['p_two_sided']:9.5f} {'*' if r['sig_05'] else ' '}")
    print("\nRegime split (CLIP leads K=1; ResNet leads K>=5):")
    for crop, s in R['regime_split'].items():
        print(f"  {crop}: K1_clip_leads={s['K1_clip_leads']}  K5_resnet={s['K5_resnet_leads']}  K10_resnet={s['K10_resnet_leads']}")

    json.dump(R, open(OUT, 'w'), indent=2)
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
