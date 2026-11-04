#!/usr/bin/env python3.8
"""
paired_kshot_a4_both.py — regenerate data/paired_kshot_a4_both.json

Provenance for the seed-PAIRED K-shot comparisons (both CLIP variants vs the
fine-tuned ResNet-18 run on the SAME per-seed support sets). This file holds the
headline K=1 LAION-vs-ResNet paired statistics (t_p=0.0003, w_p=0.0009) cited in
the paper. It is produced DETERMINISTICALLY from stored per-seed accuracies, so
every number is reproducible from repo code.

Inputs (per-seed accuracy lists, 15 seeds):
  - LAION-2B CLIP probe : data/two_model_e4.json  -> few_shot_probe[K]['accs']
  - DFN-CLIP probe      : data/paired_kshot_a4.json -> K['clip_accs']
  - fine-tuned ResNet-18 (matched support sets) : data/paired_kshot_a4.json -> K['resnet_accs']

Methods (per K, per CLIP variant):
  - clip / resnet : mean accuracy * 100
  - delta         : (clip_mean - resnet_mean) * 100   [percentage points]
  - t_p           : one-sample paired t-test on the 15 per-seed differences
  - w_p           : Wilcoxon signed-rank, zero_method='pratt' (ties kept in the
                    ranking; the correct choice when per-seed differences contain
                    ties, e.g. the two K=1 LAION seeds that exactly match ResNet).
                    Note: scipy's default zero_method='wilcox' drops ties and
                    gives a slightly larger p (0.0015 at K=1 LAION); both are
                    legitimate, the paper reports the ties-inclusive 'pratt' value.
  - ci            : non-parametric bootstrap (RandomState(10), B=10000) percentile
                    95% CI on the mean per-seed difference, in percentage points.

Reproduce:  python3.8 scripts/paired_kshot_a4_both.py
"""
import os, json
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
KSHOTS = ['K=1', 'K=5', 'K=10']
BOOT_SEED, BOOT_B = 10, 10000


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def bootstrap_ci(deltas_pp, seed=BOOT_SEED, B=BOOT_B):
    rng = np.random.RandomState(seed)
    d = np.asarray(deltas_pp, dtype=float)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(B)])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def paired_stats(clip, resnet):
    clip = np.asarray(clip, dtype=float)
    resnet = np.asarray(resnet, dtype=float)
    d = clip - resnet
    deltas_pp = d * 100.0
    t, t_p = stats.ttest_1samp(d, 0.0)
    w_p = stats.wilcoxon(clip, resnet, zero_method='pratt', correction=False).pvalue
    return {
        'ci': bootstrap_ci(deltas_pp),
        'clip': float(clip.mean() * 100.0),
        'delta': float((clip.mean() - resnet.mean()) * 100.0),
        'resnet': float(resnet.mean() * 100.0),
        't_p': float(t_p),
        'w_p': float(w_p),
    }


def main():
    laion = load('two_model_e4.json')['few_shot_probe']
    a4 = load('paired_kshot_a4.json')
    out = {}
    for K in KSHOTS:
        out[K] = {
            'laion2b_vs_resnet': paired_stats(laion[K]['accs'], a4[K]['resnet_accs']),
            'dfn_vs_resnet': paired_stats(a4[K]['clip_accs'], a4[K]['resnet_accs']),
        }
    out['note'] = ('Seed-paired K-shot: each CLIP variant vs fine-tuned ResNet-18 '
                   'trained on the identical per-seed support set (15 seeds, 138 test). '
                   'w_p uses zero_method=pratt (ties-inclusive). LAION accs from '
                   'two_model_e4.json; DFN + ResNet accs from paired_kshot_a4.json.')
    p = os.path.join(DATA, 'paired_kshot_a4_both.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved', p)
    for K in KSHOTS:
        for var, s in out[K].items():
            print(f"  {K:4s} {var:18s} clip={s['clip']:.1f} resnet={s['resnet']:.1f} "
                  f"delta={s['delta']:+.2f}pp t_p={s['t_p']:.5f} w_p={s['w_p']:.5f} ci={s['ci']}")


if __name__ == '__main__':
    main()
