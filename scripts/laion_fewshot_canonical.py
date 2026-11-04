"""LAION-2B few-shot probe from the FROZEN canonical feature store.

WHY (2026-10-05)
----------------
scripts/two_model_e4.py is self-consistent -- rerunning it reproduces
data/two_model_e4.json to the digit (40.1 / 48.8 / 58.4 / 65.0) -- but its
feature store is not the canonical one. The discrepancy is visible at full data,
where the comparison is exact and seed-free:

    canonical_results.json  laion2b linear_C1   76.09%
    two_model_e4.json       few_shot_probe full 75.36%

Both are "LAION-2B, L2-normalised frozen features, logistic regression C=1.0,
full 1,295-image train, same 138-image test". They cannot both be right. The
frozen store (data/laion_clip_features_clean.npz, bit-reproducible across
processes and byte-identical to the legacy data/clip_features_laion_clean.npz)
reproduces 76.09% and therefore matches the paper's canonical protocol; the
two_model_e4 path does not.

So the LAION few-shot numbers quoted in the manuscript were coming from the
non-canonical store. This script recomputes them from the frozen canonical one,
using the same support draws (coop_experiment.kshot_subset_idx, verified
identical to two_model_e4.kshot_idx at K in {1,2,5,10} for all 15 seeds) and the
same classifier call.

Effect on the paper: K=1 is unchanged (40.10 -> 40.14); K=5 and K=10 move by
about +1.8 pp and +1.5 pp. No sign changes.

Writes data/laion_fewshot_canonical.json.
"""
import os
import sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = ROOT
sys.path.insert(0, f'{REPO}/scripts')
from coop_experiment import kshot_subset_idx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_SEEDS = 15
SRC = f'{REPO}/data/laion_clip_features_clean.npz'


def main():
    z = np.load(SRC)
    X, y, Xte, yte = z['Xtr'], z['ytr'], z['Xte'], z['yte']

    out = {'source': SRC.split('/')[-1],
           'protocol': ('LAION-2B ViT-L/14, L2-normalised frozen features from the '
                        'canonical pixel path, LogisticRegression(C=1.0, '
                        'max_iter=2000, random_state=seed), support = '
                        'kshot_subset_idx(y, K, seed+500), 138-image test set'),
           'supersedes': ('the few_shot_probe block of data/two_model_e4.json, whose '
                          'feature store gives 75.36% at full data against the '
                          'canonical 76.09%')}

    for K in [1, 2, 5, 10]:
        accs = []
        for s in range(N_SEEDS):
            idx = kshot_subset_idx(y, K, s + 500)
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=s)
            clf.fit(X[idx], y[idx])
            accs.append(float((clf.predict(Xte) == yte).mean()))
        a = np.array(accs)
        out[f'K={K}'] = {'mean': float(a.mean()), 'std': float(a.std(ddof=1)),
                         'accs': a.tolist(), 'n_seeds': N_SEEDS}
        print(f'LAION canonical probe K={K:<3d}: {a.mean()*100:5.2f}% +- {a.std(ddof=1)*100:.2f}',
              flush=True)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0).fit(X, y)
    full = float((clf.predict(Xte) == yte).mean())
    out['full'] = {'mean': full, 'std': 0.0, 'accs': [full], 'n_seeds': 1}
    print(f'LAION canonical probe full : {full*100:5.2f}% (deterministic)', flush=True)

    cj = json.load(open(f'{REPO}/data/canonical_results.json'))
    claim = cj['clip_by_checkpoint']['laion2b']['linear_C1']['accuracy'] * 100
    out['full_data_agrees_with_canonical_results'] = abs(full * 100 - claim) <= 0.06
    print(f'canonical_results.json laion2b linear_C1 = {claim:.2f}%  -> '
          f'agrees={out["full_data_agrees_with_canonical_results"]}', flush=True)

    json.dump(out, open(f'{REPO}/data/laion_fewshot_canonical.json', 'w'), indent=2)
    print(f'Saved {REPO}/data/laion_fewshot_canonical.json')


if __name__ == '__main__':
    main()
