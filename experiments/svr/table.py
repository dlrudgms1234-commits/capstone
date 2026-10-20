"""6개 모델(SVR·GINE·Mole-BERT × 분자별·쌍) 평가 표와 분자별 → 쌍 비교 (타깃별 Wilcoxon, 평가 분자 단위 군집 부트스트랩 95% CI)"""
import glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

CAP = '/home/bising0922/Capstone'
d = pd.read_csv('/home/bising0922/share/eval_pairs_predictions_leakfree.csv')
key = ['target', 'i', 'j']
sp = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f'{CAP}/runs/tmp/svr/pred_svr_pair_g?.pkl'))], ignore_index=True)
sm = pd.read_pickle(f'{CAP}/runs/tmp/svr/pred_svr_mol.pkl')
for name, df in [('svr_pair', sp), ('svr_mol_check', sm)]:
    m = d[key].merge(df[key + ['pred']], on=key, how='left'); assert len(m) == len(d) and not m.pred.isna().any(), name
    d[name] = m.pred.values
assert np.allclose(d.svr_mol_check, d.svr, atol=1e-4), '분자별 SVR이 CSV 값과 다르다'
print('쌍 SVR 타깃별 C:', sp.groupby('target').C.first().value_counts().to_dict())

ens = lambda n: d[n] if n in ('svr', 'svr_pair') else d[[f'{n}_s{s}' for s in range(3)]].mean(1)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
y, cl = d.delta.values, d.is_cliff.values
NAMES = [('ECFP + SVR', 'svr', 'svr_pair'), ('GINE', 'gine_mol', 'gine_pair'), ('Mole-BERT', 'molebert_mol', 'molebert_pair')]
rows = []
for lab, mo, pa in NAMES:
    for kind, n in [('분자별', mo), ('쌍', pa)]:
        e = ens(n).values - y
        seeds = '' if n.startswith('svr') else ' / '.join(f"{rm(d[f'{n}_s{s}'].values[cl] - y[cl]):.3f}" for s in range(3))
        rows.append(dict(입력=kind, 모델=lab, cliff=rm(e[cl]), non_cliff=rm(e[~cl]), 전체=rm(e), 시드별_cliff=seeds))
print(pd.DataFrame(rows).round(3).to_string(index=False))

c = d[cl]
g = c.groupby(['target', 'test_mol']).indices; keys = list(g); rng = np.random.default_rng(0)
boot = [c.index.values[np.concatenate([g[keys[k]] for k in rng.integers(0, len(keys), len(keys))])] for _ in range(2000)]
def compare(a, b):
    pa, pb = ens(a).values, ens(b).values
    ta = c.assign(e=(pa[c.index] - c.delta) ** 2).groupby('target').e.mean() ** 0.5
    tb = c.assign(e=(pb[c.index] - c.delta) ** 2).groupby('target').e.mean() ** 0.5
    diffs = [rm(pa[i] - y[i]) - rm(pb[i] - y[i]) for i in boot]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return f'{a} − {b}: cliff 차이 {rm(pa[cl] - y[cl]) - rm(pb[cl] - y[cl]):+.3f} [{lo:+.3f}, {hi:+.3f}] | {a} 나은 타깃 {int((ta < tb).sum())}/30 | Wilcoxon p = {wilcoxon(ta, tb).pvalue:.2g}'
print('\n'.join(compare(a, b) for a, b in [('svr_pair', 'svr'), ('gine_pair', 'gine_mol'), ('molebert_pair', 'molebert_mol'),
                                            ('molebert_pair', 'svr_pair'), ('gine_pair', 'svr_pair'), ('molebert_pair', 'svr')]))
