# 바뀐 부분의 기여 합이 실제 Δ를 얼마나 따라가는가 (모델별, cliff / non-cliff)
import os, numpy as np, pandas as pd
os.chdir('/home/bising0922/Capstone/runs/tmp/xai')
R, A, S = pd.read_pickle('xai_rows.pkl'), pd.read_pickle('xai_atoms.pkl'), pd.read_pickle('xai_sample.pkl')
M = R.merge(A, on=['k', 'model'])
out = []
for (m, cl), d in M.groupby(['model', 'is_cliff']):
    ch = np.array([aa[list(S.atoms_a[k])].sum() + ab[list(S.atoms_b[k])].sum() for k, aa, ab in zip(d.k, d.attr_a, d.attr_b)])
    y, p = d.delta.values, d.pred.values
    out.append({'model': m, 'cliff': cl, '바뀐 부분 기여 합 vs 실제 Δ 상관': np.corrcoef(ch, y)[0, 1],
                '예측 vs 실제 Δ 상관': np.corrcoef(p, y)[0, 1], 'RMSE': np.sqrt(np.mean((p - y) ** 2))})
pd.set_option('display.width', 200)
open('/home/bising0922/Capstone/runs/_diag.txt', 'w').write(pd.DataFrame(out).set_index(['model', 'cliff']).round(3).to_string())
