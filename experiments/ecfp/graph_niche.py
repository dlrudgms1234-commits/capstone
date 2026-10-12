# 그래프가 ECFP보다 나은 곳 찾기 (설정 2: 누수 없음·우리 방식, 재학습 없음)
# ① 쌍 종류별(한 군데만 / 여러 군데 / 큰 변화 / MCS 시간 초과 / 입체만 다름) ② cliff 가려내기 AUROC ③ 타깃 크기별
import sys, os, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.argv = ['x', '2']
src = open('/home/bising0922/Capstone/runs/tmp/ecfp/report3.py').read()
exec(src[:src.index('tables = []')])
setting = 2; raw = splits(setting); A = all_pairs(setting, raw)
te = (A.split == 'test').values
E = A[te].reset_index(drop=True); y = E.delta.values; cl = np.abs(y) >= 1
P = {'ECFP + MLP 쌍': np.mean([E[['target', 'i', 'j']].merge(pd.read_pickle(f'runs/tmp/ecfp/pred_shared_pair_drop_leak_s{s}.pkl')[['target', 'i', 'j', 'pred']], on=['target', 'i', 'j'], how='left').pred.values for s in range(3)], 0)}
for name, pre, pat in [('GINE 쌍', 'pred_gnn', f'runs/preds_{TAGB}_v2_leakdrop_[0-9]*.pkl'), ('Mole-BERT 쌍', 'pred_pt', f'runs/preds_pt_{TAGB}_v2_leakdrop_molebert_*.pkl')]:
    c = pd.read_pickle(glob.glob(pat)[0]); P[name] = np.mean([c[f'{pre}_a0_s{s}'].values[te] for s in range(3)], 0)
sv = svr_mols(raw); P['SVR'] = np.array([sv[t][j] - sv[t][i] for t, i, j in zip(E.target, E.i, E.j)])
KIND = {'pass': '한 군데만 다름', 'multi_site': '여러 군데 바뀜', 'too_big': '바뀐 원자 10개 초과', 'timeout': 'MCS 시간 초과', 'identical_2d': '입체만 다름'}
rows = []
for r, kname in KIND.items():
    m = (E.reason == r).values
    for name, p in P.items():
        mc, mn = m & cl, m & ~cl
        auc = roc_auc_score(cl[m], np.abs(p[m])) if 0 < cl[m].sum() < m.sum() else np.nan
        rows.append({'쌍 종류': kname, '쌍 수': int(m.sum()), 'cliff 수': int(mc.sum()), '모델': name,
                     'cliff RMSE': rm(p[mc] - y[mc]) if mc.any() else np.nan, 'non-cliff RMSE': rm(p[mn] - y[mn]) if mn.any() else np.nan,
                     'cliff 가려내기 AUROC': auc})
R = pd.DataFrame(rows)
pd.set_option('display.width', 250)
lines = [R.round(3).to_string(index=False), '']
# 타깃 크기별 (한 군데만 다른 쌍 cliff): 학습 분자 수 하위·중간·상위 10개 타깃
ntr = {t: int((d.split == 'train').sum()) for t, d in raw.items()}
order = sorted(T, key=ntr.get); grp = {t: ['작은 타깃 10개', '중간 타깃 10개', '큰 타깃 10개'][k // 10] for k, t in enumerate(order)}
g = E.target.map(grp).values; ms = (E.reason == 'pass').values & cl
for gname in ['작은 타깃 10개', '중간 타깃 10개', '큰 타깃 10개']:
    mm = ms & (g == gname)
    lines.append(f'{gname} (학습 분자 {min(ntr[t] for t in T if grp[t] == gname)}–{max(ntr[t] for t in T if grp[t] == gname)}개, cliff 쌍 {mm.sum()}): ' +
                 ', '.join(f'{n} {rm(p[mm] - y[mm]):.3f}' for n, p in P.items()))
open('/home/bising0922/Capstone/runs/_diag.txt', 'w').write('\n'.join(lines))
