"""test 성능표 두 개 (전체 성능, ablation). 저장소 최상위에서 실행한다
  python experiments/tune/test_table.py
한 군데만 다른 test 쌍 38,711개. 그래프 모델은 시드 5개 각각의 RMSE 평균 ± 표준편차 (cliff / non-cliff / 전체). 앙상블은 쓰지 않는다
ECFP + SVR은 학습에 무작위성이 없어 값이 하나. GraphCliff는 팀원이 보낸 test 숫자를 그대로 쓴다 (예측 파일은 팀원 서버)
Mole-BERT 인코더 학습률: 사전학습 없음 1e-4, 사전학습 1e-3
"""
import glob
import numpy as np, pandas as pd

KEY = ['target', 'i', 'j']
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
MOLCACHE = 'runs/preds_{kind}_t30_h300_l3_dr0.3_wd0.01_ep1000_leakdrop_head1024_adamw{mb}_*.pkl'
molcache = lambda kind, mb: [f for f in glob.glob(MOLCACHE.format(kind=kind, mb=mb)) if f[len(MOLCACHE.format(kind=kind, mb=mb)) - 5:-4].isdigit()][:1]

base = pd.read_pickle('runs/tmp/final3/pred_final3_gine_s0.pkl')
P = base[(base.split == 'test') & (base.reason == 'pass')][KEY + ['delta']].reset_index(drop=True)
assert len(P) == 38711
y = P.delta.values; M = [np.abs(y) >= 1, np.abs(y) < 1, np.ones(len(y), bool)]

def pair(fs):
    out = []
    for f in fs:
        d = pd.read_pickle(f); col = [c for c in d.columns if c.startswith('pred_')][0]
        out.append(P[KEY].merge(d[KEY + [col]], on=KEY, how='left')[col].values)
    return out

def mol(f):
    c = pd.read_pickle(f); g = {t: x for t, x in c.groupby('target')}
    return [np.array([g[t][s].values[j] - g[t][s].values[i] for t, i, j in zip(P.target, P.i, P.j)]) for s in c.columns if s[0] == 's' and s[1:].isdigit()]

def svr(pat):
    c = pd.concat([pd.read_pickle(f) for f in glob.glob(pat) if '_pass_' not in f and '_valall_' not in f]).drop_duplicates(KEY)
    return [P[KEY].merge(c[KEY + ['pred']], on=KEY, how='left').pred.values]

def stats(ps):
    """cliff / non-cliff / 전체 각각 '평균 ± 표준편차' (모델이 하나면 값 하나)"""
    v = np.array([[rm(p[m] - y[m]) for m in M] for p in ps])
    return [f'{v[0, k]:.3f}' if len(ps) == 1 else f'{v[:, k].mean():.3f} ± {v[:, k].std():.3f}' for k in range(3)]

g = lambda d: sorted(glob.glob(f'{d}/pred_mb_s?.pkl'))
ROWS = [
    ('ECFP + SVR (베이스라인)', stats(svr('runs/tmp/svr/pred_svr_mol.pkl')), stats(svr('runs/tmp/svr/pred_svr_pair_*.pkl'))),
    ('GINE', stats(mol(molcache('mol', '')[0])), stats(pair(sorted(glob.glob('runs/tmp/final3/pred_final3_gine_s?.pkl'))))),
    # 팀원 GraphCliff test (10-23 보고)
    ('GraphCliff', ['0.921 ± 0.049', '0.529 ± 0.011', '0.639 ± 0.022'], ['0.904 ± 0.011', '0.483 ± 0.005', '0.605 ± 0.007']),
    ('Mole-BERT (사전학습 없음)', stats(mol('runs/tune/mb_scratch/pred_mol_mb.pkl')), stats(pair(g('runs/tune/mb_scratch')))),
    ('Mole-BERT (사전학습)', stats(mol('runs/tune/mb_ptlr1e-3/pred_mol_mb.pkl')), stats(pair(g('runs/tune/mb_ptlr1e-3')))),
]
print(f'## 표 1. test 성능 (한 군데만 다른 test 쌍 {len(P):,}개, cliff {int(M[0].sum()):,}개, 시드 5개 평균 ± 표준편차)\n')
print('| 모델 | 분자별 cliff | 분자별 non-cliff | 분자별 전체 | 쌍 cliff | 쌍 non-cliff | 쌍 전체 |')
print('|---|---|---|---|---|---|---|')
for name, m, p in ROWS:
    print(f'| {name} | ' + ' | '.join(m + p) + ' |')

print('\n## 표 2. 입력 정보 ablation (Mole-BERT 사전학습 + 쌍 입력, test, 시드 5개 평균 ± 표준편차)\n')
print('| 입력 | cliff | non-cliff | 전체 |')
print('|---|---|---|---|')
for name, d in [('전체 입력', 'runs/tune/mb_ptlr1e-3'), ('편집 정보 뺌', 'runs/tune/mb_lr1e-3_noedit'),
                ('타깃 정보 뺌', 'runs/tune/mb_lr1e-3_notgt'), ('둘 다 뺌', 'runs/tune/mb_lr1e-3_noedit_tgt')]:
    print(f'| {name} | ' + ' | '.join(stats(pair(g(d)))) + ' |')
