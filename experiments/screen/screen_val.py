"""검증 후보 비교: 검증 쌍의 Δ 오차만 본다 (평가 쌍은 보지 않음). 그래프 모델은 시드 0
- 범위 두 가지: 검증 쌍 전체(모델 고르는 기준) / 그중 한 군데만 다른 쌍(평가 쌍과 같은 종류)
- SVR 분자별·쌍도 같은 쌍에서 함께 본다. 시드 흔들림 참고값: 기존 설정의 시드별 검증 오차 범위
사용: python screen_val.py [앙상블 후보 A,B ...]  (예: scr_mb_N,scr_g300_N → 두 예측의 평균)"""
import glob, os, sys
import numpy as np, pandas as pd

D = '/home/bising0922/Capstone/runs/tmp'; KEY = ['target', 'i', 'j']
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
base = pd.read_pickle(f'{D}/head/pred_head_mb_s0.pkl'); V = base[base.split.str.startswith('val')][KEY + ['delta', 'reason']].reset_index(drop=True)
y = V.delta.values; cl = np.abs(y) >= 1; ps = (V.reason == 'pass').values

def load(f):
    d = pd.read_pickle(f); col = 'pred' if 'pred' in d.columns else [c for c in d.columns if c.startswith('pred_')][0]
    m = V[KEY].merge(d[KEY + [col]], on=KEY, how='left'); assert not m[col].isna().any(), f
    return m[col].values
P = {}
P['SVR 분자별'] = load(f'{D}/svr/predval_svr_mol.pkl')
sp = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f'{D}/svr/predval_svr_pair_g?.pkl'))]); sp.to_pickle(f'{D}/svr/predval_svr_pair.pkl')
P['SVR 쌍'] = load(f'{D}/svr/predval_svr_pair.pkl')
for kind, lab in [('mb', 'Mole-BERT 기존'), ('gine', 'GINE 64 기존')]:
    for f in sorted(glob.glob(f'{D}/head/pred_head_{kind}_s?.pkl')):
        P[f'{lab} 시드{f[-5]}'] = load(f)
for f in sorted(glob.glob(f'{D}/screen/pred_scr_*.pkl')):
    P[os.path.basename(f)[9:-4]] = load(f)
for pair in sys.argv[1:]:   # 앙상블 후보
    a, b = pair.split(','); P[f'평균({a}+{b})'] = (P[a.replace('scr_', '')] + P[b.replace('scr_', '')]) / 2

rows = []
for k, p in P.items():
    e = p - y
    rows.append(dict(모델=k, 전체=rm(e), cliff=rm(e[cl]), non_cliff=rm(e[~cl]),
                     한곳_전체=rm(e[ps]), 한곳_cliff=rm(e[ps & cl]), 한곳_non_cliff=rm(e[ps & ~cl])))
print(f'검증 쌍 {len(V):,}개 (한 군데만 다른 쌍 {ps.sum():,}개, 그중 cliff {(ps & cl).sum():,}개)')
print(pd.DataFrame(rows).round(3).to_string(index=False))
