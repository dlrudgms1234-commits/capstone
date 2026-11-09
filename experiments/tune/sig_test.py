"""성능 차이가 유의한지 본다. `export_preds.py`가 만든 예측값 파일 하나만 있으면 되고, 가중치·GPU·이 저장소의 다른 파일은 필요 없다

    python experiments/tune/sig_test.py preds_test.csv.gz                      # 모델별 성능 표 + 기본 비교
    python experiments/tune/sig_test.py preds_test.csv.gz mb_pair svr_pair     # 두 모델만 비교 (여러 쌍을 이어서 써도 된다)

성능 = test 쌍마다 '아는 분자(A) 실측 + 예측 Δ̂ → 예측 B' 를 만들어 B 실측과 비교한 RMSE.
    Δ̂ 의 RMSE와 수식상 같으므로 예측 Δ̂ 만으로 계산한다
표의 앞 세 칸 = 시드 5개 예측을 평균한 앙상블, 뒤 = 시드별 단일 모델의 평균 ± 표준편차 (SVR은 시드가 없다)

검정 두 가지를 함께 본다 (둘 다 만족할 때만 '유의')
  · 타깃 단위 Wilcoxon: 타깃 30개의 RMSE를 짝지어 비교 (p < 0.05)
  · 군집 부트스트랩 95% 신뢰구간: 같은 평가 분자가 들어간 쌍을 한 덩어리로 묶어 2,000번 다시 뽑는다 (구간이 0을 넘지 않아야 함).
    한 분자가 여러 쌍에 들어가므로 쌍을 따로따로 세면 신뢰구간이 실제보다 좁아진다
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

D = pd.read_csv(sys.argv[1])
y, cl, tg = D.delta.values, D.is_cliff.values.astype(bool), D.target.values
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
MODELS = dict.fromkeys(c.rsplit('_s', 1)[0] for c in D.columns if c.rsplit('_s', 1)[-1].isdigit())
seeds = lambda n: [D[c].values for c in D.columns if c.rsplit('_s', 1)[0] == n]
ens = lambda n: np.mean(seeds(n), 0)
PARTS = [('cliff', cl), ('non-cliff', ~cl), ('전체', np.ones(len(D), bool))]

print(f'{sys.argv[1]}: 쌍 {len(D):,}개 (cliff {int(cl.sum()):,}, 타깃 {D.target.nunique()}개, 평가 분자 {D.eval_mol.nunique():,}개)\n')
print('| 모델 | cliff | non-cliff | 전체 | 시드별 cliff | 시드별 non-cliff | 시드별 전체 |')
print('|---|---|---|---|---|---|---|')
for n in MODELS:
    ps = seeds(n); e = ens(n) - y
    per = ['–'] * 3 if len(ps) == 1 else [(lambda v: f'{np.mean(v):.3f} ± {np.std(v):.3f}')([rm(p[m] - y[m]) for p in ps]) for _, m in PARTS]
    print(f'| {n} | ' + ' | '.join([f'{rm(e[m]):.3f}' for _, m in PARTS] + per) + ' |')

# 군집 부트스트랩 다시 뽑기 묶음 (구간별로 한 번만 만들어 모든 비교에 같은 것을 쓴다)
rng = np.random.default_rng(0)
def boots(mask):
    idx = np.where(mask)[0]
    g = pd.DataFrame({'t': tg[idx], 'm': D.eval_mol.values[idx]}).groupby(['t', 'm']).indices; keys = list(g)
    return [idx[np.concatenate([g[keys[k]] for k in rng.integers(0, len(keys), len(keys))])] for _ in range(2000)]
B = [(lab, m, boots(m)) for lab, m in PARTS]

def compare(a, b):
    pa, pb = ens(a), ens(b)
    print(f'\n{a} − {b}  (음수 = {a}가 낫다)')
    for lab, m, bs in B:
        ta = pd.Series((pa[m] - y[m]) ** 2).groupby(tg[m]).mean() ** 0.5
        tb = pd.Series((pb[m] - y[m]) ** 2).groupby(tg[m]).mean() ** 0.5
        lo, hi = np.percentile([rm(pa[i] - y[i]) - rm(pb[i] - y[i]) for i in bs], [2.5, 97.5])
        p = wilcoxon(ta, tb).pvalue
        sig = '유의' if p < 0.05 and (hi < 0 or lo > 0) else '유의하지 않음'
        print(f'  {lab:9s} {rm(pa[m] - y[m]) - rm(pb[m] - y[m]):+.3f} [{lo:+.3f}, {hi:+.3f}] | '
              f'{a}가 나은 타깃 {int((ta < tb).sum())}/{len(ta)} | Wilcoxon p = {p:.2g} → {sig}')

DEFAULT = [('mb_pair', 'svr_pair'), ('gine_pair', 'svr_pair'), ('svr_pair', 'svr_mol'),
           ('mb_pair', 'mb_mol'), ('gine_pair', 'gine_mol'), ('mb_pair', 'gine_pair'),
           ('mb_pair_noedit', 'mb_pair'), ('mb_pair_notgt', 'mb_pair')]
args = sys.argv[2:]
for a, b in ([(args[i], args[i + 1]) for i in range(0, len(args), 2)] if args else DEFAULT):
    if a in MODELS and b in MODELS:
        compare(a, b)
    else:
        print(f'\n(건너뜀) {a} 또는 {b} 열이 없다. 있는 모델: {", ".join(MODELS)}')
