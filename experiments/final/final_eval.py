"""최종 설계 결과 (계획_고정_2026-10-18.md)
  python final_eval.py val   → 검증 표: 한 군데만 다른 검증 쌍 / 검증 쌍 전체, SVR 분자별·쌍과 같은 쌍 (평가 쌍은 보지 않음)
  python final_eval.py eval  → 평가 한 번: 6개 모델 표, 시드별 값, 타깃 평균, Wilcoxon + 평가 분자 단위 군집 부트스트랩
그래프 모델: GINE 300·Mole-BERT, 조각 인코더 없음, 합 읽기, 헤드 2048 → 1024 → 1, 시드 0–4 예측 평균"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

CAP = '/home/bising0922/Capstone'; os.chdir(CAP)
KEY = ['target', 'i', 'j']; SEEDS = range(5)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
one = lambda pat: (lambda fs: fs[0] if len(fs) == 1 else sys.exit(f'파일이 {len(fs)}개: {pat}'))(sorted(glob.glob(pat)))

A = pd.read_pickle('runs/tmp/final/pred_final_mb_s0.pkl')[KEY + ['split', 'reason', 'delta']]   # 쌍 목록 (모든 쌍, 모델마다 같은 순서)
for enc, pre in [('mb', 'pred_pt_a0_s'), ('gine', 'pred_gnn_a0_s')]:
    name = {'mb': 'molebert_pair', 'gine': 'gine_pair'}[enc]
    for s in SEEDS:
        c = pd.read_pickle(f'runs/tmp/final/pred_final_{enc}_s{s}.pkl'); assert (c[KEY].values == A[KEY].values).all()
        A[f'{name}_s{s}'] = c[f'{pre}{s}'].values
for name, pat in [('gine_mol', 'runs/preds_mol_t30_h300_*leakdrop_head2048_sum_[0-9]*.pkl'),
                  ('molebert_mol', 'runs/preds_molpt_t30_h300_*leakdrop_head2048_sum_molebert_[0-9]*.pkl')]:
    c = pd.read_pickle(one(pat)); pm = {t: g for t, g in c.groupby('target')}
    for s in SEEDS:
        v = {t: g[f's{s}'].values for t, g in pm.items()}
        A[f'{name}_s{s}'] = [v[t][j] - v[t][i] for t, i, j in zip(A.target, A.i, A.j)]

def add_svr(D, which):
    for name, f in [('svr', f'runs/tmp/svr/{"predval_" if which == "val" else "pred_"}svr_mol.pkl'),
                    ('svr_pair', f'runs/tmp/svr/{"predval_" if which == "val" else "pred_"}svr_pair_g?.pkl')]:
        c = pd.concat([pd.read_pickle(x) for x in sorted(glob.glob(f))])
        m = D[KEY].merge(c[KEY + ['pred']], on=KEY, how='left'); assert not m.pred.isna().any(), name; D[name] = m.pred.values
    return D

ens = lambda D, n: D[n].values if n.startswith('svr') else D[[f'{n}_s{s}' for s in SEEDS]].mean(1).values
ORDER = [('분자별 ECFP + SVR', 'svr'), ('분자별 GINE', 'gine_mol'), ('분자별 Mole-BERT', 'molebert_mol'),
         ('쌍 ECFP + SVR', 'svr_pair'), ('쌍 GINE', 'gine_pair'), ('쌍 Mole-BERT', 'molebert_pair')]

if sys.argv[1] == 'val':
    V = add_svr(A[A.split.str.startswith('val')].reset_index(drop=True), 'val'); y = V.delta.values; cl = np.abs(y) >= 1; ps = (V.reason == 'pass').values
    rows = []
    for lab, n in ORDER:
        e = ens(V, n) - y
        rows.append(dict(모델=lab, 한곳_전체=rm(e[ps]), 한곳_cliff=rm(e[ps & cl]), 한곳_non_cliff=rm(e[ps & ~cl]), 전체쌍_전체=rm(e), 전체쌍_cliff=rm(e[cl]), 전체쌍_non_cliff=rm(e[~cl])))
    print(f'검증 쌍 {len(V):,}개 (한 군데만 다른 쌍 {ps.sum():,}개). 그래프 모델은 시드 5개 평균')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
elif sys.argv[1] == 'eval':
    E = add_svr(A[(A.split == 'test') & (A.reason == 'pass')].reset_index(drop=True), 'eval'); assert len(E) == 38711
    E.to_pickle('runs/tmp/final/eval_preds.pkl')
    y = E.delta.values; cl = np.abs(y) >= 1; tg = E.target.values
    rows = []
    for lab, n in ORDER:
        e = ens(E, n) - y
        seeds = '' if n.startswith('svr') else ' / '.join(f'{rm(E[f"{n}_s{s}"].values[cl] - y[cl]):.3f}' for s in SEEDS)
        rows.append(dict(모델=lab, cliff=rm(e[cl]), non_cliff=rm(e[~cl]), 전체=rm(e),
                         타깃평균_cliff=float((pd.Series(e[cl] ** 2).groupby(tg[cl]).mean() ** 0.5).mean()), 시드별_cliff=seeds))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # 평가 분자 단위 군집 부트스트랩 (평가 분자 = 쌍에서 학습 분자가 아닌 쪽)
    raw = {t: pd.read_csv(f'data/moleculeace/{t}.csv').split.values for t in E.target.unique()}
    tm = np.where([raw[t][i] == 'test' for t, i in zip(E.target, E.i)], E.i, E.j)
    rng = np.random.default_rng(0)
    def boots(mask):
        idx = np.where(mask)[0]; g = pd.DataFrame({'t': tg[idx], 'm': tm[idx]}).groupby(['t', 'm']).indices; keys = list(g)
        return [idx[np.concatenate([g[keys[k]] for k in rng.integers(0, len(keys), len(keys))])] for _ in range(2000)]
    B = {'cliff': (cl, boots(cl)), 'non-cliff': (~cl, boots(~cl)), '전체': (np.ones_like(cl), boots(np.ones_like(cl)))}
    def compare(a, b):
        pa, pb = ens(E, a), ens(E, b); out = []
        for part, (m, bs) in B.items():
            ta = pd.Series((pa[m] - y[m]) ** 2).groupby(tg[m]).mean() ** 0.5; tb = pd.Series((pb[m] - y[m]) ** 2).groupby(tg[m]).mean() ** 0.5
            lo, hi = np.percentile([rm(pa[i] - y[i]) - rm(pb[i] - y[i]) for i in bs], [2.5, 97.5]); p = wilcoxon(ta, tb).pvalue
            sig = '유의' if p < 0.05 and (hi < 0 or lo > 0) else '유의하지 않음'
            out.append(f'  {part:9s} {rm(pa[m] - y[m]) - rm(pb[m] - y[m]):+.3f} [{lo:+.3f}, {hi:+.3f}] | {a} 나은 타깃 {int((ta < tb).sum())}/30 | p = {p:.2g} → {sig}')
        return f'{a} − {b}\n' + '\n'.join(out)
    for a, b in [('svr_pair', 'svr'), ('gine_pair', 'gine_mol'), ('molebert_pair', 'molebert_mol'), ('molebert_pair', 'svr_pair'), ('gine_pair', 'svr_pair')]:
        print(compare(a, b))
