"""타깃별 학습 결과 모으기 (계획_고정_2026-10-18.md)
  python collect.py val   → 2단계: 검증 쌍 Δ 오차로 타깃별 vs 한 모델 비교 (시드 0–2 평균). 평가 쌍은 보지 않는다
  python collect.py eval  → 3단계: 평가 쌍 표와 검정 (시드 5개 평균). 2단계 결정 뒤에만 실행
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

CAP = '/home/bising0922/Capstone'; os.chdir(CAP)
T = sorted(os.path.basename(f)[:-4] for f in glob.glob('data/moleculeace/CHEMBL*.csv'))
KEY = ['target', 'i', 'j']
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
one = lambda pat: (lambda fs: (fs[0] if len(fs) == 1 else sys.exit(f'파일이 {len(fs)}개: {pat}')))(sorted(glob.glob(pat)))

def splits():
    """분자 split (누수 분자 빼기 포함) — svr_delta.py와 같은 규칙"""
    raw = {}
    for t in T:
        d = pd.read_csv(f'data/moleculeace/{t}.csv').rename(columns={'y [pEC50/pKi]': 'p'})[['smiles', 'p', 'split']]
        tr = d.index[d.split == 'train']; d.loc[np.random.default_rng(0).choice(tr, int(len(tr) * 0.2), replace=False), 'split'] = 'val'
        raw[t] = d
    test_any = {t: set(d.smiles[d.split == 'test']) for t, d in raw.items()}
    for t, d in raw.items():
        others = set().union(*[test_any[u] for u in raw if u != t])
        d.loc[[k for k in d.index if d.split[k] != 'test' and d.smiles[k] in others], 'split'] = 'drop'
    return raw

def pairs(raw, which):
    KEEP = {'val_easy': 'val', 'test_easy': 'test'}
    out = []
    for t in T:
        p = pd.read_pickle(f'data/pairs/{t}_v2.pkl'); p = p[p.split.isin(KEEP)].assign(split=lambda x: x.split.map(KEEP), target=t)
        sp = raw[t].split
        p = p[[sp[i] != 'drop' and sp[j] != 'drop' for i, j in zip(p.i, p.j)]]
        p = p[p.split == which]
        if which == 'test': p = p[p.reason == 'pass']
        out.append(p[KEY + ['delta']])
    return pd.concat(out, ignore_index=True)

def pertarget(P, seeds):
    """타깃별 학습 예측: 열 = 모델_s{시드}, 행 = P의 쌍"""
    cols = {}
    for t in T:
        d = f'runs/pertarget/{t}'; m = P.target == t; q = P[m]
        for name, pat, pre in [('gine_pair', f'{d}/preds_t1_*leakdrop_[0-9]*.pkl', 'pred_gnn_a0_s'),
                               ('molebert_pair', f'{d}/preds_pt_t1_*leakdrop_molebert_[0-9]*.pkl', 'pred_pt_a0_s')]:
            c = pd.read_pickle(one(pat)); mm = q[KEY].merge(c, on=KEY, how='left'); assert len(mm) == len(q)
            for s in seeds: cols.setdefault(f'{name}_s{s}', np.full(len(P), np.nan))[m.values] = mm[f'{pre}{s}'].values
        for name, pat in [('gine_mol', f'{d}/preds_mol_t1_*head1024_[0-9]*.pkl'), ('molebert_mol', f'{d}/preds_molpt_t1_*head1024_molebert_[0-9]*.pkl')]:
            c = pd.read_pickle(one(pat))
            for s in seeds:
                pm = c[f's{s}'].values; cols.setdefault(f'{name}_s{s}', np.full(len(P), np.nan))[m.values] = pm[q.j.values] - pm[q.i.values]
    out = P.assign(**cols); assert not out.isna().any().any(), '빠진 예측이 있다'
    return out

def shared(P, seeds=range(3)):
    """30개 타깃 한 모델 방식의 예측 (검증 비교는 시드 0–2, 최종 평가는 0–4)"""
    out = P.copy()
    for kind, pre, name in [('gine', 'pred_gnn_a0_s', 'gine_pair'), ('mb', 'pred_pt_a0_s', 'molebert_pair')]:
        for s in seeds:
            c = pd.read_pickle(f'runs/tmp/head/pred_head_{kind}_s{s}.pkl'); mm = P[KEY].merge(c[KEY + [f'{pre}{s}']], on=KEY, how='left')
            out[f'{name}_s{s}'] = mm[f'{pre}{s}'].values
    for name, pat in [('gine_mol', 'runs/preds_mol_*_leakdrop_head1024_[0-9]*.pkl'), ('molebert_mol', 'runs/preds_molpt_*_leakdrop_head1024_molebert_[0-9]*.pkl')]:
        c = pd.read_pickle(one(pat))
        for s in seeds:
            pm = {t: g[f's{s}'].values for t, g in c.groupby('target')}
            out[f'{name}_s{s}'] = [pm[t][j] - pm[t][i] for t, i, j in zip(P.target, P.i, P.j)]
    assert not out.isna().any().any()
    return out

MODELS = ['gine_mol', 'gine_pair', 'molebert_mol', 'molebert_pair']
if sys.argv[1] == 'val':
    raw = splits(); V = pairs(raw, 'val'); y = V.delta.values
    print(f'검증 쌍 {len(V):,}개 (그래프 쌍 모델 조기 종료와 같은 쌍)')
    A, B = pertarget(V, range(3)), shared(V)
    for n in MODELS:
        ea = A[[f'{n}_s{s}' for s in range(3)]].mean(1).values - y; eb = B[[f'{n}_s{s}' for s in range(3)]].mean(1).values - y
        ta = pd.Series(ea ** 2).groupby(V.target).mean() ** 0.5; tb = pd.Series(eb ** 2).groupby(V.target).mean() ** 0.5
        print(f'{n:15s} 검증 Δ RMSE  타깃별 {rm(ea):.4f} | 한 모델 {rm(eb):.4f} | 타깃별이 나은 타깃 {int((ta < tb).sum())}/30 → {"타깃별 유지" if rm(ea) <= rm(eb) else "한 모델로 되돌림"}')
elif sys.argv[1] == 'eval':
    raw = splits(); E = pairs(raw, 'test'); assert len(E) == 38711
    E = shared(E, range(5))   # 2단계 결과: 네 모델 모두 한 모델 방식 (타깃별이 검증에서 더 나쁨)
    for name, f in [('svr', 'pred_svr_mol.pkl'), ('svr_pair', None)]:
        c = pd.read_pickle(f'runs/tmp/svr/{f}') if f else pd.concat([pd.read_pickle(x) for x in sorted(glob.glob('runs/tmp/svr/pred_svr_pair_g?.pkl'))])
        mm = E[KEY].merge(c[KEY + ['pred']], on=KEY, how='left'); assert not mm.pred.isna().any(); E[name] = mm.pred.values
    E.to_pickle('runs/tmp/pertarget/eval_preds.pkl')
    y = E.delta.values; cl = np.abs(y) >= 1
    ens = lambda n: E[n].values if n.startswith('svr') else E[[f'{n}_s{s}' for s in range(5)]].mean(1).values
    rows = []
    for lab, n in [('분자별 ECFP + SVR', 'svr'), ('분자별 GINE', 'gine_mol'), ('분자별 Mole-BERT', 'molebert_mol'),
                   ('쌍 ECFP + SVR', 'svr_pair'), ('쌍 GINE', 'gine_pair'), ('쌍 Mole-BERT', 'molebert_pair')]:
        e = ens(n) - y; tg = pd.Series(e[cl] ** 2).groupby(E.target.values[cl]).mean() ** 0.5
        seeds = '' if n.startswith('svr') else ' / '.join(f'{rm(E[f"{n}_s{s}"].values[cl] - y[cl]):.3f}' for s in range(5))
        rows.append(dict(모델=lab, cliff=rm(e[cl]), non_cliff=rm(e[~cl]), 전체=rm(e), 타깃평균_cliff=tg.mean(), 시드별_cliff=seeds))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # 검정: 평가 분자 단위 군집 부트스트랩 (평가 분자 = 쌍에서 test split인 쪽)
    sp = {t: raw[t].split.values for t in T}
    tm = np.where([sp[t][i] == 'test' for t, i in zip(E.target, E.i)], E.i, E.j)
    c = E[cl].assign(tm=tm[cl]); g = c.groupby(['target', 'tm']).indices; keys = list(g); rng = np.random.default_rng(0)
    idx = np.where(cl)[0]; boot = [idx[np.concatenate([g[keys[k]] for k in rng.integers(0, len(keys), len(keys))])] for _ in range(2000)]
    def compare(a, b):
        pa, pb = ens(a), ens(b)
        ta = pd.Series((pa[cl] - y[cl]) ** 2).groupby(E.target.values[cl]).mean() ** 0.5
        tb = pd.Series((pb[cl] - y[cl]) ** 2).groupby(E.target.values[cl]).mean() ** 0.5
        lo, hi = np.percentile([rm(pa[i] - y[i]) - rm(pb[i] - y[i]) for i in boot], [2.5, 97.5]); p = wilcoxon(ta, tb).pvalue
        sig = '유의' if p < 0.05 and (hi < 0 or lo > 0) else '유의하지 않음'
        return f'{a} − {b}: cliff {rm(pa[cl] - y[cl]) - rm(pb[cl] - y[cl]):+.3f} [{lo:+.3f}, {hi:+.3f}] | {a} 나은 타깃 {int((ta < tb).sum())}/30 | Wilcoxon p = {p:.2g} → {sig}'
    for a, b in [('svr_pair', 'svr'), ('gine_pair', 'gine_mol'), ('molebert_pair', 'molebert_mol'), ('molebert_pair', 'svr_pair'), ('gine_pair', 'svr_pair')]:
        print(compare(a, b))
