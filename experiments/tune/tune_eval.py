"""튜닝 결과 비교. 저장소 최상위에서 실행한다
  python experiments/tune/tune_eval.py val runs/tune/<이름> [runs/tune/<이름2> ...]
      → 한 군데만 다른 valid 쌍의 표: 시드 앙상블 + 시드별 평균 ± 표준편차 (cliff / non-cliff / 전체), SVR과 최종 설계 3 기준값을 같이 출력
      → 학습 곡선 그림을 runs/tune/<이름>/curves_*.png 로 항상 저장
  python experiments/tune/tune_eval.py test runs/tune/<이름>
      → test 표와 검정. **설정을 valid로 고른 뒤 한 번만** 실행하고 experiments/계획_고정_2026-10-18.md 에 "n번째 열기"로 기록
SVR 예측이 필요하다 (한 번만): python experiments/svr/svr_delta.py mol ; python experiments/svr/svr_delta.py pair all 1,10,100,1000 all  (CPU 30분 안팎)
"""
import glob, json, os, subprocess, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

KEY = ['target', 'i', 'j']
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
# 최종 설계 3 (AdamW 감쇠 0.01, 시드 5개 앙상블), 한 군데만 다른 valid 쌍
# (표기 순서를 출력과 같게: cliff / non-cliff / 전체)
REF = {'쌍 GINE': (0.870, 0.429, 0.559), '쌍 Mole-BERT': (0.877, 0.448, 0.573), '분자별 GINE': (0.940, 0.454, 0.599),
       '분자별 Mole-BERT': (1.115, 0.528, 0.704), '쌍 SVR': (0.933, 0.437, 0.586), '분자별 SVR': (0.929, 0.450, 0.593)}

def load(d, part):
    """한 튜닝 폴더의 예측 → (쌍 표, {모델: [시드별 Δ̂]}). part: 'val' = 한 군데만 다른 valid 쌍, 'test' = test 쌍"""
    fs = sorted(glob.glob(f'{d}/pred_*_s?.pkl')); assert fs, f'예측 파일이 없다: {d}'
    base = pd.read_pickle(fs[0])
    m = (base.split.str.startswith('val') if part == 'val' else base.split == 'test') & (base.reason == 'pass')
    P = base[m][KEY + ['delta']].reset_index(drop=True)
    out = {}
    for enc, col, lab in [('gine', 'pred_gnn_a0_s', '쌍 GINE'), ('mb', 'pred_pt_a0_s', '쌍 Mole-BERT')]:
        ss = sorted(glob.glob(f'{d}/pred_{enc}_s?.pkl'))
        if ss:
            out[lab] = [P[KEY].merge(pd.read_pickle(f)[KEY + [f'{col}{f[-5]}']], on=KEY, how='left')[f'{col}{f[-5]}'].values for f in ss]
    for f, lab in [(f'{d}/pred_mol_gine.pkl', '분자별 GINE'), (f'{d}/pred_mol_mb.pkl', '분자별 Mole-BERT')]:
        if os.path.exists(f):
            c = pd.read_pickle(f); g = {t: x for t, x in c.groupby('target')}
            out[lab] = [np.array([g[t][s].values[j] - g[t][s].values[i] for t, i, j in zip(P.target, P.i, P.j)]) for s in c.columns if s[0] == 's' and s[1:].isdigit()]
    return P, out

def svr(P, part):
    pre = 'predval_' if part == 'val' else 'pred_'
    out = {}
    for lab, pat in [('분자별 SVR', f'runs/tmp/svr/{pre}svr_mol.pkl'), ('쌍 SVR', f'runs/tmp/svr/{pre}svr_pair_*.pkl')]:
        fs = [f for f in glob.glob(pat) if '_pass_' not in f and '_valall_' not in f]
        if fs:
            c = pd.concat([pd.read_pickle(f) for f in fs]).drop_duplicates(KEY)
            m = P[KEY].merge(c[KEY + ['pred']], on=KEY, how='left')
            if not m.pred.isna().any(): out[lab] = [m.pred.values]
    if not out: print('  (SVR 예측이 없어 SVR 줄은 생략. 파일 맨 위의 svr_delta.py 명령으로 만든다)')
    return out

def per_seed(ps, y, cl):
    """시드별 단일 모델 성능의 평균 ± 표준편차 (cliff / non-cliff / 전체)"""
    if len(ps) == 1: return ''
    f = lambda m: (lambda v: f'{np.mean(v):.3f} ± {np.std(v):.3f}')([rm(p[m] - y[m]) for p in ps])
    return f' | 시드별 {f(cl)} / {f(~cl)} / {f(np.ones_like(cl))} (시드 {len(ps)}개)'

def curves(d, runs):
    """학습 곡선 그림: 시드별 train vs valid (위 전체 / 아래 valid cliff·non-cliff) + cliff·non-cliff별 train vs valid"""
    name = os.path.basename(d.rstrip('/'))
    for enc, lab in [('gine', 'GINE'), ('mb', 'Mole-BERT')]:
        if not os.path.exists(f'{d}/tag_{enc}.txt'): continue
        tag = open(f'{d}/tag_{enc}.txt').read().strip()
        seeds = ','.join(sorted(f[-5] for f in glob.glob(f'{d}/pred_{enc}_s?.pkl')))
        pat = f'{runs}/hist_a0_{tag}_seed{{s}}.csv'
        for script, kind, sub in [('plot_curves.py', '', '위: train vs valid 쌍 전체, 아래: valid cliff vs non-cliff'),
                                  ('plot_curves_split.py', '_split', '위: cliff, 아래: non-cliff (train vs valid)')]:
            subprocess.run([sys.executable, f'experiments/curves/{script}', pat, f'{name} · {lab} 쌍 학습 곡선 ({sub})', f'{d}/curves_{enc}{kind}.png', seeds], check=True)

mode, dirs = sys.argv[1], sys.argv[2:]
if mode == 'val':
    for d in dirs:
        st = json.load(open(f'{d}/settings.json'))
        P, out = load(d, 'val'); y = P.delta.values; cl = np.abs(y) >= 1
        out.update(svr(P, 'val'))
        print(f'\n== {d}  (감쇠 {st["wd"]}, L2-SP {st["l2sp"]}, 시드 {st["seeds"]})')
        print(f'한 군데만 다른 valid 쌍 {len(P):,}개 (cliff {cl.sum():,}). 괄호 = 최종 설계 3 (감쇠 0.01) 기준값')
        for lab, ps in out.items():
            e = np.mean(ps, 0) - y; r = REF.get(lab)
            per = per_seed(ps, y, cl)
            ref = f'  ({r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f})' if r else ''
            print(f'  {lab:14s} {"앙상블" if len(ps) > 1 else "단일  "} cliff {rm(e[cl]):.3f} / non-cliff {rm(e[~cl]):.3f} / 전체 {rm(e):.3f}{ref}{per}')
        curves(d, st['runs']); print(f'  학습 곡선 저장: {d}/curves_*.png')
elif mode == 'test':
    d = dirs[0]; P, out = load(d, 'test'); out.update(svr(P, 'test')); assert len(P) == 38711, len(P)
    y = P.delta.values; cl = np.abs(y) >= 1; tg = P.target.values
    E = {lab: np.mean(ps, 0) for lab, ps in out.items()}
    for lab, ps in out.items():
        e = E[lab] - y; per = per_seed(ps, y, cl)
        print(f'{lab:14s} {"앙상블" if len(ps) > 1 else "단일  "} cliff {rm(e[cl]):.3f} / non-cliff {rm(e[~cl]):.3f} / 전체 {rm(e):.3f}{per}')
    raw = {t: pd.read_csv(f'data/moleculeace/{t}.csv').split.values for t in set(tg)}
    tm = np.where([raw[t][i] == 'test' for t, i in zip(P.target, P.i)], P.i, P.j)   # 평가 분자 (군집 부트스트랩 단위)
    rng = np.random.default_rng(0)
    def boots(mask):
        idx = np.where(mask)[0]; g = pd.DataFrame({'t': tg[idx], 'm': tm[idx]}).groupby(['t', 'm']).indices; keys = list(g)
        return [idx[np.concatenate([g[keys[k]] for k in rng.integers(0, len(keys), len(keys))])] for _ in range(2000)]
    B = {'cliff': (cl, boots(cl)), 'non-cliff': (~cl, boots(~cl)), '전체': (np.ones_like(cl), boots(np.ones_like(cl)))}
    for a_, b_ in [('쌍 GINE', '쌍 SVR'), ('쌍 Mole-BERT', '쌍 SVR'), ('쌍 GINE', '분자별 GINE'), ('쌍 Mole-BERT', '분자별 Mole-BERT')]:
        if a_ not in E or b_ not in E: continue
        print(f'{a_} − {b_}')
        for part, (m, bs) in B.items():
            pa, pb = E[a_], E[b_]
            ta = pd.Series((pa[m] - y[m]) ** 2).groupby(tg[m]).mean() ** 0.5; tb = pd.Series((pb[m] - y[m]) ** 2).groupby(tg[m]).mean() ** 0.5
            lo, hi = np.percentile([rm(pa[i] - y[i]) - rm(pb[i] - y[i]) for i in bs], [2.5, 97.5]); p = wilcoxon(ta, tb).pvalue
            print(f'  {part:9s} {rm(pa[m] - y[m]) - rm(pb[m] - y[m]):+.3f} [{lo:+.3f}, {hi:+.3f}] | 나은 타깃 {int((ta < tb).sum())}/30 | p = {p:.2g} → {"유의" if p < 0.05 and (hi < 0 or lo > 0) else "유의하지 않음"}')
    print('\n※ test를 열었다. experiments/계획_고정_2026-10-18.md 에 "n번째 열기"와 이 결과를 기록할 것')
