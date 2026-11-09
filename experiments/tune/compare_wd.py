"""가중치 감쇠별 전체 성능 비교표 (valid만). 저장소 최상위에서 실행한다
  python experiments/tune/compare_wd.py
한 군데만 다른 valid 쌍에서 6개 모델 × 감쇠 0.01·0.02·0.03의 시드 앙상블과 시드별 평균 ± 표준편차 (cliff / non-cliff / 전체)
  0.01 = 최종 모델 (runs/tmp/final3, runs/의 분자별 예측 캐시), 0.02·0.03 = runs/tune/{gine,mb}_wd*/
  예측 파일이 없는 칸은 팀원이 기록한 숫자를 쓰고 '팀원 기록'으로 표시한다 (쌍 GINE 0.03은 팀원 서버에서 학습)
"""
import glob, os
import numpy as np, pandas as pd

KEY = ['target', 'i', 'j']
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
MOLCACHE = 'runs/preds_{kind}_t30_h300_l3_dr0.3_wd0.01_ep1000_leakdrop_head1024_adamw{mb}_*.pkl'
molcache = lambda kind, mb: [f for f in glob.glob(MOLCACHE.format(kind=kind, mb=mb)) if f[len(MOLCACHE.format(kind=kind, mb=mb)) - 5:-4].isdigit()]   # 이름 끝이 숫자(수정 시각)뿐인 것만: L2-SP 등 다른 설정 캐시 제외
SRC = {   # (모델, 감쇠) → 예측 파일 목록
    ('쌍 GINE', 0.01): sorted(glob.glob('runs/tmp/final3/pred_final3_gine_s?.pkl')),
    ('쌍 Mole-BERT', 0.01): sorted(glob.glob('runs/tmp/final3/pred_final3_mb_s?.pkl')),
    ('분자별 GINE', 0.01): molcache('mol', ''),
    ('분자별 Mole-BERT', 0.01): molcache('molpt', '_molebert'),
}
for w in (0.02, 0.03):
    SRC[('쌍 GINE', w)] = sorted(glob.glob(f'runs/tune/gine_wd{w}/pred_gine_s?.pkl'))
    SRC[('쌍 Mole-BERT', w)] = sorted(glob.glob(f'runs/tune/mb_wd{w}/pred_mb_s?.pkl'))
    SRC[('분자별 GINE', w)] = [f for f in [f'runs/tune/gine_wd{w}/pred_mol_gine.pkl', f'runs/tune/mb_wd{w}/pred_mol_gine.pkl'] if os.path.exists(f)][:1]
    SRC[('분자별 Mole-BERT', w)] = [f for f in [f'runs/tune/mb_wd{w}/pred_mol_mb.pkl'] if os.path.exists(f)]
# 예측 파일이 이 서버에 없는 칸: 팀원 기록 (experiments/계획_고정_2026-10-18.md, 10-21 00:40)
TEAM = {('쌍 GINE', 0.03): ((0.867, 0.432, 0.560), ((0.892, 0.005), (0.460, 0.005), (0.585, 0.004)))}

base = pd.read_pickle(SRC[('쌍 GINE', 0.01)][0])
P = base[base.split.str.startswith('val') & (base.reason == 'pass')][KEY + ['delta']].reset_index(drop=True)
y = P.delta.values; masks = [np.abs(y) >= 1, np.abs(y) < 1, np.ones(len(y), bool)]

def preds(lab, fs):
    if lab.startswith('쌍'):
        col = 'pred_gnn_a0_s' if 'GINE' in lab else 'pred_pt_a0_s'
        return [P[KEY].merge(pd.read_pickle(f)[KEY + [f'{col}{f[-5]}']], on=KEY, how='left')[f'{col}{f[-5]}'].values for f in fs]
    assert len(fs) == 1, fs
    c = pd.read_pickle(fs[0]); g = {t: x for t, x in c.groupby('target')}
    return [np.array([g[t][s].values[j] - g[t][s].values[i] for t, i, j in zip(P.target, P.i, P.j)]) for s in c.columns if s[0] == 's' and s[1:].isdigit()]

def svr(pat):
    c = pd.concat([pd.read_pickle(f) for f in glob.glob(pat) if '_pass_' not in f and '_valall_' not in f]).drop_duplicates(KEY)
    return [P[KEY].merge(c[KEY + ['pred']], on=KEY, how='left').pred.values]

rows = []
for lab, ps in [('분자별 SVR', svr('runs/tmp/svr/predval_svr_mol.pkl')), ('쌍 SVR', svr('runs/tmp/svr/predval_svr_pair_*.pkl'))]:
    rows.append([lab, '–'] + [f'{rm(ps[0][m] - y[m]):.3f}' for m in masks] + ['–'] * 3)
for lab in ['분자별 GINE', '쌍 GINE', '분자별 Mole-BERT', '쌍 Mole-BERT']:
    for w in (0.01, 0.02, 0.03):
        fs = SRC.get((lab, w), [])
        if fs:
            ps = preds(lab, fs); e = np.mean(ps, 0) - y
            ens = [f'{rm(e[m]):.3f}' for m in masks]
            per = [f'{np.mean(v):.3f} ± {np.std(v):.3f}' for v in ([rm(p[m] - y[m]) for p in ps] for m in masks)] if len(ps) > 1 else ['–'] * 3
            note = '' if len(ps) == 5 else f' (시드 {len(ps)}개)'
        elif (lab, w) in TEAM:
            (a, b, c), sd = TEAM[(lab, w)]
            ens = [f'{a:.3f}', f'{b:.3f}', f'{c:.3f}']; per = [f'{m:.3f} ± {s:.3f}' for m, s in sd]; note = ' (팀원 기록)'
        else:
            ens = per = ['학습 중'] * 3; note = ''
        rows.append([lab, f'{w}{" (최종)" if w == 0.01 else ""}{note}'] + ens + per)

print(f'한 군데만 다른 valid 쌍 {len(P):,}개 (cliff {int(masks[0].sum()):,}). 앙상블 = 시드 5개 예측 평균, 시드별 = 평균 ± 표준편차. test는 열지 않음\n')
print('| 모델 | 감쇠 | cliff | non-cliff | 전체 | 시드별 cliff | 시드별 non-cliff | 시드별 전체 |')
print('|---|---|---|---|---|---|---|---|')
for r in rows: print('| ' + ' | '.join(r) + ' |')
