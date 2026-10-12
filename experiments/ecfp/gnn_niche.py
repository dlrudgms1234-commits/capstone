# GNN 쌍 모델이 ECFP 쌍보다 나은 곳이 있는가 (설정 2: 누수 없음·우리 방식, 재학습 없음)
#  ① 평가 분자가 학습 분자와 얼마나 다른가(가장 가까운 학습 분자와의 유사도)별 cliff 오차
#  ② 두 모델 예측 평균(앙상블)이 각각보다 나은가 = GNN이 ECFP에 없는 정보를 갖는가
#  ③ 타깃별로 GNN이 이기는 타깃 수
import sys, os, glob, numpy as np, pandas as pd
sys.argv = ['x', '2']
src = open('/home/bising0922/Capstone/runs/tmp/ecfp/report3.py').read()
exec(src[:src.index('tables = []')])
from rdkit import DataStructs
setting = 2; raw = splits(setting); A = all_pairs(setting, raw)
ev = ((A.split == 'test') & (A.reason == 'pass')).values
E = A[ev].reset_index(drop=True); y = E.delta.values; cl = np.abs(y) >= 1
P = {}
P['ECFP + MLP 쌍'] = np.mean([E[['target', 'i', 'j']].merge(pd.read_pickle(f'runs/tmp/ecfp/pred_shared_pair_drop_leak_s{s}.pkl')[['target', 'i', 'j', 'pred']], on=['target', 'i', 'j'], how='left').pred.values for s in range(3)], 0)
for name, pre, pat in [('GINE 쌍', 'pred_gnn', f'runs/preds_{TAGB}_v2_leakdrop_[0-9]*.pkl'), ('Mole-BERT 쌍', 'pred_pt', f'runs/preds_pt_{TAGB}_v2_leakdrop_molebert_*.pkl')]:
    c = pd.read_pickle(glob.glob(pat)[0]); P[name] = np.mean([c[f'{pre}_a0_s{s}'].values[ev] for s in range(3)], 0)
P['ECFP + Mole-BERT 예측 평균'] = (P['ECFP + MLP 쌍'] + P['Mole-BERT 쌍']) / 2
P['ECFP + GINE 예측 평균'] = (P['ECFP + MLP 쌍'] + P['GINE 쌍']) / 2
# 평가 분자(쌍에서 test 쪽)와 그 타깃 학습 분자 중 가장 가까운 것의 유사도
FP2 = {t: [FPG.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.smiles] for t, d in raw.items()}
near = {}
for t, d in raw.items():
    tr = [FP2[t][k] for k in d.index if d.split[k] == 'train']
    for k in d.index[d.split == 'test']:
        near[t, k] = max(DataStructs.BulkTanimotoSimilarity(FP2[t][k], tr))
kb = np.where([raw[t].split[i] == 'test' for t, i in zip(E.target, E.i)], E.i, E.j)
sim = np.array([near[t, k] for t, k in zip(E.target, kb)])
bins = pd.cut(sim, [0, 0.6, 0.7, 0.8, 1.0], labels=['0.6 미만 (낯선 분자)', '0.6–0.7', '0.7–0.8', '0.8 이상 (아주 비슷)'], include_lowest=True)
rows = {}
for name, p in P.items():
    r = {'cliff 전체': rm(p[cl] - y[cl]), 'non-cliff 전체': rm(p[~cl] - y[~cl])}
    for b in bins.categories:
        m = (bins == b) & cl
        r[f'cliff · {b}'] = rm(p[m] - y[m])
    rows[name] = r
pd.set_option('display.width', 250)
print('평가 쌍', len(E), '| cliff 쌍 수 (유사도 구간별):', {b: int(((bins == b) & cl).sum()) for b in bins.categories})
print(pd.DataFrame(rows).T.round(3).to_string())
# 타깃별 cliff: GNN 쌍이 ECFP 쌍보다 나은 타깃 수
for g in ['GINE 쌍', 'Mole-BERT 쌍']:
    w = sum(rm(P[g][(E.target == t).values & cl] - y[(E.target == t).values & cl]) < rm(P['ECFP + MLP 쌍'][(E.target == t).values & cl] - y[(E.target == t).values & cl]) for t in T)
    print(f'타깃별 cliff에서 {g}이 ECFP 쌍보다 나은 타깃: {w}/30')
