"""원인 찾기 결과로 최종 설계 2를 정한다 (규칙은 결과를 보기 전에 고정, 계획_고정_2026-10-18.md 10-18 23:50)
- 읽기: 평균 (r_mean 결과로 이미 확정: 한 군데만 다른 검증 쌍 전체 오차 0.636 → 0.603)
- 조각 인코더 / 헤드 / 멈춤 기준: 되돌린 실험(시드 3)이 최종 설계 시드 3보다 한 군데만 다른 검증 쌍의 전체 오차로 0.01 이상 좋으면 되돌린다. 아니면 최종 설계 그대로
결과: config.json, decide.log"""
import json
import numpy as np, pandas as pd

D = '/home/bising0922/Capstone/runs/tmp'; KEY = ['target', 'i', 'j']
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
A = pd.read_pickle(f'{D}/final/pred_final_mb_s3.pkl'); V = A[A.split.str.startswith('val') & (A.reason == 'pass')].reset_index(drop=True)
y = V.delta.values
base = rm(V.pred_pt_a0_s3.values - y)
def score(n):
    d = pd.read_pickle(f'{D}/revert/pred_{n}.pkl'); m = V[KEY].merge(d[KEY + ['pred_pt_a0_s3']], on=KEY, how='left')
    assert not m.pred_pt_a0_s3.isna().any(); return rm(m.pred_pt_a0_s3.values - y)
cfg = {'readout': 'mean', 'frag': False, 'head': 'big2048', 'val_pairs': 'pass'}
lines = [f'최종 설계 시드 3 (한 군데만 다른 검증 쌍 전체 오차): {base:.4f}']
for n, key, rev in [('r_mean', 'readout', 'mean'), ('r_frag', 'frag', True), ('r_head', 'head', 'big'), ('r_valall', 'val_pairs', 'all')]:
    s = score(n); gain = base - s
    if n != 'r_mean' and gain >= 0.01: cfg[key] = rev
    lines.append(f'{n}: {s:.4f} (개선 {gain:+.4f}) → {key} = {cfg[key]}')
lines.append(f'최종 설계 2: {cfg}')
open(f'{D}/final2/decide.log', 'w').write('\n'.join(lines) + '\n'); json.dump(cfg, open(f'{D}/final2/config.json', 'w'))
print('\n'.join(lines))
