"""과적합 대책 후보 비교 (한 군데만 다른 검증 쌍, Mole-BERT 쌍 시드 0). 평가 쌍은 보지 않는다
기준: 같은 설계·대책 없음 시드 0 — 설계가 예전 설계와 같으면 runs/tmp/head/pred_head_mb_s0.pkl, 아니면 final2_mb_s0
규칙(계획 문서 10-19 01:00): 기준보다 전체 오차가 0.01 이상 낮으면 채택 (여러 개면 가장 낮은 것 하나)"""
import json, os
import numpy as np, pandas as pd

D = '/home/bising0922/Capstone/runs/tmp'; KEY = ['target', 'i', 'j']
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
cfg = json.load(open(f'{D}/final2/config.json'))
old = cfg == dict(readout='mean', frag=True, head='big', val_pairs='all')
bf = f'{D}/head/pred_head_mb_s0.pkl' if old else f'{D}/final2/pred_final2_mb_s0.pkl'
B = pd.read_pickle(bf); V = B[B.split.str.startswith('val') & (B.reason == 'pass')].reset_index(drop=True)
y = V.delta.values; cl = np.abs(y) >= 1
def row(lab, p): e = p - y; return dict(모델=lab, 전체=rm(e), cliff=rm(e[cl]), non_cliff=rm(e[~cl]))
rows = [row('대책 없음 (기준)', V.pred_pt_a0_s0.values)]
for r, lab in [('reg_drop', '헤드 드롭아웃 0.2'), ('reg_adamw', 'AdamW 감쇠 0.01'), ('reg_ptlr', '인코더 학습률 3e-5')]:
    f = f'{D}/final2/pred_{r}_mb_s0.pkl'
    if os.path.exists(f):
        d = pd.read_pickle(f); m = V[KEY].merge(d[KEY + ['pred_pt_a0_s0']], on=KEY, how='left'); rows.append(row(lab, m.pred_pt_a0_s0.values))
T = pd.DataFrame(rows); base = T.전체[0]
T['개선'] = base - T.전체; T['채택'] = ['–'] + ['예' if g >= 0.01 else '아니요' for g in T.개선[1:]]
print(f'설계 {cfg} (예전 설계와 같음: {old}) | 한 군데만 다른 검증 쌍 {len(V):,}개')
print(T.round(3).to_string(index=False))
