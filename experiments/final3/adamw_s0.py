# AdamW 효과 확인 (한 군데만 다른 검증 쌍, Mole-BERT 쌍 시드 0): 예전 설계 시드 0 vs 예전 설계 + AdamW 0.01 시드 0. 평가 쌍은 보지 않는다
import glob, numpy as np, pandas as pd
D = '/home/bising0922/Capstone/runs/tmp'; KEY = ['target', 'i', 'j']; rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
B = pd.read_pickle(f'{D}/head/pred_head_mb_s0.pkl'); V = B[B.split.str.startswith('val') & (B.reason == 'pass')].reset_index(drop=True)
y = V.delta.values; cl = np.abs(y) >= 1
def get(f):
    d = pd.read_pickle(f); m = V[KEY].merge(d[KEY + ['pred_pt_a0_s0']], on=KEY, how='left'); assert not m.pred_pt_a0_s0.isna().any(); return m.pred_pt_a0_s0.values
c = pd.concat([pd.read_pickle(x) for x in sorted(glob.glob(f'{D}/svr/predval_svr_pair_g?.pkl'))])
rows = [('SVR 쌍', V[KEY].merge(c[KEY + ['pred']], on=KEY, how='left').pred.values),
        ('예전 설계 시드 0 (AdamW 없음)', V.pred_pt_a0_s0.values), ('예전 설계 + AdamW 시드 0', get(f'{D}/final3/pred_final3_mb_s0.pkl')),
        ('(참고) 조각 인코더 없는 설계 시드 0', get(f'{D}/final2/pred_final2_mb_s0.pkl')), ('(참고) 조각 인코더 없는 설계 + AdamW 시드 0', get(f'{D}/final2/pred_reg_adamw_mb_s0.pkl'))]
print(pd.DataFrame([dict(모델=k, 전체=rm(p - y), cliff=rm((p - y)[cl]), non_cliff=rm((p - y)[~cl])) for k, p in rows]).round(3).to_string(index=False))
