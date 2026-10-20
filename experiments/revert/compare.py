# 원인 찾기 비교: 검증 쌍만 (한 군데만 다른 쌍 / 전체). 기준 = 최종 설계 Mole-BERT 쌍 시드 3, 참고 = SVR 쌍
import glob, numpy as np, pandas as pd
D = '/home/bising0922/Capstone/runs/tmp'; KEY = ['target', 'i', 'j']; rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
A = pd.read_pickle(f'{D}/final/pred_final_mb_s3.pkl'); V = A[A.split.str.startswith('val')].reset_index(drop=True)
y = V.delta.values; cl = np.abs(y) >= 1; ps = (V.reason == 'pass').values
c = pd.concat([pd.read_pickle(x) for x in sorted(glob.glob(f'{D}/svr/predval_svr_pair_g?.pkl'))])
P = {'SVR 쌍': V[KEY].merge(c[KEY + ['pred']], on=KEY, how='left').pred.values, '최종 설계 (기준)': V.pred_pt_a0_s3.values}
for n, lab in [('r_frag', '조각 인코더 다시 넣기'), ('r_mean', '평균 읽기로'), ('r_head', '헤드 1024 → 1024로'), ('r_valall', '멈춤 기준 검증 쌍 전체로')]:
    f = f'{D}/revert/pred_{n}.pkl'
    try:
        d = pd.read_pickle(f); m = V[KEY].merge(d[KEY + ['pred_pt_a0_s3']], on=KEY, how='left'); P[lab] = m.pred_pt_a0_s3.values
    except FileNotFoundError:
        pass
rows = [dict(모델=k, 한곳_전체=rm((p - y)[ps]), 한곳_cliff=rm((p - y)[ps & cl]), 한곳_non_cliff=rm((p - y)[ps & ~cl]), 전체쌍_전체=rm(p - y)) for k, p in P.items()]
print(pd.DataFrame(rows).round(3).to_string(index=False))
