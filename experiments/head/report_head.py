# 헤드 맞추기 결과: 누수 없음·우리 방식, 한 군데만 다른 평가 쌍 (있는 시드만)
import glob, os, numpy as np, pandas as pd
os.chdir('/home/bising0922/Capstone/runs/tmp/head')
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
lines = []
for kind, pre, name in [('gine', 'pred_gnn', 'GINE 쌍 (큰 헤드)'), ('mb', 'pred_pt', 'Mole-BERT 쌍 (큰 헤드)')]:
    fs = sorted(glob.glob(f'pred_head_{kind}_s?.pkl'))
    if not fs: continue
    ps = [pd.read_pickle(f) for f in fs]
    d = ps[0]; ev = ((d.split == 'test') & (d.reason == 'pass')).values
    y = d.delta.values[ev]; cl = np.abs(y) >= 1
    preds = [p[[c for c in p.columns if c.startswith(pre)][0]].values[ev] for p in ps]
    seed = [(rm(q[cl] - y[cl]), rm(q[~cl] - y[~cl])) for q in preds]
    e = np.mean(preds, 0)
    lines.append(f'{name}: 시드 {len(fs)}개 | 시드별 cliff {[round(s[0], 3) for s in seed]} non-cliff {[round(s[1], 3) for s in seed]} | '
                 f'앙상블 cliff {rm(e[cl] - y[cl]):.3f} non-cliff {rm(e[~cl] - y[~cl]):.3f} 전체 {rm(e - y):.3f} | 평가 쌍 {ev.sum()}')
open('/home/bising0922/Capstone/runs/_diag.txt', 'w').write('\n'.join(lines) + '\n')
