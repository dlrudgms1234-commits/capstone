# ECFP + SVR 분자별 / 쌍 (타깃별). C는 검증 쌍의 Δ 오차로 고른다 (그래프 쌍 모델의 조기 종료와 같은 검증 쌍)
# 사용: python svr_delta.py mol                       → pred_svr_mol.pkl (기존 기준 재현 확인 포함)
#       python svr_delta.py pair <타깃,...|all> [C,...] → pred_svr_pair_<이름>.pkl
#       python svr_delta.py time <타깃> <C>              → 커널 만들기·학습 시간만 잰다
# 데이터 규칙은 runs/tmp/ecfp/report3.py 설정 2(누수 분자 빼기)와 같다
import glob, os, sys, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.svm import SVR
RDLogger.DisableLog('rdApp.*')
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/svr'; os.chdir(CAP)
T = sorted(os.path.basename(f)[:-4] for f in glob.glob('data/moleculeace/CHEMBL*.csv'))
C_GRID = [1, 10, 100, 1000]
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)   # 노트북 SVR과 같은 ECFP4 2,048비트
log = lambda *m: print(time.strftime('%m-%d %H:%M:%S'), *m, flush=True)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))

def splits():
    raw = {}
    for t in T:
        d = pd.read_csv(f'data/moleculeace/{t}.csv').rename(columns={'y [pEC50/pKi]': 'p'})[['smiles', 'p', 'split']]
        tr = d.index[d.split == 'train']; d.loc[np.random.default_rng(0).choice(tr, int(len(tr) * 0.2), replace=False), 'split'] = 'val'
        raw[t] = d
    test_any = {t: set(d.smiles[d.split == 'test']) for t, d in raw.items()}
    for t, d in raw.items():   # 누수 분자 빼기: 다른 타깃의 평가 분자는 이 타깃의 학습·검증에서 뺀다
        others = set().union(*[test_any[u] for u in raw if u != t])
        d.loc[[k for k in d.index if d.split[k] != 'test' and d.smiles[k] in others], 'split'] = 'drop'
    return raw

KEEP = {'train': 'train', 'val_easy': 'val', 'test_easy': 'test'}
def pairs_of(t, d):
    p = pd.read_pickle(f'data/pairs/{t}_v2.pkl'); p = p[p.split.isin(KEEP)].assign(split=lambda x: x.split.map(KEEP))
    sp = d.split
    p = p[[sp[i] != 'drop' and sp[j] != 'drop' for i, j in zip(p.i, p.j)]]
    p = p[(p.split != 'test') | (p.reason == 'pass')]   # 평가는 한 군데만 다른 쌍
    return p.reset_index(drop=True)

def tanimoto_all(d):
    x = np.array([FPG.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in d.smiles], dtype=np.float64)
    inter = x @ x.T; n = x.sum(1)
    return inter / (n[:, None] + n[None, :] - inter)

def run_mol(raw):
    rows, new_pred, old_pred = [], [], []
    for t in T:
        d = raw[t]; K = tanimoto_all(d); p = pairs_of(t, d)
        tr = (d.split == 'train').values; y = d.p.values
        va, te = p[p.split == 'val'], p[p.split == 'test']
        vm = (d.split == 'val').values
        res = {}
        for C in C_GRID:
            pm = SVR(kernel='precomputed', C=C).fit(K[tr][:, tr], y[tr]).predict(K[:, tr])
            res[C] = (rm(pm[va.j.values] - pm[va.i.values] - va.delta.values), rm(pm[vm] - y[vm]), pm)
        c_new = min(C_GRID, key=lambda C: res[C][0]); c_old = min(C_GRID, key=lambda C: res[C][1])
        new_pred.append(te[['i', 'j', 'delta']].assign(target=t, pred=res[c_new][2][te.j.values] - res[c_new][2][te.i.values]))
        old_pred.append(te[['i', 'j', 'delta']].assign(target=t, pred=res[c_old][2][te.j.values] - res[c_old][2][te.i.values]))
        rows.append(dict(target=t, C_delta=c_new, C_mol=c_old))
        log(t, 'C(Δ 기준)', c_new, 'C(분자 기준)', c_old)
    new, old = pd.concat(new_pred, ignore_index=True), pd.concat(old_pred, ignore_index=True)
    new.to_pickle(f'{OUT}/pred_svr_mol.pkl'); old.to_pickle(f'{OUT}/pred_svr_mol_oldC.pkl')
    pd.DataFrame(rows).to_csv(f'{OUT}/svr_mol_C.csv', index=False)
    for name, df in [('분자 기준 C (기존, 0.914/0.455/0.591 재현 확인)', old), ('Δ 기준 C (새로)', new)]:
        e = df.pred.values - df.delta.values; cl = np.abs(df.delta.values) >= 1
        log(f'{name}: 쌍 {len(df):,} | cliff {rm(e[cl]):.3f} non-cliff {rm(e[~cl]):.3f} 전체 {rm(e):.3f}')

def pair_kernel(K, a1, b1, a2, b2, chunk=2048):
    """쌍 커널 K((A,B),(C,D)) = T(A,C)·T(B,D). 메모리를 아끼려 줄 단위로 만든다"""
    out = np.empty((len(a1), len(a2)))
    for s in range(0, len(a1), chunk):
        out[s:s + chunk] = K[a1[s:s + chunk]][:, a2]
        out[s:s + chunk] *= K[b1[s:s + chunk]][:, b2]
    return out

def run_pair_target(t, d, grid):
    K = tanimoto_all(d); p = pairs_of(t, d)
    tr, va, te = (p[p.split == s] for s in ('train', 'val', 'test'))
    a = np.r_[tr.i.values, tr.j.values]; b = np.r_[tr.j.values, tr.i.values]     # 두 방향으로 학습
    y = np.r_[tr.delta.values, -tr.delta.values]
    t0 = time.time(); Ktr = pair_kernel(K, a, b, a, b); t_k = time.time() - t0
    def pred(m, q):   # 반대칭 평균 Δ̂ = [g(A,B) − g(B,A)] / 2
        f = m.predict(pair_kernel(K, q.i.values, q.j.values, a, b)); r = m.predict(pair_kernel(K, q.j.values, q.i.values, a, b))
        return (f - r) / 2
    res = {}
    for C in grid:
        t0 = time.time(); m = SVR(kernel='precomputed', C=C, cache_size=2000).fit(Ktr, y); t_fit = time.time() - t0
        pv, pt = pred(m, va), pred(m, te)
        res[C] = (rm(pv - va.delta.values), pt)
        log(t, f'C={C} 학습쌍 {len(y):,} 커널 {t_k:.0f}s 학습 {t_fit:.0f}s 서포트벡터 {len(m.support_):,} 검증 Δ RMSE {res[C][0]:.4f}')
    best = min(grid, key=lambda C: res[C][0])
    return te[['i', 'j', 'delta']].assign(target=t, pred=res[best][1], C=best)

if __name__ == '__main__':
    mode = sys.argv[1]; raw = splits()
    if mode == 'mol':
        run_mol(raw)
    elif mode == 'time':
        run_pair_target(sys.argv[2], raw[sys.argv[2]], [float(sys.argv[3])])
    elif mode == 'pair':
        ts = T if sys.argv[2] == 'all' else sys.argv[2].split(',')
        grid = [float(c) for c in sys.argv[3].split(',')] if len(sys.argv) > 3 else C_GRID
        name = sys.argv[4] if len(sys.argv) > 4 else ('all' if sys.argv[2] == 'all' else ts[0])
        out = pd.concat([run_pair_target(t, raw[t], grid) for t in ts], ignore_index=True)
        out.to_pickle(f'{OUT}/pred_svr_pair_{name}.pkl'); log('끝', name, len(out))
