# 누수 실험 중간 결과: 비교군(SVR·분자별)을 세 조건(지금/무작위로 빼기/누수 분자 빼기)에서 같은 평가 쌍으로 비교
import glob, os, numpy as np, pandas as pd
from sklearn.svm import SVR
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
os.chdir('/home/bising0922/Capstone')
T = sorted(os.path.basename(f)[:-4] for f in glob.glob('data/moleculeace/CHEMBL*.csv'))
def load_raw(t):
    df = pd.read_csv(f'data/moleculeace/{t}.csv').rename(columns={'y [pEC50/pKi]': 'p'})[['smiles', 'p', 'cliff_mol', 'split']]
    tr = df.index[df.split == 'train']; rng = np.random.default_rng(0)
    df.loc[rng.choice(tr, int(len(tr) * 0.2), replace=False), 'split'] = 'val'
    return df
RAW = {t: load_raw(t) for t in T}
def drop_sets(mode):   # 노트북 drop_sets와 같은 규칙
    test_any = {t: set(d.smiles[d.split == 'test']) for t, d in RAW.items()}
    rng, out = np.random.default_rng(0), {}
    for t, d in RAW.items():
        others = set().union(*[test_any[u] for u in RAW if u != t])
        seen = [k for k in d.index if d.split[k] != 'test']
        leak = [k for k in seen if d.smiles[k] in others]
        if mode == 'drop_leak': out[t] = set(leak)
        else:
            pool = [k for k in seen if d.smiles[k] not in others]
            out[t] = set(rng.choice(pool, len(leak), replace=False).tolist()) if leak else set()
    return out
COND = {'지금': {t: set() for t in T}, '무작위로 빼기': drop_sets('drop_random'), '누수 분자 빼기': drop_sets('drop_leak')}
# 한 군데만 다른 평가 쌍 (모든 조건에 공통으로 남는 쌍: 기준 학습 분자가 어느 조건에서도 빠지지 않은 쌍)
P = pd.concat([pd.read_pickle(f'data/pairs/{t}_v2.pkl').assign(target=t) for t in T], ignore_index=True)
P = P[(P.split == 'test_easy') & (P.reason == 'pass')].reset_index(drop=True)
gone = np.zeros(len(P), bool)
for dr in COND.values():
    gone |= np.array([i in dr[t] or j in dr[t] for t, i, j in zip(P.target, P.i, P.j)])
P = P[~gone].reset_index(drop=True); cl = (P.delta.abs() >= 1).values
print(f'공통 평가 쌍 {len(P):,}개 (cliff {cl.sum():,})')
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X = {t: np.array([FP.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in RAW[t].smiles], dtype=np.float32) for t in T}
def svr_mol(t, drop):
    d, x = RAW[t], X[t]
    tr = np.array([d.split[k] == 'train' and k not in drop for k in d.index]); va = np.array([d.split[k] == 'val' and k not in drop for k in d.index])
    inter = x @ x[tr].T; K = inter / (x.sum(1)[:, None] + x[tr].sum(1)[None, :] - inter)
    best = min([1, 10, 100, 1000], key=lambda C: np.sqrt(np.mean((SVR(kernel='precomputed', C=C).fit(K[tr], d.p[tr]).predict(K[va]) - d.p[va]) ** 2)))
    return SVR(kernel='precomputed', C=best).fit(K[tr], d.p[tr]).predict(K)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def score(predmol):   # predmol[t] = 분자별 예측 배열 → 쌍 Δ̂
    yh = np.array([predmol[t][j] - predmol[t][i] for t, i, j in zip(P.target, P.i, P.j)]); y = P.delta.values
    return rm(yh[cl] - y[cl]), rm(yh[~cl] - y[~cl]), rm(yh - y)
SUF = {'지금': '', '무작위로 빼기': '_rnddrop', '누수 분자 빼기': '_leakdrop'}
rows = []
for cond, drop in COND.items():
    s = score({t: svr_mol(t, drop[t]) for t in T})
    rows.append(dict(조건=cond, 모델='SVR', cliff=s[0], **{'non-cliff': s[1], '전체': s[2]}))
    for name, pat in [('GNN 분자별', f'runs/preds_mol_t30_h64_l3_dr0.3_wd0.0001_ep1000{SUF[cond]}_[0-9]*.pkl'),
                      ('Mole-BERT 분자별', f'runs/preds_molpt_t30_h64_l3_dr0.3_wd0.0001_ep1000{SUF[cond]}_molebert_*.pkl')]:
        f = glob.glob(pat)
        if not f: continue
        c = pd.read_pickle(f[0])
        ss = np.array([score({t: g[f's{s}'].values for t, g in c.groupby('target')}) for s in range(3)])
        rows.append(dict(조건=cond, 모델=name, cliff=ss[:, 0].mean(), **{'non-cliff': ss[:, 1].mean(), '전체': ss[:, 2].mean(), 'cliff 시드 흔들림': np.ptp(ss[:, 0])}))
# 쌍 모델: 조건별 저장된 예측값 (행 순서 = 노트북 pairs_all)
KEEP = {'train': 'train', 'val_easy': 'val', 'test_easy': 'test'}
ALL = pd.concat([pd.read_pickle(f'data/pairs/{t}_v2.pkl').assign(target=t) for t in T], ignore_index=True)
ALL = ALL[ALL.split.isin(KEEP)].reset_index(drop=True)
TAGB = 't30_h64_dr0.3_wd0.0001_exp1_anchored_v2'
for cond, drop in COND.items():
    keep = np.array([not (i in drop[t] or j in drop[t]) for t, i, j in zip(ALL.target, ALL.i, ALL.j)])
    A = ALL[keep].reset_index(drop=True)[['target', 'i', 'j']]
    for name, pre, pat in [('GINE 쌍', 'pred_gnn', f'runs/preds_{TAGB}{SUF[cond]}_[0-9]*.pkl'), ('Mole-BERT 쌍', 'pred_pt', f'runs/preds_pt_{TAGB}{SUF[cond]}_molebert_*.pkl')]:
        c = pd.read_pickle(glob.glob(pat)[0]); assert len(c) == len(A) and (c[['target', 'i', 'j']].values == A.values).all()
        m = P[['target', 'i', 'j']].merge(c, on=['target', 'i', 'j'], how='left'); assert len(m) == len(P) and not m.isna().any().any()
        y = P.delta.values
        ss = np.array([[rm(m[f'{pre}_a0_s{s}'].values[cl] - y[cl]), rm(m[f'{pre}_a0_s{s}'].values[~cl] - y[~cl]), rm(m[f'{pre}_a0_s{s}'].values - y)] for s in range(3)])
        e = m[[f'{pre}_a0_s{s}' for s in range(3)]].mean(1).values
        rows.append(dict(조건=cond, 모델=name, cliff=ss[:, 0].mean(), **{'non-cliff': ss[:, 1].mean(), '전체': ss[:, 2].mean(), 'cliff 시드 흔들림': np.ptp(ss[:, 0])}))
        rows.append(dict(조건=cond, 모델=name + ' 앙상블', cliff=rm(e[cl] - y[cl]), **{'non-cliff': rm(e[~cl] - y[~cl]), '전체': rm(e - y)}))
r = pd.DataFrame(rows).set_index(['모델', '조건'])
pd.set_option('display.width', 200)
print(r.sort_index(level=0, sort_remaining=False).round(3).to_string())
