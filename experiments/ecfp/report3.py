# 세 설정 통일 비교: SVR / ECFP+MLP 쌍 / GINE 쌍 / Mole-BERT 쌍
# 쌍 모델은 모두 30개 타깃 공용 · 유사 쌍 전부로 학습 · 시드 3 앙상블. 평가 = 그 설정의 한 군데만 다른 평가 쌍
#   1 누수 있음 (MoleculeACE 분할) / 2 우리 방식 누수 없음 (누수 분자를 다른 타깃 학습에서 뺌) / 3 팀원 방식 누수 없음 (평가로 옮김)
import glob, os, sys, numpy as np, pandas as pd
from sklearn.svm import SVR
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
os.chdir('/home/bising0922/Capstone')
T = sorted(os.path.basename(f)[:-4] for f in glob.glob('data/moleculeace/CHEMBL*.csv'))
KEEP = {'train': 'train', 'val_easy': 'val', 'test_easy': 'test'}
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))

def splits(setting):
    raw = {}
    for t in T:
        d = pd.read_csv(f'data/moleculeace/{t}.csv').rename(columns={'y [pEC50/pKi]': 'p'})[['smiles', 'p', 'split']]
        if setting != 3:
            tr = d.index[d.split == 'train']; d.loc[np.random.default_rng(0).choice(tr, int(len(tr) * 0.2), replace=False), 'split'] = 'val'
        raw[t] = d
    if setting == 2:
        test_any = {t: set(d.smiles[d.split == 'test']) for t, d in raw.items()}
        for t, d in raw.items():
            others = set().union(*[test_any[u] for u in raw if u != t])
            d.loc[[k for k in d.index if d.split[k] != 'test' and d.smiles[k] in others], 'split'] = 'drop'
    if setting == 3:
        allr = pd.concat([d[['smiles', 'split']] for d in raw.values()])
        sp_ = allr.groupby('smiles').split.agg(lambda x: 'test' if (x == 'test').any() else 'train')
        trs = sp_.index[sp_ == 'train']; sp_[np.random.default_rng(0).choice(trs, int(len(trs) * 0.2), replace=False)] = 'val'
        for d in raw.values(): d['split'] = d.smiles.map(sp_)
    return raw

def all_pairs(setting, raw):   # 노트북 pairs_all 순서 (예측값 캐시 행 순서와 같다)
    v = 'v3' if setting == 3 else 'v2'
    A = pd.concat([pd.read_pickle(f'data/pairs/{t}_{v}.pkl').assign(target=t) for t in T], ignore_index=True)
    A = A[A.split.isin(KEEP)].assign(split=lambda d: d.split.map(KEEP)).reset_index(drop=True)
    if setting == 2:
        A = A[[raw[t].split[i] != 'drop' and raw[t].split[j] != 'drop' for t, i, j in zip(A.target, A.i, A.j)]].reset_index(drop=True)
    return A

FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)   # 노트북 SVR과 같음
def svr_mols(raw):
    out = {}
    for t, d in raw.items():
        x = np.array([FPG.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in d.smiles], dtype=np.float32)
        tr, va = (d.split == 'train').values, (d.split == 'val').values
        inter = x @ x[tr].T; K = inter / (x.sum(1)[:, None] + x[tr].sum(1)[None, :] - inter)
        best = min([1, 10, 100, 1000], key=lambda C: rm(SVR(kernel='precomputed', C=C).fit(K[tr], d.p[tr]).predict(K[va]) - d.p.values[va]))
        out[t] = SVR(kernel='precomputed', C=best).fit(K[tr], d.p[tr]).predict(K)
    return out

TAGB = 't30_h64_dr0.3_wd0.0001_exp1_anchored'
CACHE = {1: ('_v2', ''), 2: ('_v2', '_leakdrop'), 3: ('_v3', '')}
ECFP = {1: 'shared_pair_none', 2: 'shared_pair_drop_leak', 3: 'shared_pair_v3'}
NAME = {1: '1. 누수 있음 (MoleculeACE 분할)', 2: '2. 누수 없음 · 우리 방식 (누수 분자를 학습에서 뺌)', 3: '3. 누수 없음 · 팀원 방식 (평가로 옮김)'}
tables = []
for setting in [int(a) for a in sys.argv[1:]] or [1, 2, 3]:
    raw = splits(setting); A = all_pairs(setting, raw)
    ev = ((A.split == 'test') & (A.reason == 'pass')).values
    E = A[ev].reset_index(drop=True); y = E.delta.values; cl = np.abs(y) >= 1
    preds = {}
    sv = svr_mols(raw)
    preds['SVR'] = [np.array([sv[t][j] - sv[t][i] for t, i, j in zip(E.target, E.i, E.j)])]
    fs = [f'runs/tmp/ecfp/pred_{ECFP[setting]}_s{s}.pkl' for s in range(3)]
    if all(os.path.exists(f) for f in fs):
        preds['ECFP + MLP 쌍'] = [E[['target', 'i', 'j']].merge(pd.read_pickle(f)[['target', 'i', 'j', 'pred']], on=['target', 'i', 'j'], how='left').pred.values for f in fs]
    v, suf = CACHE[setting]
    MSUF = {1: '', 2: '_leakdrop', 3: '_v3'}[setting]   # 분자별 모델 (같은 인코더로 분자 하나씩 예측 후 빼기)
    for name, pat in [('GINE 분자별', f'runs/preds_mol_t30_h64_l3_dr0.3_wd0.0001_ep1000{MSUF}_[0-9]*.pkl'),
                      ('Mole-BERT 분자별', f'runs/preds_molpt_t30_h64_l3_dr0.3_wd0.0001_ep1000{MSUF}_molebert_*.pkl')]:
        c = pd.read_pickle(glob.glob(pat)[0])
        preds[name] = [np.array([pm[t][j] - pm[t][i] for t, i, j in zip(E.target, E.i, E.j)])
                       for pm in [{t: g[f's{s}'].values for t, g in c.groupby('target')} for s in range(3)]]
    for name, pre, pat in [('GINE 쌍', 'pred_gnn', f'runs/preds_{TAGB}{v}{suf}_[0-9]*.pkl'), ('Mole-BERT 쌍', 'pred_pt', f'runs/preds_pt_{TAGB}{v}{suf}_molebert_*.pkl')]:
        c = pd.read_pickle(glob.glob(pat)[0]); assert len(c) == len(A) and (c[['target', 'i', 'j']].values == A[['target', 'i', 'j']].values).all(), name
        preds[name] = [c[f'{pre}_a0_s{s}'].values[ev] for s in range(3)]
    rows = {}
    for name, ps in preds.items():
        assert not any(np.isnan(p).any() for p in ps), name
        e = np.mean(ps, 0)   # 시드 3개 앙상블 (SVR은 하나)
        seed_cl = [rm(p[cl] - y[cl]) for p in ps]
        rows[name] = {'cliff': rm(e[cl] - y[cl]), 'non-cliff': rm(e[~cl] - y[~cl]), '전체': rm(e - y),
                      'cliff 시드 흔들림': np.ptp(seed_cl) if len(ps) > 1 else np.nan}
    r = pd.DataFrame(rows).T
    print(f'\n## {NAME[setting]} — 한 군데만 다른 평가 쌍 {len(E):,}개 (cliff {cl.sum():,})')
    print(r.round(3).to_string())
    tables.append(r.assign(설정=setting))
pd.concat(tables).to_pickle('runs/tmp/ecfp/report3.pkl')
