# ECFP + MLP (MoleculeACE 설정) 쌍 모델 / 분자별 모델
# 사용: python ecfp_mlp.py <scope: target|shared> <kind: pair|mol> <cond: none|drop_leak> <seed>
#   target = 타깃마다 따로 학습 (MoleculeACE 그대로, 타깃별 설정) → 누수 없음
#   shared = 30개 타깃 한 모델 + 타깃 one-hot (우리 방식), 설정은 MoleculeACE에서 가장 많이 고른 값
import os, sys, glob, time, yaml, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
scope, kind, cond, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/ecfp'; os.chdir(CAP)
name = f'{scope}_{kind}_{cond}_s{seed}'
DEV = 'cuda'; torch.manual_seed(seed); np.random.seed(seed)
log = lambda *m: print(time.strftime('%H:%M:%S'), *m, flush=True)

# ── 데이터: 노트북과 같은 분할 (MoleculeACE 분할 + 학습 분자의 20%를 검증으로, SEED 0) + 누수 실험 규칙
T = sorted(os.path.basename(f)[:-4] for f in glob.glob('data/moleculeace/CHEMBL*.csv'))
def load_raw(t):
    df = pd.read_csv(f'data/moleculeace/{t}.csv').rename(columns={'y [pEC50/pKi]': 'p'})[['smiles', 'p', 'cliff_mol', 'split']]
    tr = df.index[df.split == 'train']; rng = np.random.default_rng(0)
    df.loc[rng.choice(tr, int(len(tr) * 0.2), replace=False), 'split'] = 'val'
    return df
RAW = {t: load_raw(t) for t in T}
if cond == 'v3':   # 팀원 방식: 분자 하나는 모든 타깃에서 같은 split (어디서든 평가면 평가), 검증은 분자 단위로 뽑음 (팀원 노트북 split_by_smiles와 같음)
    raw = pd.concat([pd.read_csv(f'data/moleculeace/{t}.csv', usecols=['smiles', 'split']) for t in T])
    sp_ = raw.groupby('smiles').split.agg(lambda x: 'test' if (x == 'test').any() else 'train')
    trs = sp_.index[sp_ == 'train']
    sp_[np.random.default_rng(0).choice(trs, int(len(trs) * 0.2), replace=False)] = 'val'
    for t in T:
        RAW[t]['split'] = RAW[t].smiles.map(sp_)
if cond == 'drop_leak':
    test_any = {t: set(d.smiles[d.split == 'test']) for t, d in RAW.items()}
    for t, d in RAW.items():
        others = set().union(*[test_any[u] for u in RAW if u != t])
        d.loc[[k for k in d.index if d.split[k] != 'test' and d.smiles[k] in others], 'split'] = 'drop'
# ECFP: MoleculeACE featurization과 같게 (반지름 2, 1024비트, 0/1)
FP = {t: torch.tensor(np.array([list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024)) for s in d.smiles], dtype=np.float32))
      for t, d in RAW.items()}
TI = {t: k for k, t in enumerate(T)}
CFG = {t: yaml.safe_load(open(f'{OUT}/../mace/mlp_{t}.yml')) for t in T}

# 쌍: 노트북과 같은 쌍 캐시 (유사 쌍 전부로 학습, 평가 쌍은 전부 저장해 나중에 고른다)
KEEP = {'train': 'train', 'val_easy': 'val', 'test_easy': 'test'}
def pairs_of(t):
    p = pd.read_pickle(f'data/pairs/{t}_' + ('v3' if cond == 'v3' else 'v2') + '.pkl'); p = p[p.split.isin(KEEP)].assign(split=lambda d: d.split.map(KEEP), target=t)
    sp = RAW[t].split
    return p[[sp[i] != 'drop' and sp[j] != 'drop' for i, j in zip(p.i, p.j)]].reset_index(drop=True)

class MLP(nn.Module):   # MoleculeACE MLPmodel: Linear → ReLU (→ dropout) × n_layers → Linear(1)
    def __init__(self, feat_in, n_layers, hidden, dropout=0.0):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(feat_in, hidden)] + [nn.Linear(hidden, hidden) for _ in range(n_layers - 1)])
        self.drop, self.out = dropout, nn.Linear(hidden, 1)
    def forward(self, x):
        for f in self.fc:
            x = F.dropout(F.relu(f(x)), self.drop, self.training)
        return self.out(x).squeeze(1)

def pair_x(fa, fb, tgt=None):   # [지문 A, 지문 B, 지문 B − 지문 A] (+ 타깃 one-hot)
    x = torch.cat([fa, fb, fb - fa], 1)
    return x if tgt is None else torch.cat([x, F.one_hot(tgt, len(T)).float()], 1)

def fit(make_batch, n_tr, val_fn, cfg, feat_in):
    """MoleculeACE NN.train: Adam, MSE, 묶음 32, 최대 500회차, 검증 손실이 10회차 연속 안 좋아지면 멈추고 최선으로 되돌림"""
    m = MLP(feat_in, int(cfg['n_layers']), int(cfg['hidden']), float(cfg['dropout'])).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=float(cfg['lr']))
    best, best_state, wait = float('inf'), None, 0
    for ep in range(int(os.environ.get('MAXEP', 500))):
        m.train()
        for idx in torch.randperm(n_tr).split(32):
            x, y = make_batch(idx)
            loss = F.mse_loss(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
        v = val_fn(m)
        if v <= best: best, wait, best_state = v, 0, {k: x.detach().clone() for k, x in m.state_dict().items()}
        else: wait += 1
        if wait >= 10: break
    m.load_state_dict(best_state); m.eval()
    return m, ep + 1

@torch.no_grad()
def pred_pairs(m, fa, fb, tgt=None):   # 양방향 평균 (Δ̂(A→B) − Δ̂(B→A)) / 2
    out = []
    for s in range(0, len(fa), 8192):
        a, b, t_ = fa[s:s + 8192].to(DEV), fb[s:s + 8192].to(DEV), None if tgt is None else tgt[s:s + 8192].to(DEV)
        out.append(((m(pair_x(a, b, t_)) - m(pair_x(b, a, t_))) / 2).cpu())
    return torch.cat(out).numpy()

res = []
RUN_T = T[:2] if os.environ.get('SMOKE') else T   # 확인 실행용
if scope == 'target':
    for t in RUN_T:
        cfg, fp, d = CFG[t], FP[t], RAW[t]
        if kind == 'pair':
            p = pairs_of(t); tr, va = p[p.split == 'train'], p[p.split == 'val']
            ia = torch.tensor(np.r_[tr.i.values, tr.j.values]); ib = torch.tensor(np.r_[tr.j.values, tr.i.values])   # 양방향
            y = torch.tensor(np.r_[tr.delta.values, -tr.delta.values], dtype=torch.float32)
            fpd = fp.to(DEV); ia, ib, y = ia.to(DEV), ib.to(DEV), y.to(DEV)
            vy = va.delta.values
            val_fn = lambda m: float(np.mean((pred_pairs(m, fp[va.i.values], fp[va.j.values]) - vy) ** 2))
            m, ep = fit(lambda idx: (pair_x(fpd[ia[idx]], fpd[ib[idx]]), y[idx]), len(y), val_fn, cfg, 3 * 1024)
            res.append(p[['target', 'i', 'j', 'split']].assign(pred=pred_pairs(m, fp[p.i.values], fp[p.j.values])))
        else:   # 분자별 (MoleculeACE 원래 방식): 학습 분자로 학습, 검증 분자로 멈춤 → 모든 분자 예측
            tr, va = (d.split == 'train').values, (d.split == 'val').values
            x, yv = fp.to(DEV), torch.tensor(d.p.values, dtype=torch.float32, device=DEV)
            itr = torch.tensor(np.where(tr)[0], device=DEV)
            val_fn = lambda m: float(F.mse_loss(m(x[va]), yv[va]))
            m, ep = fit(lambda idx: (x[itr[idx]], yv[itr[idx]]), int(tr.sum()), val_fn, cfg, 1024)
            with torch.no_grad(): pm = m(x).cpu().numpy()
            p = pairs_of(t)
            res.append(p[['target', 'i', 'j', 'split']].assign(pred=pm[p.j.values] - pm[p.i.values]))
        log(t, kind, f'{ep}회차')
else:   # 30개 타깃 한 모델 + 타깃 one-hot
    cfg = dict(hidden=1024, n_layers=2, lr=5e-6, dropout=0)   # MoleculeACE 타깃별 선택에서 가장 많은 값 (폭 30/30, 2층 10/30, 5e-6 19/30)
    P = pd.concat([pairs_of(t) for t in RUN_T], ignore_index=True)
    off = np.cumsum([0] + [len(FP[t]) for t in T]); ALLFP = torch.cat([FP[t] for t in T])
    gi = np.array([off[TI[t]] + i for t, i in zip(P.target, P.i)]); gj = np.array([off[TI[t]] + j for t, j in zip(P.target, P.j)])
    tg = torch.tensor([TI[t] for t in P.target])
    tr, va = (P.split == 'train').values, (P.split == 'val').values
    ia = torch.tensor(np.r_[gi[tr], gj[tr]], device=DEV); ib = torch.tensor(np.r_[gj[tr], gi[tr]], device=DEV)
    tt = torch.cat([tg[tr], tg[tr]]).to(DEV)
    y = torch.tensor(np.r_[P.delta.values[tr], -P.delta.values[tr]], dtype=torch.float32, device=DEV)
    fpd = ALLFP.to(DEV)
    vy = P.delta.values[va]
    val_fn = lambda m: float(np.mean((pred_pairs(m, ALLFP[gi[va]], ALLFP[gj[va]], tg[va]) - vy) ** 2))
    log('학습 쌍', len(y), '검증 쌍', int(va.sum()))
    m, ep = fit(lambda idx: (pair_x(fpd[ia[idx]], fpd[ib[idx]], tt[idx]), y[idx]), len(y), val_fn, cfg, 3 * 1024 + len(T))
    res.append(P[['target', 'i', 'j', 'split']].assign(pred=pred_pairs(m, ALLFP[gi], ALLFP[gj], tg)))
    log('공용 모델', f'{ep}회차')
pd.concat(res, ignore_index=True).to_pickle(f'{OUT}/pred_{name}' + ('_smoke' if os.environ.get('SMOKE') else '') + '.pkl')
log('끝', name)
