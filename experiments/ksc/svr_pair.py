"""실험 B — 쌍 입력 ECFP + SVR (`서버_인계.md` 2절 그대로).

타깃마다 따로 학습한다. 지문은 ECFP4 2,048비트(반지름 2)로 노트북 `FPGEN` 과 같고,
쌍 커널은 K((A,B),(C,D)) = T(A,C)·T(B,D) (T = Tanimoto), 예측은 Δ̂ = [g(A,B) − g(B,A)]/2,
C 는 {1,10,100,1000} 중 검증 쌍 RMSE 가 가장 낮은 값, ε = 0.1.

같은 스크립트에서 **분자별 ECFP + SVR** 도 다시 계산한다 (노트북 `svr_predict` 와 같은 절차).
이것은 비교 기준이자 게이트다 — 논문 표 1 의 0.914 / 0.455 / 0.591 이 재현되어야 한다.

  python svr_pair.py <타깃>            # 결과 → out/svr_<타깃>.pkl
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.svm import SVR

RDLogger.DisableLog("rdApp.*")
CAP = "/home/dlrudgms1211/캡스톤 프로젝트/refs/external/capstone_pair-pipeline"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out"
SVR_C = [1, 10, 100, 1000]
CAP_TRAIN = 20000                  # 커널 크기 상한 (인계 문서 허용: 넘으면 시드 0으로 무작위 추출)
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
KEEP = {"train": "train", "val_easy": "val", "test_easy": "test"}
ALL = sorted(f[:-4] for f in os.listdir(f"{CAP}/data/moleculeace") if f.startswith("CHEMBL"))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def raw_splits():
    """노트북·ecfp_mlp.py 와 같은 분할 + drop_leak 규칙."""
    R = {}
    for t in ALL:
        df = pd.read_csv(f"{CAP}/data/moleculeace/{t}.csv").rename(
            columns={"y [pEC50/pKi]": "p"})[["smiles", "p", "cliff_mol", "split"]]
        tr = df.index[df.split == "train"]
        df.loc[np.random.default_rng(0).choice(tr, int(len(tr) * 0.2), replace=False), "split"] = "val"
        R[t] = df
    test_any = {t: set(d.smiles[d.split == "test"]) for t, d in R.items()}
    for t, d in R.items():
        others = set().union(*[test_any[u] for u in R if u != t])
        d.loc[[k for k in d.index if d.split[k] != "test" and d.smiles[k] in others], "split"] = "drop"
    return R


def tanimoto(X, Y):
    inter = X @ Y.T
    return inter / (X.sum(1)[:, None] + Y.sum(1)[None, :] - inter)


def main(target):
    os.makedirs(OUT, exist_ok=True)
    df = raw_splits()[target]
    X = np.array([FPGEN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in df.smiles], np.float32)
    T = tanimoto(X, X).astype(np.float64)

    # ── 분자별 SVR (게이트)
    tr, va = (df.split == "train").values, (df.split == "val").values
    fit = lambda C: SVR(kernel="precomputed", C=C).fit(T[np.ix_(tr, tr)], df.p[tr])
    vsc = {C: rmse(fit(C).predict(T[np.ix_(va, tr)]), df.p[va]) for C in SVR_C}
    C_mol = min(vsc, key=vsc.get)
    mol_pred = fit(C_mol).predict(T[:, tr])

    # ── 쌍 SVR
    p = pd.read_pickle(f"{CAP}/data/pairs/{target}_v2.pkl")
    p = p[p.split.isin(KEEP)].assign(split=lambda d: d.split.map(KEEP))
    sp = df.split
    p = p[[sp[i] != "drop" and sp[j] != "drop" for i, j in zip(p.i, p.j)]].reset_index(drop=True)
    ev = pd.read_pickle(f"{HERE}/eval_pairs.pkl")
    ev = ev[ev.target == target][["target", "i", "j", "delta", "is_cliff"]].reset_index(drop=True)

    trp, vap = p[p.split == "train"], p[p.split == "val"]
    sub = len(trp) > CAP_TRAIN
    if sub:
        trp = trp.iloc[np.random.default_rng(0).choice(len(trp), CAP_TRAIN, replace=False)]
    a, b, y = trp.i.values, trp.j.values, trp.delta.values
    K = T[np.ix_(a, a)] * T[np.ix_(b, b)]

    def anti(m, qi, qj):                      # Δ̂ = [g(A,B) − g(B,A)] / 2
        g1 = m.predict(T[np.ix_(qi, a)] * T[np.ix_(qj, b)])
        g2 = m.predict(T[np.ix_(qj, a)] * T[np.ix_(qi, b)])
        return (g1 - g2) / 2

    models, vp = {}, {}
    for C in SVR_C:
        models[C] = SVR(kernel="precomputed", C=C).fit(K, y)
        vp[C] = rmse(anti(models[C], vap.i.values, vap.j.values), vap.delta.values)
    C_pair = min(vp, key=vp.get)

    ev["svr"] = mol_pred[ev.j.values] - mol_pred[ev.i.values]
    ev["svr_pair"] = anti(models[C_pair], ev.i.values, ev.j.values)
    ev.attrs = dict(C_mol=C_mol, C_pair=C_pair, val_mol=vsc, val_pair=vp,
                    n_train_pair=len(trp), subsampled=bool(sub))
    ev.to_pickle(f"{OUT}/svr_{target}.pkl")
    cl = ev.is_cliff.values.astype(bool)
    print(f"{target:16s} 학습쌍 {len(trp):6d}{'(추출)' if sub else '     '} C분자 {C_mol:4d} C쌍 {C_pair:4d} | "
          f"분자별 cliff {rmse(ev.svr[cl], ev.delta[cl]):.3f} 전체 {rmse(ev.svr, ev.delta):.3f} | "
          f"쌍 cliff {rmse(ev.svr_pair[cl], ev.delta[cl]):.3f} 전체 {rmse(ev.svr_pair, ev.delta):.3f}",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
