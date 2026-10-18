"""KSC 추가 실험 통계 — 논문 `stats/ksc_stats.py` 와 같은 양식으로 낸다.

  · cliff / non-cliff / 전체 RMSE (시드 3개 예측의 평균, 평가 쌍 38,711개)
  · 비교: cliff RMSE 차이 A − B, 95% CI(평가 분자 단위 군집 부트스트랩 2,000회),
    A 가 나은 타깃 수, 타깃 30개 Wilcoxon p

평가 분자 = 쌍 (i, j) 중 split 이 test 인 쪽 하나 (확인: 둘 다 test 인 쌍은 없다).

  python ksc_stats.py            # → ksc_results.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = "/home/dlrudgms1211/캡스톤 프로젝트/refs/external/capstone_pair-pipeline"
NBOOT, SEED = 2000, 0


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_all():
    ev = pd.read_pickle(f"{HERE}/eval_pairs.pkl")
    key = ["target", "i", "j"]
    sv = pd.read_pickle(f"{HERE}/svr_all.pkl")[key + ["svr", "svr_pair"]]
    ev = ev.merge(sv, on=key, how="left")
    for enc, col in (("gine_pair_nofrag", "pred_gnn_a0_s{}"), ("molebert_pair_nofrag", "pred_pt_a0_s{}")):
        pat = "pred_gine_nofrag_s{}.pkl" if enc.startswith("gine") else "pred_mb_nofrag_s{}.pkl"
        for s in (0, 1, 2):
            f = f"{HERE}/out_nb/{pat.format(s)}"
            if not os.path.exists(f):
                continue
            d = pd.read_pickle(f)
            d = d[(d.split == "test") & (d.reason == "pass")][key + [col.format(s)]]
            ev = ev.merge(d.rename(columns={col.format(s): f"{enc}_s{s}"}), on=key, how="left")
    # 평가 분자(군집) 번호
    molsplit = {}
    for t in ev.target.unique():
        molsplit[t] = pd.read_csv(f"{CAP}/data/moleculeace/{t}.csv")["split"].values
    ev["mol"] = [f"{t}:{j if molsplit[t][j] == 'test' else i}"
                 for t, i, j in zip(ev.target, ev.i, ev.j)]
    return ev


def pred(ev, name):
    cols = sorted(c for c in ev.columns if c.startswith(name + "_s"))
    if cols:
        return ev[cols].mean(1).values
    return ev[name].values


def summary(ev, name):
    p, cl = pred(ev, name), ev.is_cliff.values.astype(bool)
    y = ev.delta.values
    return dict(model=name, cliff=round(rmse(p[cl], y[cl]), 3),
                non_cliff=round(rmse(p[~cl], y[~cl]), 3), all=round(rmse(p, y), 3))


def per_target(ev, name, cliff_only=True):
    p, y = pred(ev, name), ev.delta.values
    m = ev.is_cliff.values.astype(bool) if cliff_only else np.ones(len(ev), bool)
    out = {}
    for t, idx in ev.groupby("target").indices.items():
        k = idx[m[idx]]
        if len(k):
            out[t] = rmse(p[k], y[k])
    return pd.Series(out)


def compare(ev, a, b, cliff_only=True):
    pa, pb, y = pred(ev, a), pred(ev, b), ev.delta.values
    m = ev.is_cliff.values.astype(bool) if cliff_only else np.ones(len(ev), bool)
    diff = rmse(pa[m], y[m]) - rmse(pb[m], y[m])
    # 평가 분자 단위 군집 부트스트랩
    groups = ev.mol.values
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_g[g] for g in pick])
        sel = sel[m[sel]]
        if len(sel) < 10:
            continue
        boots.append(rmse(pa[sel], y[sel]) - rmse(pb[sel], y[sel]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ta, tb = per_target(ev, a, cliff_only), per_target(ev, b, cliff_only)
    ts = ta.index.intersection(tb.index)
    w = wilcoxon(ta[ts], tb[ts])
    return dict(a=a, b=b, metric="cliff" if cliff_only else "all",
                diff=round(diff, 3), ci=[round(float(lo), 3), round(float(hi), 3)],
                a_better=f"{int((ta[ts] < tb[ts]).sum())}/{len(ts)}", p=float(w.pvalue))


if __name__ == "__main__":
    ev = load_all()
    have = [m for m in ["svr", "gine_mol", "molebert_mol", "ecfp_mlp_pair", "svr_pair",
                        "gine_pair", "molebert_pair", "gine_pair_nofrag", "molebert_pair_nofrag"]
            if m in ev.columns or any(c.startswith(m + "_s") for c in ev.columns)]
    tab = pd.DataFrame([summary(ev, m) for m in have])
    print(tab.to_string(index=False))

    pairs = [("svr_pair", "svr"), ("molebert_pair", "svr_pair"), ("molebert_pair", "svr"),
             ("molebert_pair_nofrag", "molebert_pair"), ("gine_pair_nofrag", "gine_pair")]
    cmps = []
    for a, b in pairs:
        if a not in have or b not in have:
            continue
        for co in (True, False):
            cmps.append(compare(ev, a, b, co))
    print()
    for c in cmps:
        print(f"{c['a']:22s} vs {c['b']:18s} [{c['metric']:5s}] 차이 {c['diff']:+.3f} "
              f"CI {c['ci']} 나은 타깃 {c['a_better']:6s} p={c['p']:.2g}")
    json.dump(dict(table=tab.to_dict("records"), compare=cmps,
                   n_pairs=len(ev), n_cliff=int(ev.is_cliff.sum())),
              open(f"{HERE}/ksc_results.json", "w"), ensure_ascii=False, indent=1)
