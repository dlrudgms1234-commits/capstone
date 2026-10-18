"""KSC 추가 실험용 평가 쌍 표를 서버 캐시에서 다시 만든다.

맥에 있는 `_ksc_share/paper/eval_pairs_predictions_leakfree.csv` 가 서버에 없어서, 같은 표를
캐시(`runs/tmp/head`, `runs/tmp/ecfp`, `runs/preds_*`)에서 복원한다.
복원이 맞는지는 **논문 표 1 숫자(0.914/0.455/0.591 …)를 그대로 재현하는지**로 확인한다.

  python build_eval.py            # eval_pairs.pkl 생성 + 표 1 재현 확인
"""
import glob
import os

import numpy as np
import pandas as pd

CAP = "/home/dlrudgms1211/캡스톤 프로젝트/refs/external/capstone_pair-pipeline"
OUT = os.path.dirname(os.path.abspath(__file__))
KEY = ["target", "i", "j"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_pairs():
    """쌍 신경망 예측 6개(GINE·Mole-BERT × 시드 3)에서 평가 쌍 뼈대를 만든다."""
    base = None
    for enc, col, pat in (("gine_pair", "pred_gnn_a0_s{}", "head/pred_head_gine_s{}.pkl"),
                          ("molebert_pair", "pred_pt_a0_s{}", "head/pred_head_mb_s{}.pkl")):
        for s in (0, 1, 2):
            d = pd.read_pickle(f"{CAP}/runs/tmp/{pat.format(s)}")
            d = d[(d.split == "test") & (d.reason == "pass")]
            keep = KEY + ["delta", "is_cliff"] if base is None else KEY
            d = d[keep + [col.format(s)]].rename(columns={col.format(s): f"{enc}_s{s}"})
            base = d if base is None else base.merge(d, on=KEY, how="inner")
    return base.reset_index(drop=True)


def add_ecfp_mlp(ev):
    for s in (0, 1, 2):
        d = pd.read_pickle(f"{CAP}/runs/tmp/ecfp/pred_shared_pair_drop_leak_s{s}.pkl")
        d = d[d.split == "test"][KEY + ["pred"]].rename(columns={"pred": f"ecfp_mlp_pair_s{s}"})
        ev = ev.merge(d, on=KEY, how="left")
    return ev


def add_mol(ev):
    """분자별 모델(GINE·Mole-BERT)의 분자 예측 → 쌍 Δ̂ = p(j) − p(i)."""
    for name, pat in (("gine_mol", "preds_mol_t30_*_leakdrop_head1024_*.pkl"),
                      ("molebert_mol", "preds_molpt_t30_*_leakdrop_head1024_molebert_*.pkl")):
        f = sorted(glob.glob(f"{CAP}/runs/{pat}"))
        assert len(f) == 1, (pat, f)
        d = pd.read_pickle(f[0]).set_index(["target", "k"])
        for s in (0, 1, 2):
            v = d[f"s{s}"]
            ev[f"{name}_s{s}"] = (v.loc[list(zip(ev.target, ev.j))].values
                                  - v.loc[list(zip(ev.target, ev.i))].values)
    return ev


def ens(ev, name):
    cols = [c for c in ev.columns if c.startswith(name + "_s")]
    return ev[cols].mean(1).values if cols else ev[name].values


def table(ev, names):
    rows = []
    cl = ev.is_cliff.values.astype(bool)
    for n in names:
        p = ens(ev, n)
        rows.append(dict(model=n, cliff=rmse(p[cl], ev.delta.values[cl]),
                         non_cliff=rmse(p[~cl], ev.delta.values[~cl]),
                         all=rmse(p, ev.delta.values)))
    return pd.DataFrame(rows).round(3)


if __name__ == "__main__":
    ev = add_mol(add_ecfp_mlp(load_pairs()))
    print(f"평가 쌍 {len(ev)}개 (cliff {int(ev.is_cliff.sum())}개) · 타깃 {ev.target.nunique()}개")
    ev.to_pickle(f"{OUT}/eval_pairs.pkl")
    print(table(ev, ["gine_mol", "molebert_mol", "ecfp_mlp_pair", "gine_pair", "molebert_pair"])
          .to_string(index=False))
    print("\n논문 표 1 (목표): GINE 분자별 1.157/0.514/0.713 · Mole-BERT 분자별 1.158/0.549/0.733")
    print("                   ECFP+MLP 쌍 0.898/0.514/0.622 · GINE 쌍 0.914/0.469/0.600 · "
          "Mole-BERT 쌍 0.889/0.471/0.592")
