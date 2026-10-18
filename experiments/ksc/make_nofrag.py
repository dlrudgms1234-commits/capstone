"""실험 A — 변화 부위 인코더(f)를 뺀 쌍 입력 모델 노트북을 시드별로 만든다.
(`서버_인계.md` 1절. `experiments/make_ablC.py` 를 본떴고, 인계 표대로 설정을 맞췄다.)

  타깃 30개 · LEAK_MODE='drop_leak' · HEAD='big' · SEEDS=[0,1,2](시드마다 노트북 하나)
  USE_FRAG, USE_EDIT = False, True  → 태그에 `_nofrag` 가 붙어 기존 가중치와 섞이지 않는다

한 노트북이 한 시드의 **GINE → Mole-BERT** 를 차례로 학습한다 (`train()` 이 가중치 파일이
있으면 불러오므로 GINE 은 한 번만 학습된다). Mole-BERT 는 `PTPairModel` 이 `use_frag=True` 로
박혀 있어 인계 문서가 준 코드로 바꾼다.

  python make_nofrag.py          # out_nb/*.ipynb 와 launch.sh 생성
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = "/home/dlrudgms1211/캡스톤 프로젝트/refs/external/capstone_pair-pipeline"
NB = "/home/dlrudgms1211/캡스톤 프로젝트/동료/capstone/ac_pipeline.ipynb"
OUT = f"{HERE}/out_nb"
PY = "/home/dlrudgms1211/micromamba/envs/capstone-ml/bin"
GPUS = [0, 1, 2]                      # 5번 금지

PT_NEW = '''class PTPairModel(PairModel):
    """쌍 모델에서 두 인코더만 Mole-BERT로 바꾼다. 이어 붙이기·헤드·손실은 그대로"""
    def __init__(self):
        nn.Module.__init__(self)
        self.use_frag, self.use_edit = USE_FRAG, USE_EDIT          # ablation 옵션을 GINE과 같이 따른다
        self.mol = MoleBERT()
        self.frag = MoleBERT() if self.use_frag else None
        n_in = (5 if self.use_frag else 3) * PT_DIM + (N_EDIT if self.use_edit else 0) + len(DEV_TARGETS)
        self.head = make_head(n_in)

    def param_groups(self):
        enc = list(self.mol.parameters()) + (list(self.frag.parameters()) if self.frag is not None else [])
        return [{'params': enc, 'lr': PT_LR}, {'params': self.head.parameters(), 'lr': LR}]
'''


def main():
    os.makedirs(OUT, exist_ok=True)
    nb = json.load(open(NB))
    src = lambda c: "".join(c["source"])
    cell = lambda n, cid: [c for c in n["cells"] if c.get("id") == cid][0]

    def rep(n, cid, a, b, count=1):
        c = cell(n, cid)
        s = src(c)
        assert s.count(a) == count, (cid, a[:60], s.count(a))
        c["source"] = s.replace(a, b).splitlines(keepends=True)

    cut = [i for i, c in enumerate(nb["cells"]) if c.get("id") == "molebert-train"][0]

    for seed in (0, 1, 2):
        n = copy.deepcopy(nb)
        n["cells"] = n["cells"][:cut + 1]
        # 쓰지 않는 셀은 뺀다. concat-example 은 _m.frag 를 부르므로 USE_FRAG=False 에서 죽고,
        # gnnmol-* 은 분자별 GNN 비교군이라 이 ablation 에 필요 없다 (GPU 시간만 든다).
        n["cells"] = [c for c in n["cells"]
                      if c.get("id") not in ("concat-example", "gnnmol-md", "gnnmol-train", "5acb601f")]
        # 경로
        rep(n, "8a15e3a3", "DATA, PAIRS = 'data/moleculeace', 'data/pairs'",
            f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
        rep(n, "18184d19", "RUNS = 'runs'", f"RUNS = '{HERE}/runs'")
        # 논문 설정 (인계 문서 표)
        rep(n, "8a15e3a3", "LEAK_MODE = None", "LEAK_MODE = 'drop_leak'")
        rep(n, "18184d19", "HEAD = 'small'", "HEAD = 'big'")
        # ablation 조건
        rep(n, "18184d19", "USE_FRAG, USE_EDIT = True, True", "USE_FRAG, USE_EDIT = False, True")
        # Mole-BERT 도 USE_FRAG 를 따르게 (인계 문서가 준 코드)
        old = src(cell(n, "molebert-train"))
        a = old[old.index("class PTPairModel"):old.index("PT_GRAPHS = ")]
        rep(n, "molebert-train", a, PT_NEW + "\n")
        # 시드 하나만, 예측은 따로 저장
        rep(n, "3c8f160f", "        for s in SEEDS:\n            m = train(s, B, tag)",
            f"        for s in [{seed}]:   # 나눠 학습: nofrag · 시드 {seed}\n            m = train(s, B, tag)")
        rep(n, "3c8f160f",
            "        save_cache(pred_cache('preds', tag, files), pset[KEY + cols])",
            f"        pset[KEY + ['split', 'reason', 'delta', 'is_cliff', 'pred_gnn_a0_s{seed}']]"
            f".to_pickle('{OUT}/pred_gine_nofrag_s{seed}.pkl')\n        print('GINE 태그', tag)")
        rep(n, "molebert-train", "        for s in SEEDS:\n            m = train(s, B, tag, make_model=PTPairModel)",
            f"        for s in [{seed}]:\n            m = train(s, B, tag, make_model=PTPairModel)")
        rep(n, "molebert-train",
            "        save_cache(pred_cache('preds_pt', tag, files), pset[KEY + cols])",
            f"        pset[KEY + ['split', 'reason', 'delta', 'is_cliff', 'pred_pt_a0_s{seed}']]"
            f".to_pickle('{OUT}/pred_mb_nofrag_s{seed}.pkl')\n        print('Mole-BERT 태그', tag)")
        for c in n["cells"]:
            if c["cell_type"] == "code":
                c["outputs"], c["execution_count"] = [], None
        json.dump(n, open(f"{OUT}/nofrag_s{seed}.ipynb", "w"), ensure_ascii=False)

    os.makedirs(f"{HERE}/runs", exist_ok=True)
    if not os.path.exists(f"{HERE}/runs/pretrained"):
        os.symlink(f"{CAP}/runs/pretrained", f"{HERE}/runs/pretrained")
    lines = [f"cd {OUT} && export TMPDIR={HERE}/tmp MPLCONFIGDIR={HERE}/tmp && mkdir -p {HERE}/tmp"]
    for seed, g in zip((0, 1, 2), GPUS):
        lines.append(f"CUDA_VISIBLE_DEVICES={g} setsid nohup {PY}/jupyter nbconvert --to notebook --execute "
                     f"--ExecutePreprocessor.timeout=-1 nofrag_s{seed}.ipynb "
                     f"--output nofrag_s{seed}_out.ipynb > nofrag_s{seed}.log 2>&1 &")
    open(f"{OUT}/launch.sh", "w").write("\n".join(lines) + "\nwait\n")
    print("만들었다:", OUT)


if __name__ == "__main__":
    main()
