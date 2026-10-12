# C 실험: 유사 쌍 전부로 학습 · GINE · 시드 3 × [편집 정보 전부 빼기, 바뀐 원자 수 칸만 빼기]
import json, nbformat, os, copy, shutil
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/ablC'
os.makedirs(OUT, exist_ok=True)
P = f'{CAP}/ac_pipeline.ipynb'; nb = json.load(open(P))
src = lambda c: ''.join(c['source'])
def cell(n, cid): return [c for c in n['cells'] if c.get('id') == cid][0]
def rep(n, cid, a, b):
    c = cell(n, cid); s = src(c); assert s.count(a) == 1, (cid, a[:60]); c['source'] = s.replace(a, b).splitlines(keepends=True)

# 원본 노트북에 옵션 추가 (기본값 끔 → 기존 가중치·태그 그대로)
if 'DROP_NCHANGED' not in src(cell(nb, '0362b826')):
    rep(nb, '0362b826', "MASK_CAP = None\n",
        "MASK_CAP = None\n# 지름길 점검 (ablation): True면 '바뀐 원자 수' 칸을 0으로 둔다 (= 그 칸을 빼는 것과 같다). 가중치 이름에 _nonchanged\nDROP_NCHANGED = False\n")
    rep(nb, '0362b826', "min(r.n_changed / MAX_CHANGED, 3.0), float(",
        "0.0 if DROP_NCHANGED else min(r.n_changed / MAX_CHANGED, 3.0), float(")
    rep(nb, '3c8f160f', "    tag += ('' if USE_FRAG else '_nofrag') + ('' if USE_EDIT else '_noedit')   # ablation 조건",
        "    tag += ('' if USE_FRAG else '_nofrag') + ('' if USE_EDIT else '_noedit') + ('_nonchanged' if DROP_NCHANGED else '')   # ablation 조건")
    json.dump(nb, open(P, 'w'), ensure_ascii=False, indent=1); open(P, 'a').write('\n')
    nbformat.validate(nbformat.read(P, 4))

cut = [i for i, c in enumerate(nb['cells']) if c.get('id') == '3c8f160f'][0]
def build(name, cond, seed, smoke=False):
    n = copy.deepcopy(nb); n['cells'] = n['cells'][:cut + 1]
    rep(n, '8a15e3a3', "DATA, PAIRS = 'data/moleculeace', 'data/pairs'", f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
    rep(n, '18184d19', "RUNS = 'runs'", f"RUNS = '{OUT}/smoke_runs'" if smoke else f"RUNS = '{CAP}/runs'")
    if cond == 'noedit':
        rep(n, '18184d19', "USE_FRAG, USE_EDIT = True, True", "USE_FRAG, USE_EDIT = True, False")
    else:
        rep(n, '0362b826', "DROP_NCHANGED = False", "DROP_NCHANGED = True")
    if smoke:
        rep(n, '8a15e3a3', "USE_ALL_TARGETS = True", "USE_ALL_TARGETS = False")
        rep(n, '18184d19', "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, 200", "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, 2")
    rep(n, '3c8f160f', "        for s in SEEDS:\n            m = train(s, B, tag)", f"        for s in [{seed}]:   # 나눠 학습: {cond} · 시드 {seed}\n            m = train(s, B, tag)")
    rep(n, '3c8f160f', "        save_cache(pred_cache('preds', tag, files), pset[KEY + cols])",
        f"        pset[KEY + ['split', 'reason', 'delta', 'is_cliff', 'pred_gnn_a0_s{seed}']].to_pickle('{OUT}/pred_{name}.pkl')\n"
        f"        print('태그', tag)")
    for c in n['cells']:
        if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
    json.dump(n, open(f'{OUT}/{name}.ipynb', 'w'), ensure_ascii=False)

def cmd(name, gpu):
    return (f"~/micromamba/bin/tmux new-session -d -s {name} \"cd {OUT} && export TMPDIR={CAP}/runs/tmp MPLCONFIGDIR={CAP}/runs/tmp && "
            f"CUDA_VISIBLE_DEVICES={gpu} /home/bising0922/micromamba/envs/aienv/bin/jupyter nbconvert --to notebook --execute "
            f"--ExecutePreprocessor.timeout=-1 {name}.ipynb --output {name}_out.ipynb > {name}.log 2>&1; "
            f"echo \\\"exit \\$? \\$(date +%m-%d_%H:%M)\\\" >> {name}.log\"")
shutil.rmtree(f'{OUT}/smoke_runs', ignore_errors=True); os.makedirs(f'{OUT}/smoke_runs')
build('smoke_noedit', 'noedit', 0, smoke=True); build('smoke_nonchanged', 'nonchanged', 0, smoke=True)
jobs = [(f'ablC_{c}_s{s}', c, s) for c in ['noedit', 'nonchanged'] for s in [0, 1, 2]]
gpus = [5, 5, 1, 0, 3, 1]
for name, c, s in jobs:
    build(name, c, s)
open(f'{OUT}/launch.sh', 'w').write('\n'.join(cmd(nm, g) for (nm, _, _), g in zip(jobs, gpus)) + '\n')
print('ok', [j[0] for j in jobs])
