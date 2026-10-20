# 최종 설계 (계획_고정_2026-10-18.md, 10-18 21:20 변경): 30개 타깃 한 모델 · 누수 분자 빼기 · GINE 폭 300 · 조각 인코더 없음
# · 합 읽기 · 헤드 2048 → 1024 → 1 · 카드 메모리 50% 상한. 쌍 모델은 한 군데만 다른 검증 쌍으로, 분자별은 검증 분자 활성으로 멈춘다
# 작업: 쌍 [Mole-BERT, GINE 300] × 시드 0–4 (작업 10개) + 분자별 GINE 300·Mole-BERT 시드 0–4 (작업 1개)
# 사용: python make_final.py [smoke]  → jobs/*.ipynb, jobs.txt
import json, os, copy, shutil, sys
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/final'
nb = json.load(open(f'{CAP}/ac_pipeline.ipynb'))
src = lambda c: ''.join(c['source'])
IDS = [c.get('id') for c in nb['cells']]
DROP = {'6a4888c8', '7660d566', 'b31556f2', '94d9465c', 'e1f1af72', 'svr-flow', 'edit-dist', 'model-architecture', 'gine-encoder', '697548aa'}   # 그림·예시 셀
BASE = [i for i in IDS[:IDS.index('3c8f160f') + 1] if i not in DROP]

def build(name, kind, seed=None, smoke=False):
    keep = BASE + {'gine': [], 'mb': ['molebert-train'], 'mol': ['gnnmol-train', 'molebert-train', 'molebert-mol']}[kind]
    n = copy.deepcopy(nb); n['cells'] = [c for c in n['cells'] if c.get('id') in keep]
    def rep(cid, a, b):
        c = [c for c in n['cells'] if c.get('id') == cid][0]; s = src(c); assert s.count(a) == 1, (name, cid, a[:60]); c['source'] = s.replace(a, b)
    rep('8a15e3a3', "DATA, PAIRS = 'data/moleculeace', 'data/pairs'", f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
    assert "LEAK_MODE = 'drop_leak'" in src([c for c in n['cells'] if c.get('id') == '8a15e3a3'][0])
    assert "set_per_process_memory_fraction(0.5)" in src([c for c in n['cells'] if c.get('id') == '18184d19'][0])
    if smoke: rep('8a15e3a3', "USE_ALL_TARGETS = True", "USE_ALL_TARGETS = False")
    rep('18184d19', "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, 200", f"HIDDEN, LAYERS, BATCH, LR, EPOCHS = 300, 3, 256, 1e-3, {2 if smoke else 200}")
    rep('18184d19', "USE_FRAG, USE_EDIT = True, True", "USE_FRAG, USE_EDIT = False, True")
    rep('18184d19', "HEAD = 'small'\n", "HEAD = 'big2048'\n")
    rep('18184d19', "READOUT = 'mean'\n", "READOUT = 'sum'\n")
    rep('18184d19', "VAL_PAIRS = 'all'\n", "VAL_PAIRS = 'pass'\n")
    rep('18184d19', "RUNS = 'runs'", f"RUNS = '{OUT}/smoke_runs'" if smoke else f"RUNS = '{CAP}/runs'")
    if 'molebert-train' in keep:
        rep('molebert-train', "PT_PATH = f'{RUNS}/pretrained/Mole-BERT.pth'", f"PT_PATH = '{CAP}/runs/pretrained/Mole-BERT.pth'")
    cols = "KEY + ['split', 'reason', 'delta', 'is_cliff']"
    if kind == 'gine':
        rep('3c8f160f', "        for s in SEEDS:\n            m = train(s, B, tag)", f"        for s in [{seed}]:   # 최종: 시드 {seed}\n            m = train(s, B, tag)")
        rep('3c8f160f', "        save_cache(pred_cache('preds', tag, files), pset[KEY + cols])",
            f"        pset[{cols} + ['pred_gnn_a0_s{seed}']].to_pickle('{OUT}/pred_{name}.pkl'); print('태그', tag)")
    else:
        rep('3c8f160f', "    TAGS[setting] = tag\n", "    TAGS[setting] = tag\n    continue   # 이 작업에서는 GINE 쌍 모델을 학습하지 않는다\n")
    if kind == 'mb':
        rep('molebert-train', "        for s in SEEDS:\n            m = train(s, B, tag, make_model=PTPairModel)",
            f"        for s in [{seed}]:   # 최종: 시드 {seed}\n            m = train(s, B, tag, make_model=PTPairModel)")
        rep('molebert-train', "        save_cache(pred_cache('preds_pt', tag, files), pset[KEY + cols])",
            f"        pset[{cols} + ['pred_pt_a0_s{seed}']].to_pickle('{OUT}/pred_{name}.pkl'); print('태그', tag)")
    if kind == 'mol':
        rep('8a15e3a3', "SEEDS = [0, 1, 2]", "SEEDS = [0, 1, 2, 3, 4]")
        rep('molebert-train', "    PT_TAGS[setting] = tag\n", "    PT_TAGS[setting] = tag\n    continue   # 분자별 모델만 학습\n")
        if smoke: rep('gnnmol-train', "MOL_EPOCHS = 1000", "MOL_EPOCHS = 2")
    for c in n['cells']:
        if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
    os.makedirs(f'{OUT}/jobs', exist_ok=True)
    json.dump(n, open(f'{OUT}/jobs/{name}.ipynb', 'w'), ensure_ascii=False)
    return name

if sys.argv[1:] == ['smoke']:
    shutil.rmtree(f'{OUT}/smoke_runs', ignore_errors=True); os.makedirs(f'{OUT}/smoke_runs')
    print(' '.join([build('smoke_mb', 'mb', 0, True), build('smoke_gine', 'gine', 0, True), build('smoke_mol', 'mol', smoke=True)]))
else:
    jobs = [build('final_mol', 'mol')] + [build(f'final_{k}_s{s}', k, s) for k in ['mb', 'gine'] for s in range(5)]
    open(f'{OUT}/jobs.txt', 'w').write('\n'.join(jobs) + '\n'); print(len(jobs), 'jobs')
