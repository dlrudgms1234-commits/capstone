# 누수 실험 분할 실행 노트북: 조건(drop_random, drop_leak) × [GINE 시드 3, Mole-BERT 시드 3, 분자별(GNN·Mole-BERT) 1]
import json, os, copy, shutil, sys
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/leak'
os.makedirs(OUT, exist_ok=True)
nb = json.load(open(f'{CAP}/ac_pipeline.ipynb'))
src = lambda c: ''.join(c['source'])
IDS = [c.get('id') for c in nb['cells']]
upto = lambda cid: IDS[:IDS.index(cid) + 1]

def build(name, mode, kind, seed=None, smoke=False):
    keep = upto('3c8f160f')
    if kind == 'mb':
        keep += ['molebert-md', 'molebert-train']
    if kind == 'mol':
        keep += ['gnnmol-md', 'gnnmol-train', 'molebert-md', 'molebert-train', 'molebert-mol-md', 'molebert-mol']
    if kind == 'full':
        keep = IDS
    n = copy.deepcopy(nb); n['cells'] = [c for c in n['cells'] if c.get('id') in keep]
    def rep(cid, a, b):
        c = [c for c in n['cells'] if c.get('id') == cid][0]; s = src(c); assert s.count(a) == 1, (name, cid, a[:60]); c['source'] = s.replace(a, b)
    rep('8a15e3a3', "DATA, PAIRS = 'data/moleculeace', 'data/pairs'", f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
    rep('8a15e3a3', "LEAK_MODE = None", f"LEAK_MODE = '{mode}'")
    rep('18184d19', "RUNS = 'runs'", f"RUNS = '{OUT}/smoke_runs'" if smoke else f"RUNS = '{CAP}/runs'")
    if kind == 'full':
        rep('ablC-result', "f'runs/tmp/ablC/", f"f'{CAP}/runs/tmp/ablC/")
    if smoke:
        rep('8a15e3a3', "USE_ALL_TARGETS = True", "USE_ALL_TARGETS = False")
        rep('18184d19', "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, 200", "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, 2")
        if kind == 'mol': rep('gnnmol-train', "MOL_EPOCHS = 1000", "MOL_EPOCHS = 2")
    if kind == 'gine':
        rep('3c8f160f', "        for s in SEEDS:\n            m = train(s, B, tag)", f"        for s in [{seed}]:   # 나눠 학습: GINE · 시드 {seed}\n            m = train(s, B, tag)")
        rep('3c8f160f', "        save_cache(pred_cache('preds', tag, files), pset[KEY + cols])", "        print('태그', tag)   # 나눠 학습: 예측값 저장은 전체 실행에서")
    if kind in ('mb', 'mol'):   # GINE 쌍 모델은 건너뛴다 (태그만 만든다)
        rep('3c8f160f', "    TAGS[setting] = tag\n", "    TAGS[setting] = tag\n    continue   # 나눠 학습: GINE 쌍 모델은 다른 작업에서\n")
    if kind == 'mb':
        rep('molebert-train', "        for s in SEEDS:\n            m = train(s, B, tag, make_model=PTPairModel)",
            f"        for s in [{seed}]:   # 나눠 학습: Mole-BERT · 시드 {seed}\n            m = train(s, B, tag, make_model=PTPairModel)")
        rep('molebert-train', "        save_cache(pred_cache('preds_pt', tag, files), pset[KEY + cols])", "        print('태그', tag)")
    if kind == 'mol':
        rep('molebert-train', "    PT_TAGS[setting] = tag\n", "    PT_TAGS[setting] = tag\n    continue   # 나눠 학습: Mole-BERT 쌍 모델은 다른 작업에서\n")
    for c in n['cells']:
        if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
    json.dump(n, open(f'{OUT}/{name}.ipynb', 'w'), ensure_ascii=False)

MODES = {'rnd': 'drop_random', 'leak': 'drop_leak'}
if sys.argv[1:] == ['smoke']:
    shutil.rmtree(f'{OUT}/smoke_runs', ignore_errors=True); os.makedirs(f'{OUT}/smoke_runs')
    for kind in ['gine', 'mb', 'mol']:
        build(f'smoke_{kind}', 'drop_leak', kind, 0, smoke=True)
else:
    jobs = []
    for kind in ['mb', 'mol', 'gine']:   # 오래 걸리는 것부터
        for short, mode in MODES.items():
            for s in ([0, 1, 2] if kind != 'mol' else [None]):
                name = f'{short}_{kind}' + (f'_s{s}' if s is not None else '')
                build(name, mode, kind, s); jobs.append(name)
    for short, mode in MODES.items():
        build(f'{short}_full', mode, 'full')
    open(f'{OUT}/jobs.txt', 'w').write('\n'.join(jobs) + '\n')
    print('ok', jobs)
