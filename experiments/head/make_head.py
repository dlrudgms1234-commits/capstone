# 헤드 맞추기 실험: 누수 없음·우리 방식(drop_leak) · HEAD big · [GINE 쌍, Mole-BERT 쌍] × 시드 3
import json, os, copy, shutil, sys
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/head'
nb = json.load(open(f'{CAP}/ac_pipeline.ipynb'))
src = lambda c: ''.join(c['source'])
IDS = [c.get('id') for c in nb['cells']]

def build(name, kind, seed, smoke=False):
    keep = IDS[:IDS.index('3c8f160f') + 1] + (['molebert-md', 'molebert-train'] if kind == 'mb' else [])
    n = copy.deepcopy(nb); n['cells'] = [c for c in n['cells'] if c.get('id') in keep]
    def rep(cid, a, b):
        c = [c for c in n['cells'] if c.get('id') == cid][0]; s = src(c); assert s.count(a) == 1, (name, cid, a[:60]); c['source'] = s.replace(a, b)
    rep('8a15e3a3', "DATA, PAIRS = 'data/moleculeace', 'data/pairs'", f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
    rep('8a15e3a3', "LEAK_MODE = 'drop_leak'   # 논문 설정", "LEAK_MODE = 'drop_leak'")
    rep('18184d19', "HEAD = 'small'", "HEAD = 'big'")
    rep('18184d19', "RUNS = 'runs'", f"RUNS = '{OUT}/smoke_runs'" if smoke else f"RUNS = '{CAP}/runs'")
    if smoke:
        rep('8a15e3a3', "USE_ALL_TARGETS = True", "USE_ALL_TARGETS = False")
        rep('18184d19', "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, 200", "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, 2")
    keep_cols = "KEY + ['split', 'reason', 'delta', 'is_cliff']"
    if kind == 'gine':
        rep('3c8f160f', "        for s in SEEDS:\n            m = train(s, B, tag)", f"        for s in [{seed}]:   # 나눠 학습: GINE · 시드 {seed}\n            m = train(s, B, tag)")
        rep('3c8f160f', "        save_cache(pred_cache('preds', tag, files), pset[KEY + cols])",
            f"        pset[{keep_cols} + ['pred_gnn_a0_s{seed}']].to_pickle('{OUT}/pred_{name}.pkl'); print('태그', tag)")
    else:
        rep('3c8f160f', "    TAGS[setting] = tag\n", "    TAGS[setting] = tag\n    continue   # 나눠 학습: GINE 쌍 모델은 다른 작업에서\n")
        rep('molebert-train', "        for s in SEEDS:\n            m = train(s, B, tag, make_model=PTPairModel)",
            f"        for s in [{seed}]:   # 나눠 학습: Mole-BERT · 시드 {seed}\n            m = train(s, B, tag, make_model=PTPairModel)")
        rep('molebert-train', "        save_cache(pred_cache('preds_pt', tag, files), pset[KEY + cols])",
            f"        pset[{keep_cols} + ['pred_pt_a0_s{seed}']].to_pickle('{OUT}/pred_{name}.pkl'); print('태그', tag)")
    for c in n['cells']:
        if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
    json.dump(n, open(f'{OUT}/{name}.ipynb', 'w'), ensure_ascii=False)

if sys.argv[1:] == ['smoke']:
    shutil.rmtree(f'{OUT}/smoke_runs', ignore_errors=True); os.makedirs(f'{OUT}/smoke_runs/pretrained')
    shutil.copy(f'{CAP}/runs/pretrained/Mole-BERT.pth', f'{OUT}/smoke_runs/pretrained/')
    build('smoke_gine', 'gine', 0, True); build('smoke_mb', 'mb', 0, True)
else:
    for kind in ['mb', 'gine']:
        for s in [0, 1, 2]:
            build(f'head_{kind}_s{s}', kind, s)
open(f'{CAP}/runs/_diag.txt', 'w').write('built\n')
