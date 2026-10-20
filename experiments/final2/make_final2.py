# 최종 설계 2: config.json(decide.py가 정함)의 쌍 모델 설정 + 분자별은 평균·합 읽기 둘 다. 시드 0–4, 30개 타깃 한 모델, 누수 분자 빼기, GINE 폭 300, 카드 메모리 50% 상한
# 작업: 쌍 [Mole-BERT, GINE 300] × 시드 0–4 + 분자별(GINE 300·Mole-BERT) [평균, 합]. 같은 설정의 가중치가 이미 있으면 노트북이 불러와 다시 학습하지 않는다
# 사용: python make_final2.py [smoke]
import json, os, copy, shutil, sys
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/final2'
cfg = json.load(open(os.environ.get('CFG', f'{OUT}/config.json')))   # 시험 때는 CFG 환경변수로 다른 설정 파일
nb = json.load(open(f'{CAP}/ac_pipeline.ipynb'))
src = lambda c: ''.join(c['source'])
IDS = [c.get('id') for c in nb['cells']]
DROP = {'6a4888c8', '7660d566', 'b31556f2', '94d9465c', 'e1f1af72', 'svr-flow', 'edit-dist', 'model-architecture', 'gine-encoder', '697548aa'}
BASE = [i for i in IDS[:IDS.index('3c8f160f') + 1] if i not in DROP]

def build(name, kind, seed=None, readout=None, smoke=False):
    keep = BASE + {'gine': [], 'mb': ['molebert-train'], 'mol': ['gnnmol-train', 'molebert-train', 'molebert-mol']}[kind]
    n = copy.deepcopy(nb); n['cells'] = [c for c in n['cells'] if c.get('id') in keep]
    def rep(cid, a, b):
        c = [c for c in n['cells'] if c.get('id') == cid][0]; s = src(c); assert s.count(a) == 1, (name, cid, a[:60]); c['source'] = s.replace(a, b)
    rep('8a15e3a3', "DATA, PAIRS = 'data/moleculeace', 'data/pairs'", f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
    assert "LEAK_MODE = 'drop_leak'" in src([c for c in n['cells'] if c.get('id') == '8a15e3a3'][0])
    assert "set_per_process_memory_fraction(0.5)" in src([c for c in n['cells'] if c.get('id') == '18184d19'][0])
    if smoke: rep('8a15e3a3', "USE_ALL_TARGETS = True", "USE_ALL_TARGETS = False")
    rep('18184d19', "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, 200", f"HIDDEN, LAYERS, BATCH, LR, EPOCHS = 300, 3, 256, 1e-3, {2 if smoke else 200}")
    rep('18184d19', "USE_FRAG, USE_EDIT = True, True", f"USE_FRAG, USE_EDIT = {cfg['frag']}, True")
    rep('18184d19', "HEAD = 'small'\n", f"HEAD = '{cfg['head']}'\n")
    rep('18184d19', "READOUT = 'mean'\n", f"READOUT = '{readout or cfg['readout']}'\n")
    rep('18184d19', "VAL_PAIRS = 'all'\n", f"VAL_PAIRS = '{cfg['val_pairs']}'\n")
    rep('18184d19', "RUNS = 'runs'", f"RUNS = '{OUT}/smoke_runs'" if smoke else f"RUNS = '{CAP}/runs'")
    if 'molebert-train' in keep:
        rep('molebert-train', "PT_PATH = f'{RUNS}/pretrained/Mole-BERT.pth'", f"PT_PATH = '{CAP}/runs/pretrained/Mole-BERT.pth'")
    cols = "KEY + ['split', 'reason', 'delta', 'is_cliff']"
    if kind == 'gine':
        rep('3c8f160f', "        for s in SEEDS:\n            m = train(s, B, tag)", f"        for s in [{seed}]:   # 최종 설계 2: 시드 {seed}\n            m = train(s, B, tag)")
        rep('3c8f160f', "        save_cache(pred_cache('preds', tag, files), pset[KEY + cols])",
            f"        pset[{cols} + ['pred_gnn_a0_s{seed}']].to_pickle('{OUT}/pred_{name}.pkl'); print('태그', tag)")
    else:
        rep('3c8f160f', "    TAGS[setting] = tag\n", "    TAGS[setting] = tag\n    continue   # 이 작업에서는 GINE 쌍 모델을 학습하지 않는다\n")
    if kind == 'mb':
        rep('molebert-train', "        for s in SEEDS:\n            m = train(s, B, tag, make_model=PTPairModel)",
            f"        for s in [{seed}]:   # 최종 설계 2: 시드 {seed}\n            m = train(s, B, tag, make_model=PTPairModel)")
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
    print(' '.join([build('smoke_mb', 'mb', 0, smoke=True), build('smoke_mol_mean', 'mol', readout='mean', smoke=True)]))
else:
    jobs = [build('final2_mol_mean', 'mol', readout='mean'), build('final2_mol_sum', 'mol', readout='sum')] + \
           [build(f'final2_{k}_s{s}', k, s) for k in ['mb', 'gine'] for s in range(5)]
    open(f'{OUT}/jobs.txt', 'w').write('\n'.join(jobs) + '\n'); print(len(jobs), 'jobs', cfg)
