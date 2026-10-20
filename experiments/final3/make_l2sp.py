# L2-SP (계획 문서 10-20 12:40): 최종 설계 3(노트북 기본값)에서 Mole-BERT 인코더만 사전학습 가중치 쪽으로 당김. 헤드는 AdamW 0.01 그대로
# 1단계: λ ∈ {0.01, 0.1} × Mole-BERT 쌍 시드 0–2 + Mole-BERT 분자별(시드 0–4, GINE 분자별은 저장된 가중치를 불러옴)
# 사용: python make_l2sp.py [smoke | seeds <λ> <시드,...>]
import json, os, copy, shutil, sys
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/l2sp'
nb = json.load(open(f'{CAP}/ac_pipeline.ipynb'))
src = lambda c: ''.join(c['source'])
IDS = [c.get('id') for c in nb['cells']]
DROP = {'6a4888c8', '7660d566', 'b31556f2', '94d9465c', 'e1f1af72', 'svr-flow', 'edit-dist', 'model-architecture', 'gine-encoder', '697548aa'}
BASE = [i for i in IDS[:IDS.index('3c8f160f') + 1] if i not in DROP]

def build(name, kind, lam, seed=None, smoke=False):
    keep = BASE + {'mb': ['molebert-train'], 'mol': ['gnnmol-train', 'molebert-train', 'molebert-mol']}[kind]
    n = copy.deepcopy(nb); n['cells'] = [c for c in n['cells'] if c.get('id') in keep]
    def rep(cid, a, b):
        c = [c for c in n['cells'] if c.get('id') == cid][0]; s = src(c); assert s.count(a) == 1, (name, cid, a[:60]); c['source'] = s.replace(a, b)
    cfg = src([c for c in n['cells'] if c.get('id') == '18184d19'][0])
    for must in ["HEAD = 'big'", "OPTIM = 0.0, 'adamw'", "WD = 0.3, 0.01", "EPOCHS = 300, 3, 256, 1e-3, 200", "USE_FRAG, USE_EDIT = True, True", "READOUT = 'mean'", "VAL_PAIRS = 'all'", "memory_fraction(0.5)"]:
        assert must in cfg, must   # 노트북 기본값 = 최종 설계 3인지 확인
    rep('8a15e3a3', "DATA, PAIRS = 'data/moleculeace', 'data/pairs'", f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
    if smoke:
        rep('8a15e3a3', "USE_ALL_TARGETS = True", "USE_ALL_TARGETS = False")
        rep('18184d19', "LR, EPOCHS = 300, 3, 256, 1e-3, 200", "LR, EPOCHS = 300, 3, 256, 1e-3, 2")
    rep('18184d19', "L2SP = 0.0\n", f"L2SP = {lam}\n")
    rep('18184d19', "RUNS = 'runs'", f"RUNS = '{OUT}/smoke_runs'" if smoke else f"RUNS = '{CAP}/runs'")
    rep('molebert-train', "PT_PATH = f'{RUNS}/pretrained/Mole-BERT.pth'", f"PT_PATH = '{CAP}/runs/pretrained/Mole-BERT.pth'")
    rep('3c8f160f', "    TAGS[setting] = tag\n", "    TAGS[setting] = tag\n    continue   # GINE 쌍 모델은 학습하지 않는다\n")
    if kind == 'mb':
        rep('molebert-train', "        for s in SEEDS:\n            m = train(s, B, tag, make_model=PTPairModel)",
            f"        for s in [{seed}]:   # L2-SP: 시드 {seed}\n            m = train(s, B, tag, make_model=PTPairModel)")
        rep('molebert-train', "        save_cache(pred_cache('preds_pt', tag, files), pset[KEY + cols])",
            f"        pset[KEY + ['split', 'reason', 'delta', 'is_cliff', 'pred_pt_a0_s{seed}']].to_pickle('{OUT}/pred_{name}.pkl'); print('태그', tag)")
    else:
        rep('molebert-train', "    PT_TAGS[setting] = tag\n", "    PT_TAGS[setting] = tag\n    continue   # 분자별 모델만 학습\n")
        if smoke: rep('gnnmol-train', "MOL_EPOCHS = 1000", "MOL_EPOCHS = 2")
    for c in n['cells']:
        if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
    os.makedirs(f'{OUT}/jobs', exist_ok=True)
    json.dump(n, open(f'{OUT}/jobs/{name}.ipynb', 'w'), ensure_ascii=False)
    return name

if sys.argv[1:2] == ['smoke']:
    shutil.rmtree(f'{OUT}/smoke_runs', ignore_errors=True); os.makedirs(f'{OUT}/smoke_runs')
    print(build('smoke_mb', 'mb', 0.1, 0, True), build('smoke_mol', 'mol', 0.1, smoke=True))
elif sys.argv[1:2] == ['seeds']:   # 2단계: 고른 λ로 시드 추가
    lam = float(sys.argv[2]); jobs = [build(f'l2sp{lam}_mb_s{s}', 'mb', lam, int(s)) for s in sys.argv[3].split(',')]
    open(f'{OUT}/jobs.txt', 'w').write('\n'.join(jobs) + '\n'); print(jobs)
else:
    jobs = [build(f'l2sp{lam}_mb_s{s}', 'mb', lam, s) for lam in (0.01, 0.1) for s in range(3)] + [build(f'l2sp{lam}_mol', 'mol', lam) for lam in (0.01, 0.1)]
    open(f'{OUT}/jobs.txt', 'w').write('\n'.join(jobs) + '\n'); print(len(jobs), 'jobs')
