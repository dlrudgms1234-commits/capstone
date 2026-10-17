# 분자별 모델 큰 헤드: 누수 없음·우리 방식 · HEAD big · GINE 분자별 + Mole-BERT 분자별 (한 작업, 시드 3)
import json, os, copy, shutil, sys
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/head'
nb = json.load(open(f'{CAP}/ac_pipeline.ipynb'))
src = lambda c: ''.join(c['source'])
IDS = [c.get('id') for c in nb['cells']]

def build(name, smoke=False):
    keep = IDS[:IDS.index('3c8f160f') + 1] + ['gnnmol-md', 'gnnmol-train', 'molebert-md', 'molebert-train', 'molebert-mol-md', 'molebert-mol']
    n = copy.deepcopy(nb); n['cells'] = [c for c in n['cells'] if c.get('id') in keep]
    def rep(cid, a, b):
        c = [c for c in n['cells'] if c.get('id') == cid][0]; s = src(c); assert s.count(a) == 1, (name, cid, a[:60]); c['source'] = s.replace(a, b)
    rep('8a15e3a3', "DATA, PAIRS = 'data/moleculeace', 'data/pairs'", f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
    rep('8a15e3a3', "LEAK_MODE = 'drop_leak'   # 논문 설정", "LEAK_MODE = 'drop_leak'")
    rep('18184d19', "HEAD = 'small'", "HEAD = 'big'")
    rep('18184d19', "RUNS = 'runs'", f"RUNS = '{OUT}/smoke_runs'" if smoke else f"RUNS = '{CAP}/runs'")
    if smoke:
        rep('8a15e3a3', "USE_ALL_TARGETS = True", "USE_ALL_TARGETS = False")
        rep('gnnmol-train', "MOL_EPOCHS = 1000", "MOL_EPOCHS = 2")
    rep('3c8f160f', "    TAGS[setting] = tag\n", "    TAGS[setting] = tag\n    continue   # 분자별 모델만 학습\n")
    rep('molebert-train', "    PT_TAGS[setting] = tag\n", "    PT_TAGS[setting] = tag\n    continue   # 분자별 모델만 학습\n")
    for c in n['cells']:
        if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
    json.dump(n, open(f'{OUT}/{name}.ipynb', 'w'), ensure_ascii=False)

if sys.argv[1:] == ['smoke']:
    shutil.rmtree(f'{OUT}/smoke_runs', ignore_errors=True); os.makedirs(f'{OUT}/smoke_runs/pretrained')
    shutil.copy(f'{CAP}/runs/pretrained/Mole-BERT.pth', f'{OUT}/smoke_runs/pretrained/')
    build('smoke_mol', True)
else:
    build('head_mol')
open(f'{CAP}/runs/_diag.txt', 'w').write('built\n')
