# 타깃별 학습: 타깃 하나 × [GINE 쌍 | Mole-BERT 쌍 | 분자별(GINE·Mole-BERT)] 노트북, 시드 0–4
# 논문 설정 그대로(누수 분자 빼기, 헤드 1024 → 1024 → 1)에서 학습 단위만 타깃별로. 타깃마다 가중치 폴더를 따로 둔다 (runs/pertarget/<타깃>)
# 쌍 모델 학습 상한 200 → 1,000 에폭: 작은 타깃은 에폭당 업데이트가 몇 번뿐이라, 멈춤은 '20에폭 연속 개선 없음'으로만 정해지게 한다
# 사용: python make_pertarget.py [smoke <타깃>]  → runs/tmp/pertarget/jobs/*.ipynb, jobs.txt (큰 작업 먼저)
import json, os, copy, shutil, sys, glob
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/pertarget'
nb = json.load(open(f'{CAP}/ac_pipeline.ipynb'))
src = lambda c: ''.join(c['source'])
BASE = ['8a15e3a3', '27599456', 'cd5938aa', '157fbb3d', '143e0abe', '04bf80a0', '175f172f', 'bba88552', '7135026a', '0362b826',
        '18184d19', '3c8f160f']   # 그림·예시 셀은 뺀다 (학습에 쓰는 값을 만들지 않음)
EXTRA = {'gine': [], 'mb': ['molebert-train'], 'mol': ['gnnmol-train', 'molebert-train', 'molebert-mol']}

def build(t, kind, smoke=False):
    n = copy.deepcopy(nb); n['cells'] = [c for c in n['cells'] if c.get('id') in BASE + EXTRA[kind]]
    def rep(cid, a, b):
        c = [c for c in n['cells'] if c.get('id') == cid][0]; s = src(c); assert s.count(a) == 1, (t, kind, cid, a[:60]); c['source'] = s.replace(a, b)
    runs = f'{OUT}/smoke_runs/{t}' if smoke else f'{CAP}/runs/pertarget/{t}'
    rep('8a15e3a3', "DATA, PAIRS = 'data/moleculeace', 'data/pairs'", f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
    rep('8a15e3a3', "DEV_TARGETS = ['CHEMBL1862_Ki', 'CHEMBL237_EC50', 'CHEMBL4616_EC50', 'CHEMBL2034_Ki']", f"DEV_TARGETS = ['{t}']   # 타깃별 학습")
    rep('8a15e3a3', "USE_ALL_TARGETS = True", "USE_ALL_TARGETS = False")
    rep('8a15e3a3', "SEEDS = [0, 1, 2]", "SEEDS = [0, 1, 2, 3, 4]")
    assert "LEAK_MODE = 'drop_leak'" in src([c for c in n['cells'] if c.get('id') == '8a15e3a3'][0])
    rep('18184d19', "HEAD = 'small'", "HEAD = 'big'")
    rep('18184d19', "RUNS = 'runs'", f"RUNS = '{runs}'")
    rep('18184d19', "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, 200",
        "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, " + ('2' if smoke else '1000'))
    if 'molebert-train' in EXTRA[kind]:
        rep('molebert-train', "PT_PATH = f'{RUNS}/pretrained/Mole-BERT.pth'", f"PT_PATH = '{CAP}/runs/pretrained/Mole-BERT.pth'")
    if kind != 'gine':   # GINE 쌍 모델은 다른 작업에서
        rep('3c8f160f', "    TAGS[setting] = tag\n", "    TAGS[setting] = tag\n    continue   # 이 작업에서는 GINE 쌍 모델을 학습하지 않는다\n")
    if kind == 'mol':
        rep('molebert-train', "    PT_TAGS[setting] = tag\n", "    PT_TAGS[setting] = tag\n    continue   # 분자별 모델만 학습\n")
        if smoke: rep('gnnmol-train', "MOL_EPOCHS = 1000", "MOL_EPOCHS = 2")
    for c in n['cells']:
        if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
    os.makedirs(f'{OUT}/jobs', exist_ok=True)
    name = f"{'smoke_' if smoke else ''}{t}__{kind}"
    json.dump(n, open(f'{OUT}/jobs/{name}.ipynb', 'w'), ensure_ascii=False)
    return name

if sys.argv[1:2] == ['smoke']:
    t = sys.argv[2]
    shutil.rmtree(f'{OUT}/smoke_runs/{t}', ignore_errors=True)
    print(' '.join(build(t, k, True) for k in ['gine', 'mb', 'mol']))
else:
    # 큰 작업부터: Mole-BERT 쌍 > GINE 쌍 > 분자별, 같은 종류 안에서는 학습 쌍이 많은 타깃부터
    size = {os.path.basename(f)[:-7]: os.path.getsize(f) for f in glob.glob(f'{CAP}/data/pairs/*_v2.pkl')}
    T = sorted(size, key=size.get, reverse=True)
    jobs = [build(t, k) for k in ['mb', 'gine', 'mol'] for t in T]
    open(f'{OUT}/jobs.txt', 'w').write('\n'.join(jobs) + '\n'); print(len(jobs), 'jobs')
