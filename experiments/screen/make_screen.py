# 검증 후보 비교 (시드 0): 30개 타깃 한 모델 · 누수 분자 빼기 · 쌍 모델. 후보마다 노트북 하나
# 결과: pred_<이름>.pkl (KEY, split, reason, delta, is_cliff, 예측) → 검증 쌍 Δ 오차로 고른다 (평가 쌍은 보지 않음)
# 사용: python make_screen.py [smoke]
import json, os, copy, shutil, sys
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/screen'
nb = json.load(open(f'{CAP}/ac_pipeline.ipynb'))
src = lambda c: ''.join(c['source'])
IDS = [c.get('id') for c in nb['cells']]
DROP = {'6a4888c8', '7660d566', 'b31556f2', '94d9465c', 'e1f1af72', 'svr-flow', 'edit-dist', 'model-architecture', 'gine-encoder', '697548aa'}   # 그림·예시 셀
CANDS = {   # 이름: (인코더, GINE 폭, 조각 인코더, 읽기, 헤드)
    'mb_H': ('mb', 64, True, 'mean', 'big2048'), 'mb_N': ('mb', 64, False, 'mean', 'big2048'), 'mb_N1a': ('mb', 64, False, 'sum', 'big2048'),
    'g300_H': ('gine', 300, True, 'mean', 'big2048'), 'g300_N': ('gine', 300, False, 'mean', 'big2048'), 'g300_N1a': ('gine', 300, False, 'sum', 'big2048'),
}

def build(name, enc, hidden, frag, read, head, seed=0, smoke=False):
    keep = [i for i in IDS[:IDS.index('3c8f160f') + 1] if i not in DROP] + (['molebert-train'] if enc == 'mb' else [])
    n = copy.deepcopy(nb); n['cells'] = [c for c in n['cells'] if c.get('id') in keep]
    def rep(cid, a, b):
        c = [c for c in n['cells'] if c.get('id') == cid][0]; s = src(c); assert s.count(a) == 1, (name, cid, a[:60]); c['source'] = s.replace(a, b)
    rep('8a15e3a3', "DATA, PAIRS = 'data/moleculeace', 'data/pairs'", f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
    assert "LEAK_MODE = 'drop_leak'" in src([c for c in n['cells'] if c.get('id') == '8a15e3a3'][0])
    if smoke: rep('8a15e3a3', "USE_ALL_TARGETS = True", "USE_ALL_TARGETS = False")
    rep('18184d19', "HIDDEN, LAYERS, BATCH, LR, EPOCHS = 64, 3, 256, 1e-3, 200", f"HIDDEN, LAYERS, BATCH, LR, EPOCHS = {hidden}, 3, 256, 1e-3, {2 if smoke else 200}")
    rep('18184d19', "USE_FRAG, USE_EDIT = True, True", f"USE_FRAG, USE_EDIT = {frag}, True")
    rep('18184d19', "HEAD = 'small'\n", f"HEAD = '{head}'\n")
    rep('18184d19', "READOUT = 'mean'\n", f"READOUT = '{read}'\n")
    rep('18184d19', "RUNS = 'runs'", f"RUNS = '{OUT}/smoke_runs'" if smoke else f"RUNS = '{CAP}/runs'")
    if enc == 'mb':
        rep('molebert-train', "PT_PATH = f'{RUNS}/pretrained/Mole-BERT.pth'", f"PT_PATH = '{CAP}/runs/pretrained/Mole-BERT.pth'")
    cols = "KEY + ['split', 'reason', 'delta', 'is_cliff']"
    if enc == 'gine':
        rep('3c8f160f', "        for s in SEEDS:\n            m = train(s, B, tag)", f"        for s in [{seed}]:   # 검증 후보: 시드 {seed}\n            m = train(s, B, tag)")
        rep('3c8f160f', "        save_cache(pred_cache('preds', tag, files), pset[KEY + cols])",
            f"        pset[{cols} + ['pred_gnn_a0_s{seed}']].to_pickle('{OUT}/pred_{name}.pkl'); print('태그', tag)")
    else:
        rep('3c8f160f', "    TAGS[setting] = tag\n", "    TAGS[setting] = tag\n    continue   # GINE 쌍 모델은 학습하지 않는다\n")
        rep('molebert-train', "        for s in SEEDS:\n            m = train(s, B, tag, make_model=PTPairModel)",
            f"        for s in [{seed}]:   # 검증 후보: 시드 {seed}\n            m = train(s, B, tag, make_model=PTPairModel)")
        rep('molebert-train', "        save_cache(pred_cache('preds_pt', tag, files), pset[KEY + cols])",
            f"        pset[{cols} + ['pred_pt_a0_s{seed}']].to_pickle('{OUT}/pred_{name}.pkl'); print('태그', tag)")
    for c in n['cells']:
        if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
    json.dump(n, open(f'{OUT}/{name}.ipynb', 'w'), ensure_ascii=False)

if sys.argv[1:] == ['smoke']:
    shutil.rmtree(f'{OUT}/smoke_runs', ignore_errors=True)
    build('smoke_mb_N1a', *CANDS['mb_N1a'], smoke=True); build('smoke_g300_N1a', *CANDS['g300_N1a'], smoke=True); build('smoke_mb_H', *CANDS['mb_H'], smoke=True)
else:
    for k, v in CANDS.items(): build(f'scr_{k}', *v)
print('built')
