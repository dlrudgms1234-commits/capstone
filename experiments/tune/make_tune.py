"""하이퍼파라미터 튜닝용 작업 만들기: 노트북 기본값(최종 설계 3)에서 가중치 감쇠 등만 바꿔 시드별 노트북을 만든다.
저장소 최상위에서 실행한다. 결과(예측)는 runs/tune/<이름>/ 에, 가중치는 runs/ 에 (설정마다 이름이 달라 겹치지 않음)

사용 예 (GINE 쌍·분자별, 감쇠 0.03, 시드 0–4):
    python experiments/tune/make_tune.py --name gine_wd0.03 --enc gine --wd 0.03
    python experiments/tune/run_queue.py runs/tune/gine_wd0.03 3      # 동시 3개 (GPU·CPU 사정에 맞게)
    python experiments/tune/tune_eval.py val runs/tune/gine_wd0.03     # valid 표 + 학습 곡선 그림
옵션: --enc gine|mb|both, --wd 감쇠, --l2sp λ (Mole-BERT만), --pt-wd-scale 배수 (Mole-BERT 인코더 감쇠 = wd × 배수), --seeds 0,1,2,3,4, --no-mol (분자별 생략), --smoke (타깃 4개·2에폭 시험)
"""
import argparse, copy, json, os, shutil

CAP = os.getcwd()
ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True); ap.add_argument('--enc', default='gine', choices=['gine', 'mb', 'both'])
ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--l2sp', type=float, default=0.0)
ap.add_argument('--pt-wd-scale', type=float, default=1.0)   # Mole-BERT 인코더 감쇠 = wd × 이 값 (10이면 GINE 인코더와 같은 실제 세기)
ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--no-mol', action='store_true'); ap.add_argument('--smoke', action='store_true')
a = ap.parse_args()
OUT = f'{CAP}/runs/tune/{a.name}'; RUNS = f'{OUT}/smoke_runs' if a.smoke else f'{CAP}/runs'
nb = json.load(open(f'{CAP}/ac_pipeline.ipynb'))
src = lambda c: ''.join(c['source'])
IDS = [c.get('id') for c in nb['cells']]
DROP = {'6a4888c8', '7660d566', 'b31556f2', '94d9465c', 'e1f1af72', 'svr-flow', 'edit-dist', 'model-architecture', 'gine-encoder', '697548aa'}   # 그림·예시 셀
BASE = [i for i in IDS[:IDS.index('3c8f160f') + 1] if i not in DROP]

def build(name, kind, seed=None):
    mol_cells = ['gnnmol-train'] + ([] if a.enc == 'gine' else ['molebert-train', 'molebert-mol'])   # GINE만 튜닝하면 Mole-BERT 셀은 넣지 않는다
    keep = BASE + {'gine': [], 'mb': ['molebert-train'], 'mol': mol_cells}[kind]
    n = copy.deepcopy(nb); n['cells'] = [c for c in n['cells'] if c.get('id') in keep]
    def rep(cid, x, y):
        c = [c for c in n['cells'] if c.get('id') == cid][0]; s = src(c); assert s.count(x) == 1, (name, cid, x[:60]); c['source'] = s.replace(x, y)
    rep('8a15e3a3', "DATA, PAIRS = 'data/moleculeace', 'data/pairs'", f"DATA, PAIRS = '{CAP}/data/moleculeace', '{CAP}/data/pairs'")
    if a.smoke:
        rep('8a15e3a3', "USE_ALL_TARGETS = True", "USE_ALL_TARGETS = False")
        rep('18184d19', "LR, EPOCHS = 300, 3, 256, 1e-3, 200", "LR, EPOCHS = 300, 3, 256, 1e-3, 2")
        if kind == 'mol': rep('gnnmol-train', "MOL_EPOCHS = 1000", "MOL_EPOCHS = 2")
    rep('18184d19', "DROPOUT, WD = 0.3, 0.01", f"DROPOUT, WD = 0.3, {a.wd}")
    rep('18184d19', "L2SP = 0.0\n", f"L2SP = {a.l2sp}\n")
    rep('18184d19', "PT_WD_SCALE = 1.0\n", f"PT_WD_SCALE = {a.pt_wd_scale}\n")
    rep('18184d19', "RUNS = 'runs'", f"RUNS = '{RUNS}'")
    if 'molebert-train' in keep:
        rep('molebert-train', "PT_PATH = f'{RUNS}/pretrained/Mole-BERT.pth'", f"PT_PATH = '{CAP}/runs/pretrained/Mole-BERT.pth'")
    cols = "KEY + ['split', 'reason', 'delta', 'is_cliff']"
    if kind == 'gine':
        rep('3c8f160f', "        for s in SEEDS:\n            m = train(s, B, tag)", f"        for s in [{seed}]:   # 튜닝: 시드 {seed}\n            m = train(s, B, tag)")
        rep('3c8f160f', "        save_cache(pred_cache('preds', tag, files), pset[KEY + cols])",
            f"        pset[{cols} + ['pred_gnn_a0_s{seed}']].to_pickle('{OUT}/pred_gine_s{seed}.pkl'); open('{OUT}/tag_gine.txt', 'w').write(tag)")
    else:
        rep('3c8f160f', "    TAGS[setting] = tag\n", "    TAGS[setting] = tag\n    continue   # 이 작업에서는 GINE 쌍 모델을 학습하지 않는다\n")
    if kind == 'mb':
        rep('molebert-train', "        for s in SEEDS:\n            m = train(s, B, tag, make_model=PTPairModel)",
            f"        for s in [{seed}]:   # 튜닝: 시드 {seed}\n            m = train(s, B, tag, make_model=PTPairModel)")
        rep('molebert-train', "        save_cache(pred_cache('preds_pt', tag, files), pset[KEY + cols])",
            f"        pset[{cols} + ['pred_pt_a0_s{seed}']].to_pickle('{OUT}/pred_mb_s{seed}.pkl'); open('{OUT}/tag_mb.txt', 'w').write(tag)")
    if kind == 'mol':
        if 'molebert-train' in keep:
            rep('molebert-train', "    PT_TAGS[setting] = tag\n", "    PT_TAGS[setting] = tag\n    continue   # 분자별 모델만 학습\n")
        rep('gnnmol-train', "    save_cache(pred_cache('preds_mol', MOL_TAG, files), MOL_PRED)\n",
            f"    save_cache(pred_cache('preds_mol', MOL_TAG, files), MOL_PRED)\nMOL_PRED.to_pickle('{OUT}/pred_mol_gine.pkl'); open('{OUT}/tag_mol.txt', 'w').write(MOL_TAG)\n")
        if a.enc != 'gine':
            rep('molebert-mol', "    save_cache(pred_cache('preds_molpt', PTMOL_TAG, files), PTMOL_PRED)\n",
                f"    save_cache(pred_cache('preds_molpt', PTMOL_TAG, files), PTMOL_PRED)\nPTMOL_PRED.to_pickle('{OUT}/pred_mol_mb.pkl'); open('{OUT}/tag_molmb.txt', 'w').write(PTMOL_TAG)\n")
    for c in n['cells']:
        if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
    json.dump(n, open(f'{OUT}/jobs/{name}.ipynb', 'w'), ensure_ascii=False)
    return name

if a.smoke: shutil.rmtree(RUNS, ignore_errors=True); os.makedirs(f'{RUNS}/pretrained', exist_ok=True)
os.makedirs(f'{OUT}/jobs', exist_ok=True)
seeds = [int(s) for s in a.seeds.split(',')]
encs = ['mb', 'gine'] if a.enc == 'both' else [a.enc]
jobs = [build(f'{e}_s{s}', e, s) for e in encs for s in seeds] + ([] if a.no_mol else [build('mol', 'mol')])
open(f'{OUT}/jobs.txt', 'w').write('\n'.join(jobs) + '\n')
json.dump(dict(vars(a), runs=RUNS), open(f'{OUT}/settings.json', 'w'), ensure_ascii=False)   # runs = 가중치·학습 기록 위치
print(len(jobs), 'jobs →', OUT)
