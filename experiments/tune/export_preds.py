"""모델별 예측값을 파일 하나(csv.gz)로 모은다. 가중치·GPU 없이 표와 검정을 다시 만들 수 있게 하는 것이 목적이다.
저장소 최상위에서 실행한다

    python experiments/tune/export_preds.py test share/preds_test.csv.gz          # 기본 모델 전부
    python experiments/tune/export_preds.py val  share/preds_val.csv.gz
    python experiments/tune/export_preds.py test out.csv.gz "내모델=runs/tune/내폴더/pred_mb_s?.pkl"   # 모델 추가·교체
    python experiments/tune/export_preds.py test out.csv.gz --add experiments/preds/preds_test.csv.gz \
        "gc_pair=runs/tune/gc_pair/pred_*_s?.pkl"        # 이미 있는 파일에 내 모델 열만 더하기

나오는 파일: 한 군데만 다른 쌍 한 줄마다
    target, i, j, delta(실측 Δ), is_cliff, eval_mol(쌍에서 학습 분자가 아닌 쪽 번호 = 군집 부트스트랩 단위),
    <모델>_s0 … <모델>_s4 (시드별 예측 Δ̂. SVR은 시드가 없어 _s0 하나)
검정은 `python experiments/tune/sig_test.py <이 파일>` 로 한다 (예측값만 쓰므로 이 저장소 밖에서도 돌아간다)

읽을 수 있는 예측 파일 세 가지
    쌍 모델   시드마다 파일 하나, 열 pred_…_s{시드} (`make_tune.py` → `run_queue.py` 결과)
    분자별    파일 하나에 시드 열 s0…s4 (활성 예측 → Δ̂ = p̂(B) − p̂(A))
    SVR       pred 열 하나
"""
import glob, sys
import numpy as np, pandas as pd

MODE, OUT = sys.argv[1], sys.argv[2]
assert MODE in ('val', 'test')
rest = sys.argv[3:]
ADD = rest[1] if rest[:1] == ['--add'] else None   # 이 파일에 열만 더한다 (쌍 목록·실측 Δ는 그대로)
KEY = ['target', 'i', 'j']
PRE = 'predval_' if MODE == 'val' else 'pred_'

# 기본 모델 묶음 (논문 표에 쓰는 것). 이름 = 열 앞머리, 값 = 예측 파일 glob
MODELS = {} if ADD else {
    'svr_mol':       f'runs/tmp/svr/{PRE}svr_mol.pkl',
    'svr_pair':      f'runs/tmp/svr/{PRE}svr_pair_g?.pkl',
    'gine_pair':     'runs/tmp/final3/pred_final3_gine_s?.pkl',
    'gine_mol':      'runs/preds_mol_t30_h300_*wd0.01*leakdrop_head1024_adamw_[0-9]*.pkl',
    'mb_pair':       'runs/tune/mb_ptlr1e-3/pred_mb_s?.pkl',            # 채택 설정 (인코더 학습률 1e-3)
    'mb_mol':        'runs/tune/mb_ptlr1e-3/pred_mol_mb.pkl',
    'mb_pair_lr1e4': 'runs/tmp/final3/pred_final3_mb_s?.pkl',           # 예전 설정 (인코더 1e-4)
    'mb_pair_noedit':     'runs/tune/mb_lr1e-3_noedit/pred_mb_s?.pkl',  # 빼기 실험: 편집 정보
    'mb_pair_notgt':      'runs/tune/mb_lr1e-3_notgt/pred_mb_s?.pkl',   #            타깃 one-hot
    'mb_pair_noedit_tgt': 'runs/tune/mb_lr1e-3_noedit_tgt/pred_mb_s?.pkl',
    'mb_pair_scratch':    'runs/tune/mb_scratch/pred_mb_s?.pkl',        # 사전학습 가중치 없음
}
MODELS.update(dict(a.split('=', 1) for a in (rest[2:] if ADD else rest)))

if ADD:
    P = pd.read_csv(ADD)
else:
    # 쌍 목록과 실측 Δ: 있는 쌍 모델 예측 파일 중 아무 하나에서 가져온다 (쌍 모델 예측은 모두 같은 쌍)
    def first_pair_file():
        for pat in MODELS.values():
            for f in sorted(glob.glob(pat)):
                if {'split', 'reason', 'delta'} <= set(pd.read_pickle(f).columns):   # 쌍 표 (SVR·분자별 파일에는 없다)
                    return f
        sys.exit('쌍 모델 예측 파일이 하나도 없다. 경로를 확인하거나 "이름=경로"로 직접 넘긴다')
    base = pd.read_pickle(first_pair_file())
    m = (base.split.str.startswith('val') if MODE == 'val' else base.split == 'test') & (base.reason == 'pass')
    P = base[m][KEY + ['delta', 'is_cliff']].reset_index(drop=True)
    # eval_mol = 쌍에서 학습 분자가 아닌 쪽 (val 쌍은 valid 분자, test 쌍은 test 분자).
    # 분자 split은 노트북과 같은 규칙으로 다시 만든다 (SEED 0, train의 20%를 valid로)
    SEED, VAL_FRAC = 0, 0.2
    def mol_split(t):
        s = pd.read_csv(f'data/moleculeace/{t}.csv').split
        tr = s.index[s == 'train']
        return s.mask(s.index.isin(np.random.default_rng(SEED).choice(tr, int(len(tr) * VAL_FRAC), replace=False)), 'val')
    SP = {t: mol_split(t).values for t in P.target.unique()}
    P['eval_mol'] = [i if SP[t][i] == MODE else j for t, i, j in zip(P.target, P.i, P.j)]
    assert all(SP[t][k] == MODE for t, k in zip(P.target, P.eval_mol)), 'eval_mol 규칙이 예측 파일과 맞지 않다'

def add(name, pat):
    fs = sorted(glob.glob(pat))
    if not fs:
        print(f'  (없음, 건너뜀) {name}: {pat}'); return
    first = pd.read_pickle(fs[0])
    if 'k' in first.columns:               # 분자별: 활성 예측 → Δ̂
        g = {t: x for t, x in first.groupby('target')}
        for s in [c for c in first.columns if c[0] == 's' and c[1:].isdigit()]:
            P[f'{name}_{s}'] = [g[t][s].values[j] - g[t][s].values[i] for t, i, j in zip(P.target, P.i, P.j)]
    elif 'pred' in first.columns:          # SVR: 타깃별 파일을 이어 붙인다
        c = pd.concat([pd.read_pickle(f) for f in fs]).drop_duplicates(KEY)
        P[f'{name}_s0'] = take(c, 'pred', name)
    else:                                  # 쌍: 시드마다 파일 하나
        for f in fs:
            d = pd.read_pickle(f); col = [c for c in d.columns if c.startswith('pred_')][0]
            P[f'{name}_s{f[-5]}'] = take(d, col, f)
    print(f'  {name}: 파일 {len(fs)}개 → 열 {sum(c.startswith(name + "_s") for c in P.columns)}개')

def take(d, col, who):
    """쌍 번호로 맞춰 예측 열을 가져온다 (파일에 쌍이 더 많아도 된다)"""
    v = P[KEY].merge(d[KEY + [col]].drop_duplicates(KEY), on=KEY, how='left')[col].values
    assert not np.isnan(v).any(), f'{who}: 비어 있는 쌍이 있다 (쌍 목록이 다른 파일)'
    return v

print(f'{MODE} 쌍 {len(P):,}개 (cliff {int(P.is_cliff.sum()):,})' + (f', {ADD} 에 열 추가' if ADD else ''))
for name, pat in MODELS.items():
    add(name, pat)
P.to_csv(OUT, index=False, float_format='%.5g')
print(f'→ {OUT}  ({len(P):,}행 × {len(P.columns)}열)')
