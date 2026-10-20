# 계획_고정_2026-10-18.md 2단계 결과(타깃별이 검증에서 더 나쁨) → 30개 타깃 한 모델로 되돌리고 시드 3, 4 추가
# 쌍: experiments/head/make_head.py의 build로 head_{mb,gine}_s{3,4}. 분자별: make_molhead.py의 build에 SEEDS = 0–4
# (분자별을 시드 3, 4만 돌리면 save_cache가 같은 태그의 시드 0–2 예측 저장본을 지우므로, 0–2는 가중치를 불러와 예측만 다시 한다)
import json, os, sys
CAP = '/home/bising0922/Capstone'
for f in ['experiments/head/make_head.py', 'experiments/head/make_molhead.py']:
    s = open(f'{CAP}/{f}').read()
    exec(s[:s.index("if sys.argv[1:] == ['smoke']")])
    if 'molhead' in f:
        build_mol = build
    else:
        build_pair = build
for kind in ['mb', 'gine']:
    for seed in [3, 4]:
        build_pair(f'head_{kind}_s{seed}', kind, seed)
build_mol('head_mol_s01234')
p = f'{CAP}/runs/tmp/head/head_mol_s01234.ipynb'; nb = json.load(open(p))
c = [c for c in nb['cells'] if c.get('id') == '8a15e3a3'][0]; s = ''.join(c['source'])
assert s.count('SEEDS = [0, 1, 2]') == 1; c['source'] = s.replace('SEEDS = [0, 1, 2]', 'SEEDS = [0, 1, 2, 3, 4]')
json.dump(nb, open(p, 'w'), ensure_ascii=False)
print('built')
