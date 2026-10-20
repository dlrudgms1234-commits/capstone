# 과적합 대책 후보 (계획_고정_2026-10-18.md 10-19 01:00): 정해진 설계(config.json)의 Mole-BERT 쌍 시드 0에 대책 하나씩
# reg_drop: 헤드 드롭아웃 0.2 / reg_adamw: AdamW 가중치 감쇠 0.01 / reg_ptlr: Mole-BERT 인코더 학습률 3e-5
# 사용: python make_reg.py [smoke]  → jobs/reg_*.ipynb, 작업 이름을 표준 출력으로
import json, os, sys
OUT = '/home/bising0922/Capstone/runs/tmp/final2'
src = open(f'{OUT}/make_final2.py').read()
exec(src[:src.index("if sys.argv[1:] == ['smoke']")])
smoke = sys.argv[1:] == ['smoke']
REG = {'reg_drop': [('18184d19', "HEAD_DROPOUT, OPTIM = 0.0, 'adam'", "HEAD_DROPOUT, OPTIM = 0.2, 'adam'")],
       'reg_adamw': [('18184d19', "HEAD_DROPOUT, OPTIM = 0.0, 'adam'", "HEAD_DROPOUT, OPTIM = 0.0, 'adamw'"),
                     ('18184d19', "DROPOUT, WD = 0.3, 1e-4", "DROPOUT, WD = 0.3, 0.01")],
       'reg_ptlr': [('molebert-train', "PT_DIM, PT_LAYERS, PT_LR = 300, 5, 1e-4", "PT_DIM, PT_LAYERS, PT_LR = 300, 5, 3e-5")]}
names = []
for r, edits in REG.items():
    name = build(f"{'smoke_' if smoke else ''}{r}_mb_s0", 'mb', 0, smoke=smoke)
    p = f'{OUT}/jobs/{name}.ipynb'; jb = json.load(open(p))
    for cid, a, b in edits:
        c = [c for c in jb['cells'] if c.get('id') == cid][0]; s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        assert s.count(a) == 1, (r, a); c['source'] = s.replace(a, b)
    json.dump(jb, open(p, 'w'), ensure_ascii=False); names.append(name)
print('\n'.join(names))
