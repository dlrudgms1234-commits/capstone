# 원인 찾기: 최종 설계 Mole-BERT 쌍(시드 3)에서 한 가지씩만 되돌린다. 검증 쌍으로 최종 설계 시드 3과 비교
# r_frag: 조각 인코더 다시 넣기 / r_mean: 평균 읽기 / r_head: 헤드 1024 → 1024 / r_valall: 멈춤 기준 검증 쌍 전체
import json, re, sys
CAP = '/home/bising0922/Capstone'; OUT = f'{CAP}/runs/tmp/revert'
src = open(f'{CAP}/runs/tmp/final/make_final.py').read()
exec(src[:src.index("if sys.argv[1:] == ['smoke']")].replace("OUT = f'{CAP}/runs/tmp/final'", "OUT = f'{CAP}/runs/tmp/revert'"))
REV = {'r_frag': ("USE_FRAG, USE_EDIT = False, True", "USE_FRAG, USE_EDIT = True, True"),
       'r_mean': ("READOUT = 'sum'\n", "READOUT = 'mean'\n"),
       'r_head': ("HEAD = 'big2048'\n", "HEAD = 'big'\n"),
       'r_valall': ("VAL_PAIRS = 'pass'\n", "VAL_PAIRS = 'all'\n")}
for name, (a, b) in REV.items():
    build(name, 'mb', 3)
    p = f'{OUT}/jobs/{name}.ipynb'; jb = json.load(open(p))
    c = [c for c in jb['cells'] if c.get('id') == '18184d19'][0]; s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    assert s.count(a) == 1, (name, a); c['source'] = s.replace(a, b)
    json.dump(jb, open(p, 'w'), ensure_ascii=False)
print('built', list(REV))
