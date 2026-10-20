# 최종 설계 3 (계획 문서 10-19 14:40): 예전 설계(조각 인코더 있음, 평균 읽기, 헤드 1024 → 1024, 검증 쌍 전체로 멈춤) + AdamW 가중치 감쇠 0.01
# 쌍 [Mole-BERT, GINE 300] × 시드 0–4 + 분자별(GINE 300·Mole-BERT) [평균, 합]. make_final2.py의 build를 그대로 쓰고 최적화기만 바꾼다
# 사용: CFG=.../final3/config.json python make_final3.py [smoke]
import json, os, sys
OUT3 = '/home/bising0922/Capstone/runs/tmp/final3'
src = open('/home/bising0922/Capstone/runs/tmp/final2/make_final2.py').read().replace("OUT = f'{CAP}/runs/tmp/final2'", "OUT = f'{CAP}/runs/tmp/final3'")
exec(src[:src.index("if sys.argv[1:] == ['smoke']")])
assert OUT == OUT3
def adamw(name):
    p = f'{OUT}/jobs/{name}.ipynb'; jb = json.load(open(p))
    c = [c for c in jb['cells'] if c.get('id') == '18184d19'][0]; s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    for a, b in [("HEAD_DROPOUT, OPTIM = 0.0, 'adam'", "HEAD_DROPOUT, OPTIM = 0.0, 'adamw'"), ("DROPOUT, WD = 0.3, 1e-4", "DROPOUT, WD = 0.3, 0.01")]:
        assert s.count(a) == 1, (name, a); s = s.replace(a, b)
    c['source'] = s; json.dump(jb, open(p, 'w'), ensure_ascii=False); return name
if sys.argv[1:] == ['smoke']:
    import shutil; shutil.rmtree(f'{OUT}/smoke_runs', ignore_errors=True); os.makedirs(f'{OUT}/smoke_runs')
    print(adamw(build('smoke_mol', 'mol', readout='mean', smoke=True)))
else:
    jobs = [adamw(build(f'final3_mb_s{s}', 'mb', s)) for s in range(5)] + [adamw(build(f'final3_gine_s{s}', 'gine', s)) for s in range(5)] + \
           [adamw(build('final3_mol_mean', 'mol', readout='mean')), adamw(build('final3_mol_sum', 'mol', readout='sum'))]
    open(f'{OUT}/jobs.txt', 'w').write('\n'.join(jobs) + '\n'); print(len(jobs), 'jobs', cfg)
