# XAI 노트북 만들기: 누수 없음·우리 방식 전체 실행 노트북(runs/tmp/leak/leak_full.ipynb)의 Mole-BERT 분자별 칸까지 + xai_cell.py
# 실행: python runs/tmp/xai/make_xai.py → runs/tmp/xai/xai.ipynb 를 nbconvert로 실행
import json, os
D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(D, '..', '..', '..'))
nb = json.load(open(f'{ROOT}/runs/tmp/leak/leak_full.ipynb'))
ids = [c.get('id') for c in nb['cells']]
nb['cells'] = nb['cells'][:ids.index('molebert-mol') + 1]
for c in nb['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
nb['cells'].append(dict(cell_type='code', id='xai', metadata={}, outputs=[], execution_count=None, source=open(f'{D}/xai_cell.py').read()))
json.dump(nb, open(f'{D}/xai.ipynb', 'w'), ensure_ascii=False)
print('ok', len(nb['cells']))
