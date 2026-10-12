# 팀 논의용 PDF: 세 설정(누수 있음 / 우리 방식 / 팀원 방식) 비교 + 다음 방향
import os, subprocess, datetime, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = '/home/bising0922/Capstone/runs/tmp/ecfp'; OUT = '/home/bising0922/Capstone'
R = pd.read_pickle(f'{D}/report3.pkl')
ORDER = ['SVR', 'ECFP + MLP 쌍', 'GINE 분자별', 'GINE 쌍', 'Mole-BERT 분자별', 'Mole-BERT 쌍']
SET = {1: '① 누수 있음 (MoleculeACE 분할)', 2: '② 누수 없음 · 우리 방식', 3: '③ 누수 없음 · 팀원 방식'}
NPAIR = {1: '43,896', 2: '38,711', 3: '52,900'}
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
COL = {'SVR': '#eb6834', 'ECFP + MLP 쌍': '#eda100', 'GINE 분자별': '#9fd9c4', 'GINE 쌍': '#1baf7a', 'Mole-BERT 분자별': '#a9c8f0', 'Mole-BERT 쌍': '#2a78d6'}

# 그림: cliff 평가 RMSE
fig, ax = plt.subplots(figsize=(10, 4.2))
w = 0.13
for k, m in enumerate(ORDER):
    vals = [R[(R.설정 == s)].loc[m, 'cliff'] if m in R[(R.설정 == s)].index else np.nan for s in [1, 2, 3]]
    x = np.arange(3) + (k - 2.5) * w
    ax.bar(x, vals, w * 0.92, color=COL[m], label=m, zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, (v if not np.isnan(v) else 0.02) + 0.015, '학습 중' if np.isnan(v) else f'{v:.2f}', ha='center', fontsize=6.5, rotation=90, color='0.3')
ax.set_xticks(range(3)); ax.set_xticklabels([SET[s] for s in [1, 2, 3]], fontsize=9)
ax.set_ylabel('cliff 평가 RMSE (낮을수록 좋음)'); ax.set_ylim(0, 1.65)
ax.grid(axis='y', alpha=0.3, zorder=0); ax.spines[['top', 'right']].set_visible(False)
ax.legend(ncol=3, fontsize=8, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(f'{D}/fig_cliff.png', dpi=200); plt.close(fig)

def table(s):
    t = R[R.설정 == s]
    best = {c: t[c].min() for c in ['cliff', 'non-cliff', '전체']}
    rows = ''
    for m in ORDER:
        if m in t.index:
            cells = ''.join(f"<td>{'<b>' if abs(t.loc[m, c] - best[c]) < 1e-9 else ''}{t.loc[m, c]:.3f}{'</b>' if abs(t.loc[m, c] - best[c]) < 1e-9 else ''}</td>" for c in ['cliff', 'non-cliff', '전체'])
        else:
            cells = '<td colspan="3" style="color:#888">학습 중 (끝나면 다시 생성)</td>'
        rows += f'<tr><td class="l">{m}</td>{cells}</tr>'
    return f'<p class="cap">{SET[s]} — 한 군데만 다른 평가 쌍 {NPAIR[s]}개</p><table><tr><th class="l">모델</th><th>cliff</th><th>non-cliff</th><th>전체</th></tr>{rows}</table>'

def gain(s):
    t = R[R.설정 == s]
    return ''.join(f'<td>{t.loc[a, "cliff"]:.3f} → <b>{t.loc[b, "cliff"]:.3f}</b></td>' for a, b in [('GINE 분자별', 'GINE 쌍'), ('Mole-BERT 분자별', 'Mole-BERT 쌍')])

today = datetime.date.today().strftime('%Y-%m-%d')
html = f'''<html><head><meta charset="utf-8"><style>
body {{ font-family: 'Noto Sans CJK KR'; font-size: 10pt; }}
h1 {{ font-size: 16pt; }} h2 {{ font-size: 12.5pt; margin-top: 14pt; }}
table {{ border-collapse: collapse; margin: 4pt 0 10pt 0; }}
th, td {{ border: 0.5pt solid #999999; padding: 3pt 8pt; text-align: center; font-size: 9.5pt; }}
th {{ background: #eeeeee; }} .l {{ text-align: left; }} .cap {{ font-weight: bold; margin: 8pt 0 2pt 0; }}
.note {{ color: #555; font-size: 9pt; }}
</style></head><body>
<h1>누수 처리 방식별 모델 비교와 다음 방향</h1>
<p class="note">{today} · 박윤정 · 팀 논의용</p>

<h2>1. 무엇을 비교했나</h2>
<p>데이터 분할만 바꾼 <b>세 설정</b>에서 같은 모델들을 같은 방식으로 비교했다.</p>
<table>
<tr><th class="l">설정</th><th class="l">데이터 분할</th><th>평가 분자</th></tr>
<tr><td class="l">① 누수 있음</td><td class="l">MoleculeACE 분할 그대로. 한 타깃의 평가 분자가 다른 타깃에서는 학습 분자일 수 있다</td><td>9,802</td></tr>
<tr><td class="l">② 누수 없음 · 우리 방식</td><td class="l">그런 분자(5,093개)를 다른 타깃의 학습·검증에서 <b>뺀다</b>. 평가 분자는 MoleculeACE 그대로</td><td>9,802</td></tr>
<tr><td class="l">③ 누수 없음 · 팀원 방식</td><td class="l">어느 타깃에서든 평가인 분자는 모든 타깃에서 <b>평가로 옮긴다</b>. 평가 비율 약 31%</td><td>14,895</td></tr>
</table>
<p><b>모든 설정에서 같은 것</b></p>
<ul>
<li>쌍 모델(ECFP + MLP, GINE, Mole-BERT)은 30개 타깃을 한 모델로, 유사 쌍 전부(ECFP4 유사도 ≥ 0.5)로 학습</li>
<li>분자별 모델은 같은 인코더로 분자 하나씩 예측한 뒤 뺀다. SVR은 ECFP + 타니모토 커널, 타깃마다 학습</li>
<li>평가: 한 군데만 다른 평가 쌍. <b>평가 RMSE = 아는 분자 A의 실측 + 예측 Δ로 만든 B와 B 실측의 차이</b>. cliff = |Δ| ≥ 1</li>
<li>신경망은 시드 3개 예측의 평균(앙상블). ECFP + MLP는 MoleculeACE의 MLP 설정(폭 1,024, 2층, 학습률 5e-6, 묶음 32)</li>
</ul>
<p class="note">설정마다 평가 쌍이 달라서 숫자는 같은 표 안에서만 비교한다. 굵은 값 = 그 표에서 가장 좋은 값.</p>

<h2>2. 결과</h2>
<img src="{D}/fig_cliff.png" width="640" height="269">
{table(1)}{table(2)}{table(3)}

<h2>3. 알게 된 것</h2>
<ul>
<li><b>누수가 그래프 쌍 모델을 부풀렸다.</b> 누수가 있으면 Mole-BERT 쌍이 cliff에서 SVR을 이기지만(0.893 vs 0.928),
누수를 없애면 두 방식 모두에서 SVR에 진다(②: 0.992 vs 0.914, ③: 1.057 vs 0.977). 두 방식의 결론이 같다.</li>
<li><b>같은 인코더라면 쌍 입력이 분자별보다 확실히 낫다.</b> 세 설정 모두, GINE·Mole-BERT 모두에서 cliff 평가 RMSE가 0.17–0.31 낮아진다 (아래 표).
ECFP에서도 같은 경향이다(타깃별 학습, 누수 실험 공통 평가 쌍 기준 분자별 MLP 1.084 → 쌍 MLP 0.939).</li>
<li><b>ECFP + MLP 쌍이 cliff에서 가장 좋다.</b> 누수를 없애도(②) SVR보다 조금 낫다(0.898 vs 0.914). 차이는 작다.
MoleculeACE 논문도 성능 차이는 알고리즘보다 표현(ECFP)에서 온다고 보고했다.</li>
<li><b>non-cliff와 전체 RMSE는 모든 설정에서 SVR이 가장 좋다.</b></li>
</ul>
<table><tr><th class="l">cliff 평가 RMSE: 분자별 → 쌍</th><th>GINE</th><th>Mole-BERT</th></tr>
<tr><td class="l">① 누수 있음</td>{gain(1)}</tr><tr><td class="l">② 누수 없음 · 우리 방식</td>{gain(2)}</tr><tr><td class="l">③ 누수 없음 · 팀원 방식</td>{gain(3)}</tr></table>

<h2>4. 다음 방향 (제안)</h2>
<p><b>주장: "GNN은 쌍으로 넣어야 cliff를 잘 맞힌다" + XAI로 이유 설명</b></p>
<ul>
<li>비교 상대를 같은 GNN의 분자별 예측으로 두면 표현(ECFP vs 그래프) 논란 없이 주장이 하나로 정리된다. 위 표처럼 세 설정 모두에서 성립한다.</li>
<li><b>XAI (진행 중, ② 설정):</b> Integrated Gradients로 예측 Δ에 대한 원자별 기여를 구해,
쌍 모델이 분자별 모델보다 <b>바뀐 부분</b>에 더 집중하는지 숫자(바뀐 부분 집중도)와 그림으로 비교한다.
공통 부분 원자는 A와 B의 짝 원자 기여를 더해 순 기여로 본다.</li>
<li>SVR·ECFP 결과는 숨기지 않고 "누수가 없으면 그래프 모델은 아직 지문 기반 방법보다 cliff를 못 맞힌다"는 한계로 함께 적는다.</li>
</ul>
<p><b>함께 정할 것</b></p>
<ul>
<li>최종 보고에 쓸 누수 처리: ② 우리 방식(평가 분자 유지, 학습에서 제거)과 ③ 팀원 방식(평가로 이동) 중 무엇을 기본으로 할지.
②는 MoleculeACE·기존 결과와 같은 평가 분자로 비교할 수 있고, ③은 데이터를 버리지 않는다</li>
<li>ECFP 쌍 모델을 결과에 넣을지, 넣는다면 GNN 주장과 어떻게 함께 보여 줄지</li>
<li>XAI 결과가 기대와 다르게 나오면(집중도 차이 없음) 주장을 어떻게 바꿀지</li>
</ul>
</body></html>'''
open(f'{D}/report.html', 'w').write(html)
subprocess.run(['soffice', '--headless', '--infilter=HTML (StarWriter)', '--convert-to', 'pdf:writer_pdf_Export', '--outdir', D, f'{D}/report.html'],
               check=True, capture_output=True, env={**os.environ, 'HOME': '/home/bising0922/Capstone/runs/tmp'})
os.replace(f'{D}/report.pdf', f'{OUT}/누수_비교_보고서.pdf'); print('ok', f'{OUT}/누수_비교_보고서.pdf')
