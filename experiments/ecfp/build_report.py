# 팀원 보고용 PDF: 네 설정 결과 + 논문 쓰는 방향
#  ① 누수 있음 ② 누수 없음·팀원 방식 ③ 누수 없음·우리 방식 ④ ③ + 큰 헤드 (그래프 쌍 모델 헤드를 ECFP + MLP와 같게)
import os, glob, subprocess, datetime, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = '/home/bising0922/Capstone/runs/tmp/ecfp'; H = '/home/bising0922/Capstone/runs/tmp/head'; OUT = '/home/bising0922/Capstone'
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
R = pd.read_pickle(f'{D}/report3.pkl')   # 설정 1(누수 있음) 2(우리) 3(팀원) — report3.py 번호
# 설정 ④: ③(우리 방식) 표에 큰 헤드 그래프 쌍 모델을 더한다 (SVR·ECFP·분자별은 같은 데이터라 그대로)
S4 = R[R.설정 == 2].copy(); n_seed = {}
for kind, pre, name in [('gine', 'pred_gnn', 'GINE 쌍'), ('mb', 'pred_pt', 'Mole-BERT 쌍')]:
    fs = sorted(glob.glob(f'{H}/pred_head_{kind}_s?.pkl')); n_seed[name] = len(fs)
    ps = [pd.read_pickle(f) for f in fs]; d = ps[0]
    ev = ((d.split == 'test') & (d.reason == 'pass')).values; y = d.delta.values[ev]; cl = np.abs(y) >= 1
    e = np.mean([p[[c for c in p.columns if c.startswith(pre)][0]].values[ev] for p in ps], 0)
    S4.loc[name] = [rm(e[cl] - y[cl]), rm(e[~cl] - y[~cl]), rm(e - y), np.nan, 4]
SETS = {1: R[R.설정 == 1], 3: R[R.설정 == 3], 2: R[R.설정 == 2], 4: S4}
NAME = {1: '① 누수 있음', 3: '② 누수 없음 · 팀원 방식', 2: '③ 누수 없음 · 우리 방식', 4: '④ ③ + 큰 헤드'}
NPAIR = {1: '43,896', 3: '52,900', 2: '38,711', 4: '38,711'}
ORDER = ['SVR', 'ECFP + MLP 쌍', 'GINE 분자별', 'GINE 쌍', 'Mole-BERT 분자별', 'Mole-BERT 쌍']
PAIR = ['ECFP + MLP 쌍', 'GINE 쌍', 'Mole-BERT 쌍']
g = lambda s, m, c: SETS[s].loc[m, c]

# 그림: 네 설정의 cliff / 전체
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
COL = {'SVR': '#eb6834', 'ECFP + MLP 쌍': '#eda100', 'GINE 분자별': '#9fd9c4', 'GINE 쌍': '#1baf7a', 'Mole-BERT 분자별': '#a9c8f0', 'Mole-BERT 쌍': '#2a78d6'}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
for ax, metric, ylim in [(axes[0], 'cliff', 1.6), (axes[1], '전체', 0.95)]:
    w = 0.13
    for k, m in enumerate(ORDER):
        vals = [g(s, m, metric) for s in [1, 3, 2, 4]]
        x = np.arange(4) + (k - 2.5) * w
        ax.bar(x, vals, w * 0.92, color=COL[m], label=m, zorder=2)
    ax.set_xticks(range(4)); ax.set_xticklabels(['① 누수 있음', '② 팀원 방식', '③ 우리 방식', '④ ③+큰 헤드'], fontsize=8.5)
    ax.set_ylabel(f'{metric} 평가 RMSE (낮을수록 좋음)', fontsize=9); ax.set_ylim(0, ylim)
    ax.grid(axis='y', alpha=0.3, zorder=0); ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(ncol=2, fontsize=7.5, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(f'{D}/fig_report.png', dpi=200); plt.close(fig)

def table(s):
    t = SETS[s]; best = {c: t[c].min() for c in ['cliff', 'non-cliff', '전체']}
    pbest = {c: t.loc[PAIR, c].min() for c in ['cliff', 'non-cliff', '전체']}
    rows = ''
    for m in ORDER:
        cells = ''
        for c in ['cliff', 'non-cliff', '전체']:
            v = f'{t.loc[m, c]:.3f}'
            if abs(t.loc[m, c] - best[c]) < 1e-9: v = f'<b>{v}</b>'
            if m in PAIR and abs(t.loc[m, c] - pbest[c]) < 1e-9: v = f'<u>{v}</u>'
            cells += f'<td>{v}</td>'
        tag = ' (큰 헤드)' if s == 4 and m in ('GINE 쌍', 'Mole-BERT 쌍') else ''
        rows += f'<tr><td class="l">{m}{tag}</td>{cells}</tr>'
    return f'<p class="cap">{NAME[s]} — 한 군데만 다른 평가 쌍 {NPAIR[s]}개</p><table><tr><th class="l">모델</th><th>cliff</th><th>non-cliff</th><th>전체</th></tr>{rows}</table>'

mb_note = '' if n_seed['Mole-BERT 쌍'] == 3 else f" (Mole-BERT 큰 헤드는 시드 {n_seed['Mole-BERT 쌍']}개 — 마지막 시드 학습 중, 끝나면 이 문서가 자동으로 갱신된다)"
gain = lambda s, enc: (g(s, f'{enc} 분자별', 'cliff'), g(s, f'{enc} 쌍', 'cliff'))
G4, M4, E4, V4 = [S4.loc[m] for m in ['GINE 쌍', 'Mole-BERT 쌍', 'ECFP + MLP 쌍', 'SVR']]
today = datetime.date.today().strftime('%Y-%m-%d')
html = f'''<html><head><meta charset="utf-8"><style>
body {{ font-family: 'Noto Sans CJK KR'; font-size: 10pt; }}
h1 {{ font-size: 16pt; }} h2 {{ font-size: 12.5pt; margin-top: 14pt; }} h3 {{ font-size: 11pt; margin-top: 10pt; }}
table {{ border-collapse: collapse; margin: 4pt 0 10pt 0; }}
th, td {{ border: 0.5pt solid #999999; padding: 3pt 8pt; text-align: center; font-size: 9.5pt; }}
th {{ background: #eeeeee; }} .l {{ text-align: left; }} .cap {{ font-weight: bold; margin: 8pt 0 2pt 0; }}
.note {{ color: #555; font-size: 9pt; }}
</style></head><body>
<h1>결과 보고: 누수 처리와 예측 헤드 수정 후 모델 비교</h1>
<p class="note">{today} · 박윤정 · 팀 공유{mb_note}</p>

<h2>요약</h2>
<ul>
<li><b>누수를 없애면 그래프 쌍 모델의 cliff 성능이 떨어진다.</b> 팀원 방식·우리 방식 모두 같은 결론이다.</li>
<li><b>그래프 쌍 모델의 예측 헤드가 너무 작았다</b>(폭 64). ECFP + MLP와 같은 크기(1,024 → 1,024 → 1)로 바꾸자
GINE 쌍 cliff가 {g(2, 'GINE 쌍', 'cliff'):.3f} → <b>{G4.cliff:.3f}</b>, 전체가 {g(2, 'GINE 쌍', '전체'):.3f} → <b>{G4['전체']:.3f}</b>로 좋아졌다. 이 수정본을 기본으로 쓴다.</li>
<li>수정 후(④) <b>Mole-BERT 쌍이 cliff에서 가장 낮고</b>({M4.cliff:.3f}; ECFP + MLP 쌍 {E4.cliff:.3f}, ECFP + SVR {V4.cliff:.3f}), <b>전체 오차는 ECFP + SVR과 대등</b>하다({M4['전체']:.3f} vs {V4['전체']:.3f}). non-cliff는 ECFP + SVR이 조금 낮다.</li>
<li>GINE 쌍도 같은 쌍 방식의 ECFP + MLP보다 non-cliff·전체 오차가 낮다. 두 그래프 모델 모두 지문 기반 쌍 모델보다 균형이 좋다.</li>
</ul>

<h2>1. 실험 조건</h2>
<table>
<tr><th class="l">설정</th><th class="l">내용</th><th>평가 분자</th></tr>
<tr><td class="l">① 누수 있음</td><td class="l">MoleculeACE 분할 그대로. 한 타깃의 평가 분자가 다른 타깃에서는 학습 분자일 수 있다</td><td>9,802</td></tr>
<tr><td class="l">② 누수 없음 · 팀원 방식</td><td class="l">어느 타깃에서든 평가인 분자는 모든 타깃에서 평가로 옮긴다</td><td>14,895</td></tr>
<tr><td class="l">③ 누수 없음 · 우리 방식</td><td class="l">그런 분자(5,093개)를 다른 타깃의 학습·검증에서 뺀다. 평가 분자는 MoleculeACE 그대로</td><td>9,802</td></tr>
<tr><td class="l">④ ③ + 큰 헤드</td><td class="l">③과 같고, GINE·Mole-BERT 쌍 모델의 예측 헤드만 ECFP + MLP와 같게(1,024 → 1,024 → 1, 드롭아웃 0)</td><td>9,802</td></tr>
</table>
<ul>
<li>쌍 모델(ECFP + MLP, GINE, Mole-BERT): 30개 타깃을 한 모델로, 유사 쌍 전부로 학습. 시드 3개 예측의 평균(앙상블)</li>
<li>분자별 모델: 같은 인코더로 분자 하나씩 예측한 뒤 뺀다. ECFP + SVR은 타깃마다 학습(MoleculeACE 표준 방법)</li>
<li>평가: 한 군데만 다른 평가 쌍. <b>평가 RMSE = 아는 분자 A의 실측 + 예측 Δ로 만든 B와 B 실측의 차이</b>. cliff = |Δ| ≥ 1</li>
</ul>
<p class="note">설정마다 평가 쌍이 달라서 숫자는 같은 표 안에서만 비교한다. <b>굵게</b> = 그 표 전체에서 가장 좋은 값, <u>밑줄</u> = 쌍 모델 중 가장 좋은 값.</p>

<h2>2. 결과</h2>
<img src="{D}/fig_report.png" width="660" height="234">
{table(1)}{table(3)}{table(2)}{table(4)}

<h2>3. 알게 된 것</h2>
<ul>
<li><b>누수가 그래프 쌍 모델을 부풀렸다.</b> ①에서는 Mole-BERT 쌍이 cliff에서 SVR을 이기지만({g(1, 'Mole-BERT 쌍', 'cliff'):.3f} vs {g(1, 'SVR', 'cliff'):.3f}),
②·③에서는 진다({g(3, 'Mole-BERT 쌍', 'cliff'):.3f} vs {g(3, 'SVR', 'cliff'):.3f}, {g(2, 'Mole-BERT 쌍', 'cliff'):.3f} vs {g(2, 'SVR', 'cliff'):.3f}). 누수 처리 방식이 달라도 결론은 같다.</li>
<li><b>쌍 입력은 같은 인코더의 분자별 예측보다 확실히 낫다.</b> 모든 설정에서 cliff가 크게 좋아진다 (④: GINE {gain(4, 'GINE')[0]:.3f} → {gain(4, 'GINE')[1]:.3f}, Mole-BERT {gain(4, 'Mole-BERT')[0]:.3f} → {gain(4, 'Mole-BERT')[1]:.3f}).</li>
<li><b>헤드 크기가 병목이었다.</b> 인코더가 뽑은 정보를 폭 64 헤드가 다 쓰지 못했다. 이전에 Mole-BERT가 GINE보다 나았던 것도 인코더 출력이 커서(1,566칸 vs 356칸) 이 병목을 덜 받았기 때문으로 보인다.</li>
<li><b>같은 쌍 방식에서 그래프 표현은 지문보다 non-cliff·전체가 낫다</b> (④ 전체: GINE {G4['전체']:.3f}, ECFP + MLP {E4['전체']:.3f}). cliff는 ECFP + MLP가 {G4.cliff - E4.cliff:.3f} 낮다.</li>
<li><b>사전학습 인코더(Mole-BERT)가 가장 좋았다.</b> 큰 헤드에서 cliff {M4.cliff:.3f}로 모든 모델 중 가장 낮고, 전체 {M4['전체']:.3f}로 ECFP + SVR({V4['전체']:.3f})과 대등하다. 다만 ECFP + MLP 쌍과의 cliff 차이({E4.cliff - M4.cliff:.3f})는 시드 흔들림보다 작다.</li>
<li><b>ECFP + SVR은 non-cliff에서 여전히 가장 낮다</b> ({V4['non-cliff']:.3f} vs Mole-BERT {M4['non-cliff']:.3f}).</li>
</ul>

<h2>4. 논문을 쓸 때 장점이 잘 보이는 구성 (제안)</h2>
<h3>주장 (하나로)</h3>
<p><b>"activity cliff의 활성 차이를 예측할 때, GNN은 분자를 하나씩 보는 대신 쌍으로 넣어야 한다.
사전학습 GNN(Mole-BERT)을 쌍으로 넣으면 누수를 통제한 평가에서 cliff 오차가 가장 낮고,
전체 오차는 분야 표준(ECFP + SVR)과 대등하다."</b></p>
<h3>표·그림 구성</h3>
<ol>
<li><b>표 1 (주 결과, ④ 설정):</b> 비교 대상을 역할별로 묶는다 — 분야 표준(ECFP + SVR) / 같은 조건 대조군(ECFP + MLP 쌍) / 우리(Mole-BERT 쌍, GINE 쌍).
cliff·non-cliff·전체를 모두 싣고, 쌍 모델 중 최고를 표시한다</li>
<li><b>표 2 (쌍 입력의 효과):</b> 같은 인코더의 분자별 → 쌍. 감소율(%)로 보여 주면 효과가 크게 보인다 (cliff: GINE {(1 - gain(4, 'GINE')[1] / gain(4, 'GINE')[0]) * 100:.0f}%, Mole-BERT {(1 - gain(4, 'Mole-BERT')[1] / gain(4, 'Mole-BERT')[0]) * 100:.0f}% 감소)</li>
<li><b>표 3 (헤드 크기 분석):</b> 작은 헤드 → 큰 헤드. "공정한 비교 조건을 맞추면 그래프가 따라잡는다"는 근거</li>
<li><b>표 4 (누수 분석):</b> ①·②·③ 비교. "누수를 통제한 공정한 평가"를 했다는 신뢰도 근거. 누수가 성능을 부풀릴 수 있음을 보인 것 자체도 기여</li>
<li><b>그림:</b> 세 지표를 한 번에 보는 막대 또는 레이더 그림. Mole-BERT 쌍이 cliff는 최고, 전체는 최고와 대등하다는 "균형"을 보여 준다</li>
</ol>
<h3>쓸 수 있는 문장 / 피해야 할 문장</h3>
<table>
<tr><th class="l">쓸 수 있다</th><th class="l">피해야 한다</th></tr>
<tr><td class="l">쌍 입력이 같은 GNN의 cliff 오차를 크게 줄인다</td><td class="l">모든 지표에서 SVR을 이긴다 (non-cliff는 SVR이 낮음)</td></tr>
<tr><td class="l">같은 쌍 조건에서 그래프 표현은 지문보다 전체 오차가 낮다</td><td class="l">그래프가 모든 지표에서 지문보다 낫다 (cliff는 ECFP 쌍이 낮음)</td></tr>
<tr><td class="l">사전학습 GNN 쌍 모델이 누수 없는 평가에서 cliff 오차가 가장 낮다</td><td class="l">ECFP 쌍보다 cliff가 확실히 낫다 (차이가 시드 흔들림 안)</td></tr>
<tr><td class="l">누수를 통제하면 그래프 모델의 성능이 과대평가되어 있었음이 드러난다</td><td class="l">누수 있는 설정의 숫자를 주 결과로 쓰기</td></tr>
</table>
<h3>한계로 적을 것</h3>
<ul>
<li>ECFP + SVR은 non-cliff에서 여전히 가장 낮다</li>
<li>Mole-BERT와 ECFP + MLP 쌍의 cliff 차이는 작다 (시드 흔들림 안)</li>
<li>헤드 크기 외의 학습 설정(학습률·묶음 크기)은 모델마다 다르다</li>
</ul>

<h2>5. 함께 정할 것</h2>
<ul>
<li>최종 보고의 기본 설정: ③(평가 분자 유지) vs ②(평가로 이동). ④는 지금 ③ 기준이라, ②를 기본으로 하면 큰 헤드로 다시 학습해야 한다(약 2–4시간)</li>
<li>주 모델: Mole-BERT 쌍(성능 최고)으로 할지, GINE 쌍(작고 빠름)도 함께 둘지</li>
<li>노트북 기본값을 큰 헤드로 바꾸고, 작은 헤드 결과는 비교 기록으로 남기기</li>
</ul>
</body></html>'''
open(f'{D}/report2.html', 'w').write(html)
subprocess.run(['soffice', '--headless', '--infilter=HTML (StarWriter)', '--convert-to', 'pdf:writer_pdf_Export', '--outdir', D, f'{D}/report2.html'],
               check=True, capture_output=True, env={**os.environ, 'HOME': '/home/bising0922/Capstone/runs/tmp',
                                                     'TMPDIR': '/home/bising0922/Capstone/runs/tmp', 'TMP': '/home/bising0922/Capstone/runs/tmp'})
os.replace(f'{D}/report2.pdf', f'{OUT}/결과_보고_헤드수정.pdf')
open('/home/bising0922/Capstone/runs/_diag.txt', 'w').write(f'ok 시드 {n_seed}\n' + S4.round(3).to_string())
