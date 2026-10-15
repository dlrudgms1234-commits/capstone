# 논문용 그림 만들기 (학술지·학회 스타일: 흑백 위주 + 강조 한 색, 9pt 이상, 벡터 PDF)
# 입력: eval_pairs_predictions_leakfree.csv (같은 폴더)
# 출력: fig_target_diff.pdf/.png, fig_target_scatter.pdf/.png, fig_main_bars.pdf/.png
# 실행: python make_figures.py
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

D = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(D, 'eval_pairs_predictions_leakfree.csv'))

# ── 스타일: 한글 글꼴은 있는 것 중 첫 번째 (서버 / 윈도 / 맥 / 나눔)
have = {f.name for f in font_manager.fontManager.ttflist}
KFONT = next((f for f in ['Noto Sans CJK KR', 'Noto Sans CJK JP', 'Malgun Gothic', 'AppleGothic', 'NanumGothic'] if f in have), 'DejaVu Sans')
plt.rcParams.update({
    'font.family': KFONT, 'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'axes.unicode_minus': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
MM = 1 / 25.4
COL_W = 88 * MM            # 2단 원고의 한 단 폭 (A4, 좌우 여백 10mm 기준 약 90mm)
OURS, BASE, LIGHT = '#1f3b73', '0.35', '0.75'   # 강조(우리 모델) 남색 한 가지 + 회색 계열
def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(D, f'{name}.{ext}'), dpi=600)
    plt.close(fig)

ens = lambda name: d['svr'].values if name == 'svr' else d[[f'{name}_s{s}' for s in range(3)]].mean(1).values
y, cl = d.delta.values, d.is_cliff.values.astype(bool)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def per_target(name, mask):
    e = (ens(name) - y) ** 2
    return pd.Series(e[mask]).groupby(d.target.values[mask]).mean() ** 0.5

# ── 그림 A: 타깃별 cliff RMSE 차이 (분자별 − 쌍), 두 인코더
fig, axes = plt.subplots(2, 1, figsize=(COL_W, 70 * MM), sharex=False)
for ax, enc, label in [(axes[0], 'molebert', 'Mole-BERT'), (axes[1], 'gine', 'GINE')]:
    diff = (per_target(f'{enc}_mol', cl) - per_target(f'{enc}_pair', cl)).sort_values(ascending=False)
    ax.bar(range(len(diff)), diff.values, 0.75, color=[OURS if v > 0 else LIGHT for v in diff.values], linewidth=0)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks([]); ax.set_xlim(-0.7, len(diff) - 0.3)
    ax.set_ylabel('ΔRMSE')
    ax.text(0.99, 0.95, f'{label}: {int((diff > 0).sum())}/{len(diff)} 타깃 개선', transform=ax.transAxes, ha='right', va='top')
axes[1].set_xlabel('타깃 (차이 순)')
fig.tight_layout(h_pad=0.6)
save(fig, 'fig_target_diff')

# ── 그림 B: 타깃별 산점도 (x = 분자별, y = 쌍), 대각선 아래 = 쌍이 더 정확
fig, ax = plt.subplots(figsize=(COL_W, 70 * MM))
for enc, label, mk, fc in [('molebert', 'Mole-BERT', 'o', OURS), ('gine', 'GINE', 's', 'white')]:
    a, b = per_target(f'{enc}_mol', cl), per_target(f'{enc}_pair', cl)
    ax.scatter(a, b.loc[a.index], s=16, marker=mk, facecolor=fc, edgecolor=OURS if fc == 'white' else 'none', linewidth=0.7, label=label, zorder=3)
lo, hi = 0.5, 1.8
ax.plot([lo, hi], [lo, hi], color='0.5', lw=0.6, ls='--', zorder=1)
ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='분자별 입력 cliff RMSE', ylabel='쌍 입력 cliff RMSE')
ax.set_aspect('equal'); ax.legend(frameon=False, loc='upper left', handletextpad=0.2)
save(fig, 'fig_target_scatter')

# ── 그림 C: 주 결과 막대 (cliff / 전체), 우리 모델만 강조
rows = [('ECFP + SVR', 'svr', BASE, ''), ('ECFP + MLP 쌍', 'ecfp_mlp_pair', LIGHT, ''),
        ('GINE 분자별', 'gine_mol', LIGHT, '////'), ('GINE 쌍', 'gine_pair', OURS, ''),
        ('Mole-BERT 분자별', 'molebert_mol', LIGHT, '////'), ('Mole-BERT 쌍', 'molebert_pair', OURS, '')]
fig, axes = plt.subplots(1, 2, figsize=(2 * COL_W, 55 * MM))
for ax, (title, mask) in zip(axes, [('cliff', cl), ('전체', np.ones_like(cl))]):
    vals = [rm(ens(k)[mask] - y[mask]) for _, k, _, _ in rows]
    bars = ax.bar(range(len(rows)), vals, 0.7, color=[c for *_, c, _ in rows], hatch=None, edgecolor='black', linewidth=0.5)
    for b, (_, _, _, h) in zip(bars, rows):
        if h: b.set_hatch(h); b.set_facecolor('white')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(vals[0], color='black', lw=0.6, ls='--')        # ECFP + SVR 기준선
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=30, ha='right')
    ax.set_ylabel(f'{title} RMSE'); ax.set_ylim(0, max(vals) * 1.15)
fig.tight_layout(w_pad=1.5)
save(fig, 'fig_main_bars')
print('완료: fig_target_diff, fig_target_scatter, fig_main_bars (.pdf, .png) / 글꼴', KFONT)
