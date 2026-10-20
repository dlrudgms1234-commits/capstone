"""학습 곡선 (시드별 작은 그림): 위 = 학습 쌍 vs 검증 쌍 전체 RMSE, 아래 = 검증 cliff vs non-cliff RMSE. 세로 점선 = 최적 에폭
사용: python plot_curves.py <hist 파일 패턴(시드 자리에 {s})> <제목> <저장 경로> [시드 목록, 예: 0,1,2,3,4 (기본)]"""
import sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

pat, title, out = sys.argv[1:4]
SEEDS = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else list(range(5))
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 9})
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#2b2b2b', '#6b6b6b', '#e6e6e6'
H = {s: pd.read_csv(pat.format(s=s)) for s in SEEDS}
fig, ax = plt.subplots(2, len(H), figsize=(3 * len(H) + 1, 6), sharex='col', sharey='row', squeeze=False)
rows = [(('train', 'train 쌍', BLUE, '-'), ('val', 'valid 쌍 전체', ORANGE, '--')),
        (('val_cliff', 'valid cliff', BLUE, '-'), ('val_noncliff', 'valid non-cliff', ORANGE, '--'))]
for k, (s, d) in enumerate(H.items()):
    b = int(d.val.idxmin())
    for r, series in enumerate(rows):
        a = ax[r, k]
        for col, lab, c, ls in series:
            a.plot(d.epoch, d[col], color=c, ls=ls, lw=2, label=lab)
            if k == 0:   # 첫 칸에만 선 끝에 이름
                a.annotate(lab, (d.epoch.iloc[-1], d[col].iloc[-1]), xytext=(4, 0), textcoords='offset points', color=INK, fontsize=8, va='center')
        a.axvline(b, color=MUTED, ls=':', lw=1)
        a.grid(axis='y', color=GRID, lw=0.6); a.set_axisbelow(True)
        a.spines[['top', 'right']].set_visible(False); a.spines[['left', 'bottom']].set_color('#bdbdbd'); a.tick_params(colors=MUTED, length=3)
        if r == 0: a.set_title(f'시드 {s}  (최적 {b}에폭, valid {d.val[b]:.3f})', fontsize=9, color=INK, loc='left')
        if r == 1: a.set_xlabel('에폭', color=MUTED)
ax[0, 0].set_ylabel('RMSE (pKi 단위)', color=MUTED); ax[1, 0].set_ylabel('RMSE (pKi 단위)', color=MUTED)
for r in range(2):
    ax[r, len(H) - 1].legend(frameon=False, fontsize=8, loc='upper right' if r == 0 else 'center right')
fig.suptitle(title, x=0.01, ha='left', fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(out, dpi=110); print(out)
