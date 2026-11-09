"""논문 그림 1–5 생성. 저장소 최상위에서 실행한다
  python experiments/figs/make_figs.py [출력폴더]
그림 1–3은 논문_그림_코드(KSC2026 새 원고)의 코드를 옮겨 한 파일로 합친 것이다
  - 그림 1 데이터 구성·누수 통제: Sankey (fig_data_options.py opt3)
  - 그림 2 쌍 입력 구조: fig1.py
  - 그림 3 단일 입력 구조: fig_single_arch.py
양식: 1단 폭 88 mm, 글자 9pt, 600 dpi, 무채색 + 남색 (style.py). 글꼴은 함초롬돋움이 없으면 대체 글꼴을 쓴다
숫자는 test (한 군데만 다른 test 쌍 38,711개), 시드 5개 평균 ± 표준편차
그림 2·3의 예시 쌍은 data/moleculeace/CHEMBL287_Ki.csv에서 읽는다 (없으면 두 그림만 건너뛴다)
"""
import io, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon, Patch

OUT = sys.argv[1] if len(sys.argv) > 1 else 'figs'
os.makedirs(OUT, exist_ok=True)

# ── 공통 양식 (논문_그림_코드/capstone/논문/src/style.py) ────────────────────
DPI = 600
INK, GRAY, LIGHTC, ACCENT = '#000000', '#7a7a7a', '#c4c4c4', '#1f4e79'
FONTS = ['HCR Dotum', '함초롬돋움', 'HCRDotum']
FALLBACK = ['Apple SD Gothic Neo', 'AppleGothic', 'Noto Sans CJK KR', 'Noto Sans CJK JP', 'NanumGothic']


def _pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for f in FONTS:
        if f in have:
            return f
    for f in FALLBACK:
        if f in have:
            warnings.warn(f'FONT-FALLBACK: 함초롬돋움이 없어 {f}로 그림 (최종본 아님)')
            return f
    raise RuntimeError('한글 글꼴을 찾을 수 없음')


def style_apply():
    font = _pick_font()
    plt.rcParams.update({
        'font.family': font, 'font.size': 7, 'axes.unicode_minus': True,
        'mathtext.fontset': 'custom', 'mathtext.rm': font, 'mathtext.it': font, 'mathtext.bf': font,
        'axes.linewidth': 0.6, 'axes.labelsize': 7, 'axes.titlesize': 7,
        'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'axes.axisbelow': True,
        'axes.prop_cycle': matplotlib.cycler(color=[INK, GRAY, ACCENT]),
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'lines.linewidth': 1.0, 'lines.markersize': 3.5,
        'legend.fontsize': 6.5, 'legend.frameon': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'savefig.dpi': DPI, 'savefig.bbox': 'standard',
        'pdf.fonttype': 42, 'svg.fonttype': 'path',
    })


# ── 그림 1–3 공통 값 ───────────────────────────────────────────────────────
NAVY = '#1f3b73'
FS = 9                                     # 모든 글자 최소 크기 (pt)
LW = 1.0
FILL = '#efefef'                           # 부가 정보 (옅은 회색)
NAVY_L = '#e1e8f4'                         # 옅은 남색
MOL_FILL, CHG_FILL, AUX_FILL = '#e3eee6', '#e1e8f4', '#efefef'
ATOM_HL = '#c9d6ec'
W_MM = 88.0                                # KSC 1단 폭
W_PX = round(W_MM / 25.4 * DPI)
W = W_PX / DPI * 72                        # 좌표 폭 (pt)

# 데이터 구성 수치 (논문_그림_코드/capstone/논문/stats/descriptive.json)
TOTAL = 48714
TRAIN, VAL, TEST_N, DROP = 27061, 6758, 9802, 5093          # 통제 후
TRAIN0, VAL0 = 31144, 7768                                   # 통제 전 학습·검증
DROP_TR, DROP_VAL = 4083, 1010                               # 각각에서 제거되는 행
TRVAL = TOTAL - TEST_N
LEAK = 3884                                                  # 다른 태스크의 학습·검증에도 있는 평가 행
NC, NN = 8778, 29933
PCT_LEAK, PCT_DROP = LEAK / TEST_N, DROP / TRVAL
assert TRAIN + VAL + DROP == TRVAL and TRAIN0 + VAL0 == TRVAL
assert TRAIN0 - DROP_TR == TRAIN and VAL0 - DROP_VAL == VAL and DROP_TR + DROP_VAL == DROP


def T(ax, x, y, s, c=INK, fs=FS, **kw):
    kw.setdefault('ha', 'center')
    kw.setdefault('va', 'center')
    return ax.text(x, y, s, fontsize=fs, color=c, zorder=5, linespacing=1.05, **kw)


def R(ax, x, y, w, h, fc='white', ec=INK, hatch=None, lw=LW, z=2, round_=False):
    if round_:
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=4',
                           fc=fc, ec=ec, lw=lw, hatch=hatch, zorder=z)
    else:
        p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, hatch=hatch, zorder=z,
                      joinstyle='miter')
    ax.add_patch(p)


def line(ax, xs, ys, c=INK, lw=LW):
    ax.plot(xs, ys, color=c, lw=lw, solid_capstyle='butt', solid_joinstyle='miter', zorder=0)


def arrow(ax, x0, y0, x1, y1, c=INK, lw=LW):
    ax.annotate('', (x1, y1), (x0, y0), zorder=4, arrowprops=dict(
        arrowstyle='-|>,head_width=0.25,head_length=0.5', color=c, lw=lw, shrinkA=0, shrinkB=0))


def check(fig, name):
    """글자 겹침·잘림 검사"""
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    ts = [t for ax in fig.axes for t in ax.texts if t.get_text().strip()]
    bb = [t.get_window_extent(rd) for t in ts]
    fb = fig.bbox
    for t, b in zip(ts, bb):
        if not (b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fb.x1 + 1 and b.y1 <= fb.y1 + 1):
            print('OUT OF RANGE', name, repr(t.get_text()))
    for i in range(len(bb)):
        for j in range(i + 1, len(bb)):
            if bb[i].overlaps(bb[j]):
                print('OVERLAP', name, repr(ts[i].get_text()), repr(ts[j].get_text()))


# ── 그림 1: 데이터 구성과 태스크 간 누수 통제 (Sankey) ───────────────────────
def band(ax, xa, ya0, ya1, xb, yb0, yb1, fc, z=1, hatch=None):
    t = np.linspace(0, 1, 40)
    s = 3 * t ** 2 - 2 * t ** 3                       # 부드러운 S자
    x = xa + (xb - xa) * t
    top = ya1 + (yb1 - ya1) * s
    bot = ya0 + (yb0 - ya0) * s
    ax.add_patch(Polygon(np.r_[np.c_[x, top], np.c_[x[::-1], bot[::-1]]], closed=True,
                         fc=fc, ec='none', zorder=z, hatch=hatch, alpha=1))


def fig1():
    """Sankey: 통제 전 → 통제 후로 행이 갈라지고 빠져나간다 (fig_data_options.py opt3)"""
    style_apply()
    plt.rcParams['hatch.linewidth'] = 0.6
    plt.rcParams['hatch.color'] = GRAY
    WF = W                                # 가로 (1단 폭 전체)
    H = 168.0                             # 세로
    wpx, hpx = round(WF / 72 * DPI), round(H / 72 * DPI)
    fig = plt.figure(figsize=(wpx / DPI, hpx / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WF); ax.set_ylim(0, H); ax.axis('off')

    k = 126 / TOTAL                       # 전체 48,714행 = 126 pt 높이
    g = 4                                 # 기둥 사이 간격
    xa, xb, cw = 62, 160, 8               # 왼쪽·오른쪽 기둥 x, 기둥 폭
    top = H - 16
    T(ax, xa + 4, H - 7, '통제 전')
    T(ax, xb + 4, H - 7, '통제 후')
    # 왼쪽 기둥: 학습 / 검증 / 평가 (학습·검증 안에서 제거될 행을 빗금으로)
    l = {}
    y = top
    for name, n in [('학습', TRAIN0), ('검증', VAL0)]:
        l[name] = (y - n * k, y)
        y -= n * k + g
    l['평가'] = (y - g - TEST_N * k, y - g)
    # 오른쪽 기둥: 학습 / 검증 / 제거 / 평가
    r = {}
    y = top
    for name, n in [('학습', TRAIN), ('검증', VAL), ('제거', DROP)]:
        r[name] = (y - n * k, y)
        y -= n * k + g
    r['평가'] = (y - TEST_N * k, y)
    # 띠: 남는 행은 같은 이름으로, 제거될 행은 '제거' 기둥으로 흐른다
    BAND_KEEP, BAND_DROP, BAND_TEST = '#d9d9d9', '#b9c8e2', '#ebebeb'   # 흐름 띠 (기둥보다 옅게)
    dr = r['제거'][1]
    for name, keep, drop in [('학습', TRAIN, DROP_TR), ('검증', VAL, DROP_VAL)]:
        y0, y1 = l[name]
        band(ax, xa + cw, y1 - keep * k, y1, xb, r[name][0], r[name][1], BAND_KEEP)
        band(ax, xa + cw, y0, y0 + drop * k, xb, dr - drop * k, dr, BAND_DROP)
        dr -= drop * k
    band(ax, xa + cw, l['평가'][0], l['평가'][1], xb, r['평가'][0], r['평가'][1], BAND_TEST)
    for name, keep, drop in [('학습', TRAIN, DROP_TR), ('검증', VAL, DROP_VAL)]:
        y0, y1 = l[name]
        R(ax, xa, y0 + drop * k, cw, keep * k, fc=GRAY, ec=INK)
        R(ax, xa, y0, cw, drop * k, fc='white', ec=NAVY, hatch='////')
    R(ax, xa, l['평가'][0], cw, TEST_N * k, fc='white')
    for name, fc in [('학습', GRAY), ('검증', GRAY), ('제거', NAVY), ('평가', 'white')]:
        y0, y1 = r[name]
        R(ax, xb, y0, cw, y1 - y0, fc=fc, ec=NAVY if name == '제거' else INK)
    T(ax, xa - 3, sum(l['학습']) / 2, f'학습 {TRAIN0:,}', ha='right')
    T(ax, xa - 3, sum(l['검증']) / 2, f'검증 {VAL0:,}', ha='right')
    T(ax, xa - 3, sum(l['평가']) / 2, f'평가 {TEST_N:,}', ha='right')
    x = xb + cw + 3
    T(ax, x, sum(r['학습']) / 2, f'학습 {TRAIN:,}', ha='left')
    T(ax, x, sum(r['검증']) / 2 + 1, f'검증 {VAL:,}', ha='left')
    T(ax, x, sum(r['제거']) / 2 - 1.5, f'제거 {DROP:,}', c=NAVY, ha='left')
    T(ax, x, sum(r['평가']) / 2 - 2, f'평가 {TEST_N:,}', ha='left')

    check(fig, 'fig1')
    fig.savefig(f'{OUT}/fig1_leak.png', dpi=DPI)
    plt.close(fig)


# ── 그림 2·3 공통: 예시 쌍과 분자 그림 (fig1.py) ───────────────────────────
TARGET, J_TRAIN, I_EVAL = 'CHEMBL287_Ki', 8, 19     # A = 기준(train), B = 평가(test)
DATA = 'data/moleculeace'


def load_pair():
    import pandas as pd
    t = pd.read_csv(os.path.join(DATA, TARGET + '.csv'))
    a, b = t.iloc[J_TRAIN], t.iloc[I_EVAL]
    y = 'y [pEC50/pKi]'
    assert a.split == 'train' and b.split == 'test'
    return a.smiles, b.smiles, a[y], b[y], b[y] - a[y]


def changed_atoms(ma, mb):
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    res = rdFMCS.FindMCS([ma, mb], timeout=5, ringMatchesRingOnly=True, completeRingsOnly=True)
    core = Chem.MolFromSmarts(res.smartsString)
    ka, kb = ma.GetSubstructMatch(core), mb.GetSubstructMatch(core)
    ca = [k for k in range(ma.GetNumAtoms()) if k not in ka]
    cb = [k for k in range(mb.GetNumAtoms()) if k not in kb]
    return core, ka, kb, ca, cb


def depict(ma, mb, ka, kb):
    """A를 가로로 눕히고 B를 같은 골격 좌표에 맞춘다"""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(ma)
    c = ma.GetConformer()
    xy = np.array([[c.GetAtomPosition(k).x, c.GetAtomPosition(k).y] for k in range(ma.GetNumAtoms())])
    xy -= xy.mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    xy = xy @ vt.T
    for k, (x, y_) in enumerate(xy):
        c.SetAtomPosition(k, (float(x), float(y_), 0.0))
    cmap = {kb[n]: Chem.rdGeometry.Point2D(c.GetAtomPosition(ka[n]).x, c.GetAtomPosition(ka[n]).y)
            for n in range(len(ka))}
    rdDepictor.Compute2DCoords(mb, coordMap=cmap)


def px(pt):
    return int(round(pt / 72 * DPI))


def ink_left(img, w_pt):
    """분자 그림에서 선이 실제로 시작하는 x (pt). RDKit 캔버스 여백만큼 라벨을 들여쓰기 위함"""
    a = np.asarray(img)
    ink = (a[..., 3] > 10) if a.shape[-1] == 4 else (a[..., :3].min(-1) < 245)
    cols = np.where(ink.any(0))[0]
    return 0.0 if not len(cols) else float(cols[0]) / a.shape[1] * w_pt


def mol_img(m, hl, w_pt, h_pt):
    """ACS 1996 양식, 흑백 원자, 바뀐 원자만 옅은 남색 강조"""
    from PIL import Image
    from rdkit.Chem.Draw import rdMolDraw2D
    d = rdMolDraw2D.MolDraw2DCairo(px(w_pt), px(h_pt))
    o = d.drawOptions()
    rdMolDraw2D.SetACS1996Mode(o, rdMolDraw2D.MeanBondLength(m))
    o.useBWAtomPalette()
    o.clearBackground = False
    o.padding = 0.03
    o.fixedBondLength = -1
    o.scaleBondWidth = False
    o.bondLineWidth = px(0.8)
    o.maxFontSize = -1
    o.minFontSize = -1
    o.fixedFontSize = px(FS)
    o.multipleBondOffset = 0.18
    o.fillHighlights = True
    o.atomHighlightsAreCircles = True
    o.highlightRadius = 0.5
    rgb = matplotlib.colors.to_rgb(ATOM_HL)
    d.DrawMolecule(m, highlightAtoms=hl, highlightBonds=[], highlightAtomColors={k: rgb for k in hl})
    d.FinishDrawing()
    return np.asarray(Image.open(io.BytesIO(d.GetDrawingText())).convert('RGBA'))


# ── 그림 2: 쌍 입력 모델 구조 (fig1.py) ────────────────────────────────────
def fig2_pair():
    from rdkit import Chem
    sa, sb, ya, yb, dlt = load_pair()
    ma, mb = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    _, ka, kb, ca, cb = changed_atoms(ma, mb)
    depict(ma, mb, ka, kb)
    print(f'{TARGET}  A(기준) pKi {ya:.2f} 바뀐 원자 {ca} / B(평가) pKi {yb:.2f} 바뀐 원자 {cb} / Δ {dlt:+.3f}')

    style_apply()
    H = 140.5
    H_PX = round(H / 72 * DPI); H = H_PX / DPI * 72
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    # 1) 입력 쌍
    mw, mh = 122, 34
    mol_y = H - 13 - mh
    xs = [0.5, W - 0.5 - mw]
    for x0, (m, hl, lab) in zip(xs, [(ma, ca, 'A  기준 분자'), (mb, cb, 'B  평가 분자')]):
        img = mol_img(m, hl, mw, mh)
        ax.imshow(img, extent=(x0, x0 + mw, mol_y, mol_y + mh), interpolation='none', zorder=2)
        T(ax, x0 + ink_left(img, mw), H - 6, lab, ha='left')   # 구조가 실제로 시작하는 x에 맞춤

    # 2) 인코더 둘 (A, B가 각각 두 인코더를 모두 지난다)
    eh = 13
    ey = mol_y - 20 - eh
    bus = mol_y - 6
    ma_x, mb_x = xs[0] + mw / 2 + 12, xs[1] + mw / 2 + 12
    line(ax, [ma_x, ma_x], [mol_y + 2, bus])
    line(ax, [mb_x, mb_x], [mol_y + 2, bus])

    # 3) 이어 붙인 입력 벡터
    cells = [('h(A)', INK, 24, MOL_FILL), ('h(B)', INK, 24, MOL_FILL), ('h(B)−h(A)', INK, 46, MOL_FILL),
             ('f(A)', NAVY, 22, CHG_FILL), ('f(B)', NAVY, 22, CHG_FILL),
             ('편집 정보 6', INK, 50, AUX_FILL), ('태스크 30', INK, 46, AUX_FILL)]
    vw = sum(c[2] for c in cells)
    vx0 = (W - vw) / 2
    ch = 13
    vy = ey - 17 - ch
    x = vx0
    cx = []
    for lab, c, w, fc in cells:
        R(ax, x, vy, w, ch, ec=c, lw=LW * (1.3 if c == NAVY else 1), fc=fc)
        T(ax, x + w / 2, vy + ch / 2, lab, c=c)
        cx.append((x, x + w))
        x += w
    for (lab, c, w, _), (a0, a1) in zip(cells, cx):
        if c == NAVY:
            R(ax, a0, vy, w, ch, ec=c, lw=LW * 1.3, fc='none')

    hx = (cx[0][0] + cx[2][1]) / 2
    fx = (cx[3][0] + cx[4][1]) / 2
    ew_h, ew_f = 66, 84
    ehx = max(vx0, hx - ew_h / 2)
    efx = W - 4 - ew_f
    R(ax, ehx, ey, ew_h, eh, fc=MOL_FILL)
    T(ax, ehx + ew_h / 2, ey + eh / 2, '분자 인코더 h')
    R(ax, efx, ey, ew_f, eh, ec=NAVY, lw=LW * 1.3, fc=CHG_FILL)
    T(ax, efx + ew_f / 2, ey + eh / 2, '변화 부위 인코더 f', c=NAVY)
    line(ax, [ehx + ew_h / 2, mb_x], [bus, bus])
    line(ax, [mb_x, efx + ew_f / 2], [bus, bus], c=NAVY)
    arrow(ax, ehx + ew_h / 2, bus, ehx + ew_h / 2, ey + eh)
    arrow(ax, efx + ew_f / 2, bus, efx + ew_f / 2, ey + eh, c=NAVY)
    arrow(ax, ehx + ew_h / 2, ey, ehx + ew_h / 2, vy + ch)
    my = ey - 8
    line(ax, [efx + ew_f / 2, efx + ew_f / 2], [ey, my], c=NAVY)
    line(ax, [efx + ew_f / 2, fx], [my, my], c=NAVY)
    arrow(ax, fx, my, fx, vy + ch, c=NAVY)

    # 4) 헤드 1,024 → 1,024 → 1 → Δ̂
    hh = 13
    hy = vy - 15 - hh
    boxes = [('1,024', 32), ('1,024', 32), ('1', 16)]
    gap = 12
    lab_w = 22
    # 헤드 줄(라벨 + 상자 3개 + Δ̂)을 그림 가운데에 놓는다
    row_w = lab_w + sum(w for _, w in boxes) + gap * len(boxes) + 11
    x = (W - row_w) / 2
    T(ax, x, hy + hh / 2, '헤드', ha='left')
    x += lab_w
    first = x
    for lab, w in boxes:
        R(ax, x, hy, w, hh)
        T(ax, x + w / 2, hy + hh / 2, lab)
        x += w
        arrow(ax, x, hy + hh / 2, x + gap, hy + hh / 2)
        x += gap
    T(ax, x + 1, hy + hh / 2 + 0.5, r'$\hat{\Delta}$', fs=11, ha='left')
    by = vy - 6
    line(ax, [vx0, vx0, vx0 + vw, vx0 + vw], [vy - 1.5, by, by, vy - 1.5])
    arrow(ax, first + 16, by, first + 16, hy + hh)

    check(fig, 'fig2')
    fig.savefig(f'{OUT}/fig2_pair.png', dpi=DPI)
    plt.close(fig)


# ── 그림 3: 단일 입력 모델 구조 (fig_single_arch.py) ───────────────────────
def fig3_single():
    from rdkit import Chem
    sa, sb, *_ = load_pair()
    ma, mb = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    _, ka, kb, _, _ = changed_atoms(ma, mb)
    depict(ma, mb, ka, kb)

    style_apply()
    # 단일 입력은 분자 하나의 활성만 예측하므로 평가 분자 B 한 줄만 그린다 (기준 분자는 쓰지 않는다)
    W3 = W * 0.82
    H = 116.0
    W3_PX = round(W3 / 72 * DPI); W3 = W3_PX / DPI * 72
    H_PX = round(H / 72 * DPI); H = H_PX / DPI * 72
    fig = plt.figure(figsize=(W3_PX / DPI, H_PX / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W3); ax.set_ylim(0, H); ax.axis('off')

    mw, mh = 122, 30
    mol_y = H - 7 - mh
    eh, ch = 13, 13
    ey = mol_y - 8 - eh
    vy = ey - 10 - ch                             # [h | 태스크 30] 줄
    py = vy - 11 - ch                             # 헤드 → p̂ 줄
    ew = 60
    x0 = (W3 - mw) / 2 - 10                       # p̂(B) 라벨 자리를 오른쪽에 둔다
    cx = x0 + mw / 2
    img = mol_img(mb, [], mw, mh)
    ax.imshow(img, extent=(x0, x0 + mw, mol_y, mol_y + mh), interpolation='none', zorder=2)
    L = cx - ew / 2
    R(ax, L, ey, ew, eh, fc=MOL_FILL)
    T(ax, L + ew / 2, ey + eh / 2, '분자 인코더 h')
    arrow(ax, cx, mol_y - 0.5, cx, ey + eh)
    vw3 = 24 + 44
    x = cx - vw3 / 2
    for c, w, fc in [('h(B)', 24, MOL_FILL), ('태스크 30', 44, AUX_FILL)]:
        R(ax, x, vy, w, ch, fc=fc)
        T(ax, x + w / 2, vy + ch / 2, c)
        x += w
    arrow(ax, cx, ey, cx, vy + ch)
    boxes = [('1,024', 26), ('1,024', 26), ('1', 13)]
    gap, lab_w = 8, 20
    hw = lab_w + sum(w for _, w in boxes) + gap * len(boxes) + 11
    x = cx - hw / 2
    T(ax, x, py + ch / 2, '헤드', ha='left')
    x += lab_w
    arrow(ax, cx, vy, x + boxes[0][1] / 2, py + ch)
    for lab2, w in boxes:
        R(ax, x, py, w, ch)
        T(ax, x + w / 2, py + ch / 2, lab2)
        x += w
        arrow(ax, x, py + ch / 2, x + gap, py + ch / 2)
        x += gap
    T(ax, x + 1, py + ch / 2 + 0.5, r'$\hat{p}$(B)', ha='left')

    check(fig, 'fig3')
    fig.savefig(f'{OUT}/fig3_single.png', dpi=DPI)
    plt.close(fig)


# ── 그림 4·5: 막대그래프
EDGE = '#404040'
LIGHT = '#FFFFFF'

MODELS = ['ECFP+SVR', 'GINE', 'GraphCliff', 'Mole-BERT']
SINGLE = {'cliff': [(0.914, 0), (1.020, 0.104), (0.921, 0.049), (1.060, 0.117)],
          'all': [(0.591, 0), (0.692, 0.044), (0.639, 0.022), (0.706, 0.055)]}
PAIR = {'cliff': [(0.911, 0), (0.886, 0.004), (0.904, 0.011), (0.864, 0.005)],
        'all': [(0.586, 0), (0.593, 0.004), (0.605, 0.007), (0.581, 0.002)]}
ABL = [('전체 입력', [(0.864, 0.005), (0.467, 0.002), (0.581, 0.002)]),
       ('편집 정보 제외', [(0.864, 0.010), (0.475, 0.005), (0.586, 0.007)]),
       ('태스크 정보 제외', [(0.949, 0.012), (0.497, 0.001), (0.629, 0.004)]),
       ('편집·태스크 정보 제외', [(0.962, 0.018), (0.511, 0.006), (0.641, 0.008)])]

# ── 그림 4: 단일 입력 / 쌍 입력 비교
def fig4():
    pair_navy = '#536F9B'

    for key, name, filename in [
        ('cliff', 'cliff 쌍', 'fig4_input_cliff.png'),
        ('all', '전체 쌍', 'fig4_input_total.png')
    ]:
        fig, ax = plt.subplots(figsize=(3.8, 2.7))

        # 모델 사이 간격 확대
        x = np.arange(len(MODELS)) * 1.5
        w = 0.6

        for k, (data, lab, fc) in enumerate([
            (SINGLE[key], '단일 입력', LIGHT),
            (PAIR[key], '쌍 입력', pair_navy)
        ]):
            v = [m for m, _ in data]
            e = [s for _, s in data]
            xpos = x + (k - 0.5) * w

            ax.bar(
                xpos, v, w * 0.92,
                yerr=e, capsize=2.5,
                color=fc, edgecolor=NAVY, lw=0.9,
                label=lab, zorder=2,
                error_kw=dict(
                    elinewidth=1.0,
                    ecolor='black',
                    capthick=1.0
                )
            )

            for xi, vi, ei in zip(xpos, v, e):
                ax.text(
                    xi,
                    vi + ei + (0.03 if key == 'cliff' else 0.018),
                    f'{vi:.3f}',
                    ha='center', fontsize=6.8, color=EDGE
                )

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=7.5, rotation=0)
        # cliff 그림은 0.75부터 그린다 (막대 차이가 작아 0부터 그리면 구분이 어렵다)
        # 막대 차이가 작아 0부터 그리면 구분이 어려우므로 아래를 잘라 그린다
        ax.set_ylim(*((0.75, 1.25) if key == 'cliff' else (0.50, 0.80)))
        ax.set_ylabel('RMSE', fontsize=9)
        ax.grid(axis='x', visible=False)
        ax.tick_params(axis='y', labelsize=9)

        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

        ax.legend(
            fontsize=8.5,
            frameon=False,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.06),
            ncol=2,
            columnspacing=1.0,
            handletextpad=0.4,
            handlelength=1.2,
            borderaxespad=0
        )

        fig.subplots_adjust(
            left=0.16, right=0.98,
            bottom=0.23, top=0.83
        )

        fig.savefig(
            f'{OUT}/{filename}',
            bbox_inches='tight', facecolor='white'
        )
        plt.close(fig)


# ── 그림 5: 입력 정보 ablation
def fig5():
    fig, ax = plt.subplots(figsize=(4.6, 3.5))

    groups = ['cliff', 'non-cliff', '전체']
    x = np.arange(len(groups)) * 1.0

    # 막대 간격: 숫자 라벨끼리 붙지 않을 만큼 띄운다
    w = 0.22

    shades = [
        '#536F9B',
        '#7591B8',
        '#A7B9D2',
        '#D5DEE9'
    ]

    for k, (lab, vals) in enumerate(ABL):
        v = [m for m, _ in vals]
        e = [s for _, s in vals]
        xpos = x + (k - 1.5) * w

        ax.bar(
            xpos, v, w * 0.9,
            yerr=e, capsize=2.5,
            color=shades[k], edgecolor=NAVY, lw=0.8,
            label=lab, zorder=2,
            error_kw=dict(
                elinewidth=1.0,
                ecolor='black',
                capthick=1.0
            )
        )

        for xi, vi, ei in zip(xpos, v, e):
            ax.text(
                xi, vi + ei + 0.022,
                f'{vi:.3f}',
                ha='center', fontsize=6.8, color=EDGE
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10, rotation=0)
    ax.set_ylabel('RMSE', fontsize=9)
    ax.set_ylim(0.4, 1.05)   # 막대 차이가 작아 0.4부터 그린다

    # 범례를 위쪽에 가로 정렬하고 항목 사이 간격 확대
    ax.legend(
        fontsize=9,
        frameon=False,
        ncol=4,
        loc='upper center',
        bbox_to_anchor=(0.44, 1.12),
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.0,
        borderaxespad=0
    )

    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='y', labelsize=9)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.subplots_adjust(
        left=0.13, right=0.99,
        bottom=0.21, top=0.78
    )
    fig.savefig(
        f'{OUT}/fig5_ablation.png',
        bbox_inches='tight', facecolor='white'
    )
    plt.close(fig)

# ── 그림 2+3 합치기: 두 패널을 원래 배율 그대로 쌓고 왼쪽에 (a)/(b)를 적는다
def fig23_combined():
    from PIL import Image
    GUT, GAP = 150, 70                        # 라벨 칸, 패널 사이 간격 (px)
    a, b = (Image.open(f'{OUT}/{n}.png') for n in ('fig2_pair', 'fig3_single'))
    W, H = max(a.width, b.width) + GUT, a.height + GAP + b.height
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    y = H
    for img, lab in [(a, '(a)'), (b, '(b)')]:
        x0 = GUT + (W - GUT - img.width) / 2
        ax.imshow(img, extent=(x0, x0 + img.width, y - img.height, y))
        ax.text(GUT * 0.45, y - 46, lab, ha='center', va='top', fontsize=10.5, weight='bold')
        y -= img.height + GAP
    fig.savefig(f'{OUT}/fig2_arch.png', dpi=DPI, facecolor='white')
    plt.close(fig)


def main():
    fig1()
    if os.path.exists(os.path.join(DATA, TARGET + '.csv')):
        fig2_pair(); fig3_single(); fig23_combined()
    else:
        print(f'건너뜀: {DATA}/{TARGET}.csv 가 없어 그림 2·3은 그리지 않는다')
    style_apply()
    fig4(); fig5()


main()
print('저장:', OUT, sorted(os.listdir(OUT)))
