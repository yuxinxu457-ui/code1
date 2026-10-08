#!/usr/bin/env python3
"""Supplementary Fig. S1 -- how the pattern-separated memory is read out, and how it fails when overloaded.

Builds the same 256-cell memory as the capacity test (code draw 0: same places, same random 8-cell codes, sparsified
patterns, additive storage) at 256 and at 640 stored places, and shows, for three example places, the rows of the
place's 8 cells, the plain sum of those rows, the coincidence (elementwise minimum) readout, and the attractor after the
coincidence readout has been injected for 12 steps. The 8 rows are drawn for one module: the largest (17 x 17 cells) when
recall succeeds, and the module whose attractor lands farthest from the place when it fails.

Examples are chosen by rule, in place-index order: (1) the first place at 256 places; (2) at 640 places, the first
place whose recall from the plain sum lands more than 0.10 m away while the coincidence readout lands within 0.10 m;
(3) at 640 places, the first place whose coincidence recall is catastrophic (ends more than 0.20 m farther from the
place than it started). The slow part (JAX) is cached in results/mechanism_examples.npz.
"""
import os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'results', 'mechanism_examples.npz')
OUT = os.path.join(ROOT, 'figures', 'figS1_mechanism.pdf')
MOD = 2                       # largest module (17 x 17), drawn for the 8 rows when recall succeeds


def build_cache():
    import pattern_separation_pilot as PS
    from slam.weight_l2_slam import inject_flat, bump_flat, crt_decode_res, K_REC, N_SETTLE, CANN_SIZES, WRAP_SCALES
    PS.A_LIVE = PS.measure_live_amplitude()
    rng = np.random.RandomState(1000); pool = PS.sample_places(768, rng)          # code draw 0, as in the capacity test
    sparse = [PS.sparsify(PS.live_pattern(p)) for p in pool[:640]]
    crng = np.random.RandomState(7)
    codes = {k: PS.codes_random(768, k, crng) for k in (2, 4, 8)}[8]              # same draw order as the capacity test

    def recall_from(pattern, true_xy, start_xy):
        s = PS.seed_live(start_xy); p = PS.per_module_norm(pattern)
        inject_flat(s.pose, p, K=K_REC, n_settle=N_SETTLE)
        bump = np.asarray(bump_flat(s.pose)).ravel(); xy, _ = crt_decode_res(bump)
        return bump, np.asarray(xy, float)

    def example(M, p, start_rng):
        W, _ = PS.store(codes[:M], sparse[:M], 'additive'); cs = codes[p]
        d = start_rng.uniform(*PS.DISP); ang = start_rng.uniform(0, 2 * np.pi)
        start = pool[p] + d * np.array([np.cos(ang), np.sin(ang)])
        rows = W[cs]; s_sum = PS.readout(W, cs, 'sum'); s_min = PS.readout(W, cs, 'coinc')
        bump_min, xy_min = recall_from(s_min, pool[p], start)
        _, xy_sum = recall_from(s_sum, pool[p], start)
        sharers = [int(sum(1 for q in range(M) if c in codes[q]) - 1) for c in cs]
        return dict(M=M, p=p, rows=rows, s_sum=s_sum, s_min=s_min, bump=bump_min, true=pool[p], start=start,
                    xy_min=xy_min, xy_sum=xy_sum, err_min=float(np.hypot(*(xy_min - pool[p]))),
                    err_sum=float(np.hypot(*(xy_sum - pool[p]))), d0=float(d), sharers=np.array(sharers))

    ex = [example(256, 0, np.random.RandomState(0))]
    got_sum_fail = got_cat = None
    for p in range(640):
        e = example(640, p, np.random.RandomState(p))
        if got_sum_fail is None and e['err_sum'] > 0.10 and e['err_min'] < 0.10: got_sum_fail = e
        if got_cat is None and e['err_min'] > e['d0'] + 0.20: got_cat = e
        if got_sum_fail is not None and got_cat is not None: break
    ex += [got_sum_fail, got_cat]
    out = {}
    for i, e in enumerate(ex):
        for k, v in e.items(): out[f'{i}_{k}'] = np.asarray(v)
    out['cann_sizes'] = np.array(CANN_SIZES); out['wrap_scales'] = np.array(WRAP_SCALES, float)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True); np.savez(CACHE, **out)
    print('cached examples:', [(int(e['M']), int(e['p']), round(e['err_sum'] * 100, 1), round(e['err_min'] * 100, 1)) for e in ex])


if not os.path.exists(CACHE):
    build_cache()

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from figstyle import FULL_W, BLUE, ORANGE, INK, MUTED, panel_title, save

Z = np.load(CACHE); sizes = [int(c) for c in Z['cann_sizes']]; scales = [float(v) for v in Z['wrap_scales']]
offs = np.cumsum([0] + [c ** 2 for c in sizes])
blk = lambda flat, m: np.asarray(flat[offs[m]:offs[m + 1]], float).reshape(sizes[m], sizes[m])
cmap = LinearSegmentedColormap.from_list('mem', ['#ffffff', '#cfe0f5', '#6ea5e6', BLUE, '#0d3c78'])


def phase(xy, m):
    s = scales[m]; c = sizes[m]; return (float(xy[0]) % s) / s * c, (float(xy[1]) % s) / s * c


def show(ax, img, m, true_xy=None, dec_xy=None, dec_ok=True, ms=6.0):
    mx = img.max(); ax.imshow(img / mx if mx > 0 else img, cmap=cmap, vmin=0, vmax=1, origin='lower', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color('#c9c8c3'); sp.set_linewidth(0.6)
    if true_xy is not None:
        tx, ty = phase(true_xy, m); ax.plot([tx], [ty], marker='o', ms=ms, mfc='none', mec=INK, mew=0.9)
    if dec_xy is not None:
        dx, dy = phase(dec_xy, m); ax.plot([dx], [dy], marker='x', ms=ms * 0.75, mec=INK if dec_ok else ORANGE, mew=1.2)


ROW_TEXT = ['256 places stored (as many places as cells): correct recall',
            '640 places stored: the sum of the rows fails but the coincidence readout recovers the place',
            '640 places stored: interference weakens the coincidence and the recall goes to a wrong place']
COLS = ["The place's 8 rows (one module)", 'Sum of the rows', 'Coincidence (minimum)', 'Attractor after recall']


def phase_gap(a, b, m):
    # distance between two phases on the periodic sheet of module m, in cells
    c = sizes[m]; d = np.abs(np.array(phase(a, m)) - np.array(phase(b, m))); d = np.minimum(d, c - d); return float(np.hypot(*d))


fig = plt.figure(figsize=(FULL_W, 5.5))
outer = fig.add_gridspec(3, 1, hspace=0.52, left=0.005, right=0.995, top=0.89, bottom=0.055)
rows_info = []
for i in range(3):
    M = int(Z[f'{i}_M']); true = Z[f'{i}_true']; rows = Z[f'{i}_rows']; sh = Z[f'{i}_sharers']
    ok = float(Z[f'{i}_err_min']) < 0.10
    mod = MOD if ok else max(range(3), key=lambda m: phase_gap(Z[f'{i}_xy_min'], true, m))   # the module where recall fails
    row = outer[i].subgridspec(1, 4, width_ratios=[1.55, 1.15, 1.15, 1.15], wspace=0.10)
    mini = row[0].subgridspec(2, 4, wspace=0.05, hspace=0.05)
    minis = []
    for j in range(8):
        ax = fig.add_subplot(mini[j // 4, j % 4]); ax.set_anchor('N' if j < 4 else 'S')   # block top level with the strips
        show(ax, blk(rows[j], mod), mod, true_xy=true, ms=5.0); minis.append(ax)
    err_sum = float(Z[f'{i}_err_sum']) * 100; err_min = float(Z[f'{i}_err_min']) * 100; d0 = float(Z[f'{i}_d0']) * 100
    strips = []
    for col, key in ((1, 's_sum'), (2, 's_min'), (3, 'bump')):
        sg = row[col].subgridspec(1, 3, wspace=0.08); axs = []
        for m in range(3):
            ax = fig.add_subplot(sg[m]); ax.set_anchor('N')          # strips aligned with the top of the row
            show(ax, blk(Z[f'{i}_{key}'], m), m, true_xy=true, dec_xy=Z[f'{i}_xy_min'] if key == 'bump' else None, dec_ok=ok, ms=5.0)
            axs.append(ax)
        strips.append(axs)
    rows_info.append((minis, strips, sh, err_sum, err_min, d0, ok, mod))

fig.canvas.draw()                    # fix the final (aspect-adjusted) panel positions before placing any text
for i, (minis, strips, sh, err_sum, err_min, d0, ok, mod) in enumerate(rows_info):
    top = max(ax.get_position().y1 for ax in minis + strips[0])
    # column headings, top row only: centered over each block, just above the panels
    if i == 0:
        blocks = ((minis[0], minis[3], COLS[0]),) + tuple((s[0], s[2], c) for s, c in zip(strips, COLS[1:]))
        for a0, a1, txt in blocks:
            fig.text((a0.get_position().x0 + a1.get_position().x1) / 2, top + 0.012, txt, fontsize=7.5, color=INK,
                     ha='center', va='bottom', fontweight='bold')
    # row heading (above the column headings in the top row)
    yh = top + (0.052 if i == 0 else 0.016)
    fig.text(0.006, yh, 'ABC'[i], fontsize=9, fontweight='bold', ha='left', va='bottom')
    fig.text(0.030, yh, ROW_TEXT[i], fontsize=8, ha='left', va='bottom')
    # result annotations under the strips
    def under(axs, txt, color):
        p0, p2 = axs[0].get_position(), axs[2].get_position()
        fig.text((p0.x0 + p2.x1) / 2, p0.y0 - 0.008, txt, fontsize=7, color=INK, ha='center', va='top')   # all text in ink
    p0, p3 = minis[4].get_position(), minis[7].get_position()
    fig.text((p0.x0 + p3.x1) / 2, p0.y0 - 0.008, f"{scales[mod]:.2f} m module; each of the place's 8 cells\nis shared with "
             f"{int(sh.min())}\u2013{int(sh.max())} other places", fontsize=7, color=INK, ha='center', va='top', linespacing=1.15)
    under(strips[0], f'recall from the sum: {err_sum:.0f} cm off', ORANGE if err_sum >= 10 else INK)
    under(strips[1], 'input to the attractor', MUTED)
    under(strips[2], f'lands {err_min:.1f} cm from the place' if ok else f'lands {err_min:.0f} cm away (started {d0:.0f} cm)',
          INK if ok else ORANGE)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save(fig, OUT)
print('wrote', os.path.normpath(OUT))
