#!/usr/bin/env python3
"""Supplementary Fig. S2 -- the lifecycle of every place on one run (seed 930, the run shown in Fig. 3).

A  one row per place, in order of creation: time in the window (learning), eviction (consolidation), and the
   read-only period afterwards, with every recall of the place from the memory marked; laps of loop B shaded.
B  how many places are in the window and how many are held only by the memory, along the run.
Input: results/lifecycle_s930.npz (made by lifecycle_data.py).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figstyle import FULL_W, BLUE, ORANGE, INK, MUTED, FAINT, SHADE, LIGHT_BLUE, panel_title, save

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SEED = 930
Z = np.load(os.path.join(ROOT, 'results', f'lifecycle_s{SEED}.npz'))
OUT = os.path.join(ROOT, 'figures', 'figS2_lifecycle.pdf')
gt = Z['gt']; K = len(gt)
dist = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(gt, axis=0).T))])
P = Z['place']; created = Z['created']; evicted = Z['evicted']
rk, rp, rin = Z['recall_kf'], Z['recall_place'], Z['recall_inwin']


def fit_circle(p):
    A = np.c_[2 * p, np.ones(len(p))]; b = (p ** 2).sum(1); cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([cx, cy]), np.sqrt(c + cx ** 2 + cy ** 2)


def loop_b_mask(p, iters=25):
    c = p[[np.argmin(p[:, 0]), np.argmax(p[:, 0])]].copy()
    for _ in range(10):
        lab = np.argmin(((p[:, None, :] - c[None]) ** 2).sum(-1), axis=1); c = np.array([p[lab == j].mean(0) for j in (0, 1)])
    for _ in range(iters):
        circ = [fit_circle(p[lab == j]) for j in (0, 1)]
        d = np.stack([np.abs(np.hypot(*(p - cj).T) - rj) for cj, rj in circ], 1); new = np.argmin(d, 1)
        if np.array_equal(new, lab): break
        lab = new
    return lab == int(np.argmax([r for _, r in circ]))


onB = loop_b_mask(gt)
edges = np.flatnonzero(np.diff(np.r_[0, onB.astype(int), 0]))
spans = [(dist[s0], dist[min(s1, K - 1)]) for s0, s1 in zip(edges[::2], edges[1::2])]
place_loop_B = np.array([onB[min(c, K - 1)] for c in created])

fig = plt.figure(figsize=(FULL_W, 4.9))
gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.30, left=0.06, right=0.985, top=0.855, bottom=0.095)
ax = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1], sharex=ax)

# ---------------- A: lifecycle raster ----------------
for a0, a1 in spans:
    ax.axvspan(a0, a1, color=SHADE, lw=0, zorder=0); axB.axvspan(a0, a1, color=SHADE, lw=0, zorder=0)
n = len(P)                                     # rows grouped by the loop of creation, in creation order within each group
order = np.lexsort((np.arange(n), place_loop_B.astype(int))); rank = np.empty(n, int); rank[order] = np.arange(n); ys = n - 1 - rank
yof = {int(p): y for p, y in zip(P, ys)}
for p, c, e, y in zip(P, created, evicted, ys):
    x0 = dist[c]; x1 = dist[e] if e >= 0 else dist[-1]
    ax.plot([x0, x1], [y, y], color=LIGHT_BLUE, lw=3.4, solid_capstyle='butt', zorder=1)
    if e >= 0:
        ax.plot([x1, dist[-1]], [y, y], color=FAINT, lw=0.8, zorder=1)
        ax.plot([x1], [y], marker='|', ms=5.5, mew=1.1, color=INK, zorder=3)
ev = rin == 0
ax.scatter(dist[rk[ev]], [yof[int(p)] for p in rp[ev]], s=9, color=BLUE, edgecolor='white', linewidth=0.4, zorder=4)
if (~ev).any():
    ax.scatter(dist[rk[~ev]], [yof[int(p)] for p in rp[~ev]], s=14, facecolor='none', edgecolor=INK, linewidth=0.8, zorder=4)
ax.set_ylim(-1, n); ax.set_yticks([]); ax.grid(axis='y', visible=False)
# brackets on the left: places created on the first laps of loop A, and on the first lap of loop B
firstB = spans[0][0]
groups = (('created on loop A', ~place_loop_B), ('created on loop B', place_loop_B))
for lab, mask in groups:
    yy = ys[mask]
    if len(yy):
        y0, y1 = yy.min() - 0.3, yy.max() + 0.3
        ax.annotate('', xy=(-0.012, y0), xytext=(-0.012, y1), xycoords=('axes fraction', 'data'),
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8), annotation_clip=False)
        ax.annotate(lab, (-0.022, (y0 + y1) / 2), xycoords=('axes fraction', 'data'), ha='right', va='center', fontsize=7,
                    color=INK, rotation=90, annotation_clip=False)
ax.tick_params(labelbottom=False)
fig.legend(handles=[Patch(facecolor=LIGHT_BLUE, edgecolor='none', label='in the window (learning)'),
                   Line2D([], [], color=INK, ls='', marker='|', ms=6, mew=1.1, label='evicted (consolidated)'),
                   Line2D([], [], color=FAINT, lw=1.0, label='held by the memory only (read-only)'),
                   Line2D([], [], ls='', marker='o', ms=3.6, mfc=BLUE, mec='white', mew=0.4, label='recall of an evicted place'),
                   Line2D([], [], ls='', marker='o', ms=4.2, mfc='none', mec=INK, mew=0.8, label='in-window closure'),
                   Patch(facecolor=SHADE, edgecolor='none', label='lap of loop B')],
          loc='upper center', ncol=3, fontsize=7, columnspacing=1.6, handlelength=1.5, bbox_to_anchor=(0.52, 1.0))
panel_title(ax, 'A', f'Lifecycle of every place on one run ({n} places, {int((evicted >= 0).sum())} evicted, {int(ev.sum())} recalls of evicted places)')

# ---------------- B: occupancy ----------------
kf = np.arange(K)
in_win = np.array([np.sum((created <= k) & ((evicted < 0) | (evicted > k))) for k in kf])
held = np.array([np.sum((evicted >= 0) & (evicted <= k)) for k in kf])
axB.plot(dist, in_win, color=BLUE, lw=1.3, label='places in the window', zorder=3)
axB.plot(dist, held, color=MUTED, lw=1.3, ls=(0, (3, 1.5)), label='places held by the memory only', zorder=3)
axB.axhline(25, color=INK, lw=0.7, ls=':', zorder=2); axB.text(dist[-1] * 0.005, 25.8, 'window size $W=25$', ha='left', va='bottom', fontsize=7, color=INK)
axB.set_ylim(-1, 33); axB.set_yticks([0, 10, 20, 30]); axB.set_ylabel('Places'); axB.set_xlabel('Distance traveled (m)')
axB.set_xlim(0, dist[-1] + 0.12); axB.set_xticks([0, 5, 10, 15])   # small pad so the last recall dot is not cut; ticks as in Fig. 3C
axB.legend(loc='lower right', fontsize=7, borderaxespad=0.3)
panel_title(axB, 'B', 'Window and memory contents along the run')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save(fig, OUT)
print('wrote', os.path.normpath(OUT), '| places', n, '| evicted', int((evicted >= 0).sum()), '| recalls from memory', int(ev.sum()), '| in-window closures', int((~ev).sum()))
