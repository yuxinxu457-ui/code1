#!/usr/bin/env python3
"""Fig. 3 -- recall of evicted places.

Top row, one representative run (the seed whose error reduction is closest to the median over the seeds that
have returns to evicted places):
  A  true path and window-alone estimate          B  true path and two-layer estimate, with recalls marked
  C  position error along the run for both systems, laps of loop B shaded, recalls of evicted places as ticks
Bottom row, all 12 seeds:
  D  true distance between vehicle and recalled place at every recall (precision)
  E  error at returns to evicted places, window alone vs two-layer, one point per seed
  F  stored-pattern error at each successive recall (the stored patterns do not degrade)
"""
import os, json, glob
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figstyle import FULL_W, BLUE, ORANGE, INK, MUTED, FAINT, TRUTH, SHADE, GRID, panel_title, save

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RES = os.path.join(ROOT, 'results', 'paper2_results.jsonl'); NPZ = os.path.join(ROOT, 'results', 'npz')
OUT = os.path.join(ROOT, 'figures', 'fig3_recall.pdf')

rows = [json.loads(l) for l in open(RES) if l.strip()]
dep = {r['seed']: r for r in rows if r['phase'] == 'core' and r['arm'] == 'deployed'}
alone = {r['seed']: r for r in rows if r['phase'] == 'core' and r['arm'] == 'l1only'}
seeds = sorted(s for s in alone if alone[s]['n_lost'] > 0 and s in dep)
gain = {s: dep[s]['raw'] - alone[s]['raw'] for s in seeds}
med = np.median(list(gain.values()))
REP = min(seeds, key=lambda s: (abs(gain[s] - med), s))            # representative seed (median gain)


def npz(arm, seed):
    return np.load(os.path.join(NPZ, f'core_{arm}_cap25_s{seed}.npz'))


D, L = npz('deployed', REP), npz('l1only', REP)


def draw_path(ax, est, snaps, color):
    """Estimated path with each closure's position correction drawn dotted: the solid line runs to the position
    before the correction, the dotted segment is the correction itself, and the path resumes from the corrected
    position."""
    jump = {int(r[0]): np.array(r[1:3]) for r in snaps}
    pts, cors = [est[0]], []
    for b in range(1, len(est)):
        if b in jump:
            pre = est[b] - jump[b]
            pts += [pre, np.array([np.nan, np.nan]), est[b]]; cors.append((pre, est[b]))
        else:
            pts.append(est[b])
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.9, zorder=2)
    for a, c in cors:
        ax.plot([a[0], c[0]], [a[1], c[1]], color=color, lw=0.8, ls=(0, (1, 1.3)), zorder=2)
gt = D['gt']; est_d = D['est']; est_l = L['est']
dist = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(gt, axis=0).T))])          # metres travelled
err_d = np.hypot(*(est_d - gt).T) * 100; err_l = np.hypot(*(est_l - gt).T) * 100


def fit_circle(p):
    """Least-squares circle through points p (algebraic fit); returns centre and radius."""
    A = np.c_[2 * p, np.ones(len(p))]; b = (p ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([cx, cy]), np.sqrt(c + cx ** 2 + cy ** 2)


def loop_b_mask(p, iters=25):
    """Label each keyframe as loop A or loop B: 2-means for a start, then assign every point to the circle it
    lies closest to (distance to the circumference) and refit both circles. Loop B is the larger circle."""
    c = p[[np.argmin(p[:, 0]), np.argmax(p[:, 0])]].copy()
    for _ in range(10):
        lab = np.argmin(((p[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        c = np.array([p[lab == j].mean(0) for j in (0, 1)])
    for _ in range(iters):
        circ = [fit_circle(p[lab == j]) for j in (0, 1)]
        d = np.stack([np.abs(np.hypot(*(p - cj).T) - rj) for cj, rj in circ], 1)
        new = np.argmin(d, 1)
        if np.array_equal(new, lab): break
        lab = new
    return lab == int(np.argmax([r for _, r in circ]))


onB = loop_b_mask(gt)
rec = D['snaps']; ev_kf = rec[:, 0][rec[:, 3] == 0].astype(int)          # recalls of evicted places (keyframe index)

fig = plt.figure(figsize=(FULL_W, 5.15))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.86], width_ratios=[1, 1, 1.28], hspace=0.66, wspace=0.34,
                      left=0.06, right=0.995, top=0.94, bottom=0.085)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
gsC = gs[0, 2].subgridspec(2, 1, height_ratios=[0.075, 1], hspace=0.06)
axEv, axC = fig.add_subplot(gsC[0]), fig.add_subplot(gsC[1])
axD, axE, axF = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])

# ---------------- A, B: trajectories ----------------
pad = 0.07
allp = np.vstack([gt, est_d, est_l])
lo = allp.min(0) - pad; hi = allp.max(0) + pad; span = (hi - lo).max()
ctr = (lo + hi) / 2; lo, hi = ctr - span / 2, ctr + span / 2
for ax, est, col, lab, letter in ((axA, est_l, ORANGE, 'Window alone', 'A'), (axB, est_d, BLUE, 'Two-layer system', 'B')):
    ax.plot(gt[:, 0], gt[:, 1], color=TRUTH, lw=2.4, alpha=0.55, solid_capstyle='round', zorder=1)
    draw_path(ax, est, (D if est is est_d else L)['snaps'], col)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.grid(True)
    panel_title(ax, letter, f'{lab}, one run')
axA.set_ylabel('y (m)')
axB.scatter(est_d[ev_kf, 0], est_d[ev_kf, 1], s=9, color=BLUE, edgecolor='white', linewidth=0.5, zorder=3)
MAP_HANDLES = [Line2D([], [], color=TRUTH, lw=2.4, alpha=0.55, label='true path'),
               Line2D([], [], color=ORANGE, lw=1.0, label='window alone'),
               Line2D([], [], color=BLUE, lw=1.0, label='two-layer system'),
               Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 1.3)), label='correction at a closure'),
               Line2D([], [], ls='', marker='o', ms=3.6, mfc=BLUE, mec='white', mew=0.5, label='recall of an evicted place')]

# ---------------- C: error along the run ----------------
edges = np.flatnonzero(np.diff(np.r_[0, onB.astype(int), 0]))
for s0, s1 in zip(edges[::2], edges[1::2]):
    axC.axvspan(dist[s0], dist[min(s1, len(dist) - 1)], color=SHADE, lw=0, zorder=0)
axC.plot(dist, err_l, color=ORANGE, lw=1.1, label='window alone', zorder=2)
axC.plot(dist, err_d, color=BLUE, lw=1.1, label='two-layer system', zorder=3)
axC.set_ylim(0, max(err_l.max(), err_d.max()) * 1.42)
axC.set_xlim(0, dist[-1]); axC.set_xlabel('Distance traveled (m)'); axC.set_ylabel('Position error (cm)')
# the two lines are keyed in the shared legend below A-C; here only the shading needs a key
axC.legend(handles=[Patch(facecolor=SHADE, edgecolor=FAINT, lw=0.6, label='lap of loop B')], loc='upper left', fontsize=7,
           handlelength=1.3, borderaxespad=0.3)
# event strip: recalls of evicted places
for s0, s1 in zip(edges[::2], edges[1::2]):
    axEv.axvspan(dist[s0], dist[min(s1, len(dist) - 1)], color=SHADE, lw=0, zorder=0)
axEv.vlines(dist[ev_kf], 0.12, 0.88, color=BLUE, lw=0.6)
axEv.set_xlim(0, dist[-1]); axEv.set_ylim(0, 1); axEv.set_yticks([]); axEv.set_xticks([]); axEv.grid(False)
for sp in ('left', 'bottom'): axEv.spines[sp].set_visible(False)
axEv.text(1.0, 0.5, ' recalls', transform=axEv.transAxes, ha='left', va='center', fontsize=7, color=INK)
panel_title(axEv, 'C', 'Position error along the same run')

# ---------------- D: recall precision ----------------
DP = {s: npz('deployed', s) for s in dep}
al = np.concatenate([x['alias_list'][:, 1][x['alias_list'][:, 2] == 0] for x in DP.values() if len(x['alias_list'])])
axD.hist(al, bins=np.arange(0, 33, 1.0), color=BLUE, edgecolor='white', linewidth=0.5, zorder=2)
axD.axvline(30, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
axD.text(29.3, axD.get_ylim()[1] * 0.93, 'revisit radius\n30 cm', ha='right', va='top', fontsize=7, color=INK)
axD.set_xlim(0, 32); axD.set_xlabel('True distance to recalled place (cm)'); axD.set_ylabel('Recalls')
panel_title(axD, 'D', f'{int((al < 30).sum())} of {len(al)} recalls are true revisits')

# ---------------- E: error at evicted revisits, per seed ----------------
y0 = np.array([alone[s]['at_lost'] for s in seeds]); y1 = np.array([dep[s]['after'] for s in seeds])
n_s = len(seeds); j0 = np.empty(n_s); j1 = np.empty(n_s)   # offsets by rank within each column: close values get different x
j0[np.argsort(y0)] = (np.arange(n_s) % 3 - 1) * 0.07; j1[np.argsort(y1)] = (np.arange(n_s) % 3 - 1) * 0.07
for a, b, ja, jb in zip(y0, y1, j0, j1):
    axE.plot([ja, 1 + jb], [a, b], color=GRID, lw=0.9, zorder=1)
axE.scatter(j0, y0, s=18, color=ORANGE, edgecolor='white', linewidth=0.6, zorder=3)
axE.scatter(1 + j1, y1, s=18, color=BLUE, edgecolor='white', linewidth=0.6, zorder=3)
for x, y in ((-0.2, y0), (1.2, y1)):
    h = stats.t.ppf(0.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y))
    axE.errorbar([x], [y.mean()], yerr=[h], fmt='o', ms=3.5, color=INK, capsize=3, lw=1.0, zorder=4)
i_rep = seeds.index(REP)
axE.plot([j0[i_rep], 1 + j1[i_rep]], [y0[i_rep], y1[i_rep]], color=MUTED, lw=1.0, ls=(0, (2, 1.5)), zorder=2)
axE.set_xticks([0, 1]); axE.set_xticklabels(['window alone', 'two-layer system'])
axE.set_xlim(-0.45, 1.45); axE.set_ylim(0, None); axE.set_ylabel('Error at returns to\nevicted places (cm)'); axE.grid(axis='x', visible=False)
panel_title(axE, 'E', f'Every seed improves ($n={len(seeds)}$)')

# ---------------- F: stored-pattern error over successive recalls ----------------
# each run's recalls placed on the fraction of that run's sequence of recalls, so that every seed spans the whole axis;
# the median and interquartile range are taken over seeds of the per-seed means in 20 equal bins
NB = 20; fe = np.linspace(0, 1, NB + 1); fc = (fe[:-1] + fe[1:]) / 2; binned = []
for x in DP.values():
    rr = x['recall_rows']; rr = rr[rr[:, 6] == 0]
    if len(rr):
        f = (np.arange(len(rr)) + 0.5) / len(rr); v = rr[:, 4]
        axF.plot(f, v, color=BLUE, lw=0.5, alpha=0.18, zorder=1)
        idx = np.clip(np.digitize(f, fe) - 1, 0, NB - 1)
        binned.append([v[idx == b].mean() if np.any(idx == b) else np.nan for b in range(NB)])
A = np.array(binned)
axF.fill_between(fc, np.nanpercentile(A, 25, axis=0), np.nanpercentile(A, 75, axis=0), color=BLUE, alpha=0.16, lw=0, zorder=2)
axF.plot(fc, np.nanmedian(A, axis=0), color=BLUE, lw=1.5, zorder=3)
axF.set_xlim(0, 1); axF.set_ylim(0, None)
axF.set_xlabel("Fraction of the run's recalls of evicted places"); axF.set_ylabel('Stored-pattern error (cm)')
axF.legend(handles=[Line2D([], [], color=BLUE, lw=1.5, label='median over seeds'),
                    Patch(facecolor=BLUE, alpha=0.16, label='interquartile range'),
                    Line2D([], [], color=BLUE, lw=0.6, alpha=0.4, label='individual seeds')], loc='upper right', fontsize=7)
panel_title(axF, 'F', 'Stored patterns do not degrade with use')

fig.canvas.draw()                    # align C with A and B (whose equal-aspect boxes are only fixed at draw time)
pa, pb = axA.get_position(), axB.get_position(); pe, pc = axEv.get_position(), axC.get_position()
gap = pe.y0 - pc.y1
axEv.set_position([pe.x0, pa.y1 - pe.height, pe.width, pe.height])
axC.set_position([pc.x0, pa.y0, pc.width, (pa.y1 - pe.height - gap) - pa.y0])
fig.legend(handles=MAP_HANDLES, loc='upper center', ncol=5, fontsize=7, handlelength=1.4, columnspacing=1.3,
           bbox_to_anchor=(0.5, pa.y0 - 0.075), bbox_transform=fig.transFigure)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save(fig, OUT)
print('wrote', os.path.normpath(OUT), '| representative seed', REP, f'(gain {gain[REP]:.2f} cm, median {med:.2f})',
      '| evicted recalls on that run', len(ev_kf))
