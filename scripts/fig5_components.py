#!/usr/bin/env python3
"""Fig. 5 -- component dependence (forest plot).

Each row removes or replaces one component of the deployed system (12 seeds, paired by seed). Left panels: change in
position error and ATE, configuration minus deployed, mean with 95% t confidence interval (positive = worse). Right:
evicted-place recalls served per run, and false and catastrophic recalls summed over the 12 seeds.
"""
import os, json
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from scipy import stats
import matplotlib.pyplot as plt
from figstyle import FULL_W, BLUE, ORANGE, INK, MUTED, FAINT, GRID, panel_title, save

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RES = os.path.join(ROOT, 'results', 'paper2_results.jsonl'); NPZ = os.path.join(ROOT, 'results', 'npz')
OUT = os.path.join(ROOT, 'figures', 'fig5_components.pdf')

rows = [json.loads(l) for l in open(RES) if l.strip()]
by = {}
for r in rows:
    by.setdefault((r['phase'], r['arm']), {})[r['seed']] = r
DEP = by[('core', 'deployed')]

GROUPS = [
    ('Memory', [('core', 'l1only', 'Layer 2 recall off (window alone)'),
                ('components', 'symstore', 'Symbolic-recall control'),
                ('store', 'readout', 'Copy control')]),
    ('Learning', [('components', 'eta0', 'In-window learning off'),
                  ('components', 'noconsol', 'Consolidation off')]),
    ('Verification', [('components', 'noverify', 'Sequence verifier off'),
                      ('components', 'nogate', 'Relative-displacement check off'),
                      ('components', 'noverify_nogate', 'Verifier and relative-displacement check off')]),
]


def r0(x):
    return int(Decimal(str(x)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def tci(v):
    v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return v.mean(), h


def false_count(phase, arm):
    n = f = 0
    for s in by[(phase, arm)]:
        a = np.load(os.path.join(NPZ, f'{phase}_{arm}_cap25_s{s}.npz'))['alias_list']
        if len(a):
            d = a[a[:, 2] == 0][:, 1]; n += len(d); f += int((d > 30.0).sum())
    return f, n


def summary(phase, arm):
    d = by[(phase, arm)]; seeds = sorted(DEP)
    de = tci([d[s]['raw'] - DEP[s]['raw'] for s in seeds]); da = tci([d[s]['ate'] - DEP[s]['ate'] for s in seeds])
    served = tci([d[s]['n_served'] for s in seeds]); cat = sum(d[s]['catastrophic'] for s in seeds)
    f, n = false_count(phase, arm)
    return dict(de=de, da=da, served=served, false=f, n=n, cat=cat)


# y layout: reference row, then groups separated by a gap with a group heading
items = []          # (y, label, summary or None, kind)
y = 0.0
items.append((y, 'Deployed (reference)', summary('core', 'deployed'), 'ref')); y -= 1.0
heads = []
for g, members in GROUPS:
    y -= 0.35; heads.append((y, g)); y -= 0.85
    for ph, arm, lab in members:
        items.append((y, lab, summary(ph, arm), 'row')); y -= 1.0
ymin = y + 0.4

fig = plt.figure(figsize=(FULL_W, 3.05))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.02], wspace=0.08, left=0.335, right=0.995, top=0.86, bottom=0.14)
axE, axA, axT = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

for ax, key, xlab, letter, title in ((axE, 'de', '$\\Delta$ position error (cm)', 'A', 'Position error'),
                                     (axA, 'da', '$\\Delta$ ATE (cm)', 'B', 'ATE')):
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for yy, lab, sm, kind in items:
        if kind == 'ref':
            ax.plot([0], [yy], marker='D', ms=3.6, color=INK, zorder=3); continue
        m, h = sm[key]
        worse = (m - h) > 0
        col = ORANGE if worse else BLUE
        ax.plot([m - h, m + h], [yy, yy], color=col, lw=1.6, solid_capstyle='round', zorder=2)
        ax.plot([m], [yy], marker='o', ms=4.2, color=col, mec='white', mew=0.6, zorder=3)
        ax.plot([m - h, m + h], [yy, yy], ls='none', marker='|', ms=4.5, mew=0.9, color=col, zorder=4)   # end caps keep narrow intervals visible
    for yy, g in heads:
        ax.axhline(yy + 0.42, color=GRID, lw=0.6, zorder=0)
    ax.set_ylim(ymin, 0.6); ax.set_yticks([]); ax.grid(axis='y', visible=False)
    ax.set_xlabel(xlab)
    lim = max(abs(v) for _, _, sm, k in items if k == 'row' for v in (sm[key][0] - sm[key][1], sm[key][0] + sm[key][1])) * 1.08
    ax.set_xlim(-lim, lim)
    ax.spines['left'].set_visible(False)
    panel_title(ax, letter, title)

# row labels (left of panel A) and group headings
for yy, lab, sm, kind in items:
    axE.text(-0.03, yy, lab, transform=axE.get_yaxis_transform(), ha='right', va='center', fontsize=7.5,
             color=INK, fontweight='bold' if kind == 'ref' else 'normal')
for yy, g in heads:
    axE.text(-0.03, yy, g, transform=axE.get_yaxis_transform(), ha='right', va='center', fontsize=7.5, color=INK, style='italic')

# text columns
axT.set_ylim(ymin, 0.6); axT.set_xlim(0, 1); axT.axis('off')
cols = [(0.16, 'Served / run'), (0.50, 'False'), (0.83, 'Catastrophic')]
for x, h in cols:
    axT.annotate(h, (x, 1), xycoords=('data', 'axes fraction'), xytext=(0, 6), textcoords='offset points', ha='center', va='bottom',
                 fontsize=7.5, color=INK, annotation_clip=False)   # same baseline as the panel titles
for yy, lab, sm, kind in items:
    if lab.startswith('Layer 2 recall off'):
        vals = ['0', '—', '—']
    else:
        m, h = sm['served']; vals = [f'{r0(m)} ± {r0(h)}', f"{sm['false']} / {sm['n']}", f"{sm['cat']} / {sm['n']}"]
    for (x, _), v in zip(cols, vals):
        bad = v.split(' / ')[0] not in ('0', '—') and _ != 'Served / run'
        axT.text(x, yy, v, ha='center', va='center', fontsize=7.5, color=INK, fontweight='bold' if bad else 'normal')   # nonzero counts in bold
for yy, g in heads:
    axT.axhline(yy + 0.42, color=GRID, lw=0.6)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save(fig, OUT)
print('wrote', os.path.normpath(OUT))
for yy, lab, sm, kind in items:
    print(f"{lab:48s} dErr {sm['de'][0]:+.1f}±{sm['de'][1]:.1f}  dATE {sm['da'][0]:+.1f}±{sm['da'][1]:.1f}  served {sm['served'][0]:.0f}±{sm['served'][1]:.0f}  false {sm['false']}/{sm['n']}  cat {sm['cat']}")
