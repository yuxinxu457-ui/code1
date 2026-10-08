#!/usr/bin/env python3
"""Fig. 2. (A) Bounded window alone vs window size W (cliff phase and core:l1only, 12 seeds): in-window closures
served and returns to evicted places lost per run, mean with 95% t-interval and per-seed points; window sizes between
40 and 60 were not run, so that stretch is drawn dashed. (B) Memory size versus trajectory length: the two-layer system
(fixed matrix plus window) against an unbounded pose graph at 56 B per keyframe (grey), measured at the base length
(core:deployed) and the longer schedules (length phase).
Reads results/paper2_results.jsonl; writes figures/fig2_cliff_footprint.pdf and .png."""
import os, json, numpy as np
import matplotlib.pyplot as plt
from figstyle import FULL_W, BLUE, ORANGE, INK, MUTED, GRID, panel_title, save
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from scipy import stats
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); R=os.path.join(ROOT,'results','paper2_results.jsonl')
rows=[json.loads(l) for l in open(R) if l.strip()]
caps={}
for r in rows:
    if r['arm'].startswith('l1only') and r['phase'] in ('cliff', 'core') and r.get('steps', 3500) == 3500: caps.setdefault(r['cap'],[]).append(r)   # base-length runs only
W=sorted(caps)
def mci(v):
    v=np.array(v,float); h=stats.t.ppf(0.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0; return v.mean(), h
fig,(axA,axB)=plt.subplots(1,2,figsize=(FULL_W,2.75), gridspec_kw={'wspace':0.32})
rng=np.random.default_rng(0)
for col,key,lab in ((BLUE,'relaxes','in-window closures served'),(ORANGE,'n_lost','returns to evicted places lost\nby the window alone')):
    m=[mci([r[key] for r in caps[w]])[0] for w in W]; h=[mci([r[key] for r in caps[w]])[1] for w in W]
    for w in W:  # per-seed points, jittered
        y=[r[key] for r in caps[w]]; axA.scatter(np.full(len(y),w)+rng.uniform(-0.8,0.8,len(y)), y, s=6, color=col, alpha=0.35, linewidths=0, zorder=2)
    axA.errorbar(W, m, yerr=h, color=col, ls='none', marker='o', ms=4, capsize=2, elinewidth=1.0, zorder=3)
    k40 = W.index(40)                  # solid across the swept sizes 25-40, dashed across the untested 40-60 gap
    axA.plot(W[:k40 + 1], m[:k40 + 1], color=col, lw=1.5, zorder=3, label=lab)
    axA.plot(W[k40:], m[k40:], color=col, lw=1.3, ls=(0, (3, 2)), zorder=3)
axA.axvline(25, color=MUTED, lw=0.8, ls=':'); axA.text(23.2, 0.42, 'operating point, $W=25$', va='center', ha='right', rotation=90, fontsize=7, color=INK, transform=axA.get_xaxis_transform())
axA.set_xlabel('Window size $W$ (places)'); axA.set_ylabel('Count per run'); axA.set_xticks(W); axA.set_xlim(21, 64); axA.set_ylim(-5, 132)
panel_title(axA, 'A', 'The bounded window: closures served and lost')
axA.legend(loc='upper right', fontsize=7, handlelength=1.6)
# panel B: memory size vs trajectory length, from the core and length phases
L={}
for r in rows:
    if r['phase'] in ('core','length') and r['arm'].startswith('deployed') and r.get('cap')==25: L.setdefault(r['steps'],[]).append(r)
if len(L)>=2:
    BN=56; WB=592896; KSTAR=WB/BN
    xs=sorted(L); Ks=np.array([np.mean([r['K'] for r in L[x]]) for x in xs])
    kk=np.linspace(Ks.min(), KSTAR*1.6, 200)
    axB.plot(kk, kk*BN/1e3, color=MUTED, lw=1.2, ls='--', alpha=0.8); axB.plot(Ks, Ks*BN/1e3, color=MUTED, lw=1.5, marker='o', ms=4, label='unbounded pose graph')
    axB.plot(kk, np.full_like(kk, (WB+25*BN)/1e3), color=BLUE, lw=1.2, ls='--', alpha=0.7); axB.plot(Ks, np.full_like(Ks,(WB+25*BN)/1e3), color=BLUE, lw=1.5, marker='o', ms=4, label='two-layer system')
    axB.axvline(KSTAR, color=MUTED, lw=0.8, ls=':'); axB.text(KSTAR*0.93, 760, f'crossover at about {round(KSTAR,-2):,.0f} keyframes', fontsize=7, color=INK, ha='right', va='center')
    axB.set_xscale('log'); axB.set_yscale('log'); axB.set_xlabel('Keyframes $K$ (trajectory length)'); axB.set_ylabel('Memory size (kB)'); axB.legend(fontsize=7, loc='lower right', bbox_to_anchor=(1.0, 0.09), handlelength=1.5, frameon=True, facecolor='white', edgecolor='none', framealpha=1.0)
    axB.yaxis.set_major_locator(FixedLocator([30, 100, 300, 1000])); axB.yaxis.set_major_formatter(FixedFormatter(['30', '100', '300', '1000'])); axB.yaxis.set_minor_locator(NullLocator())
    axB.text(Ks.max()*1.02, 0.03, 'tested lengths', fontsize=7, color=INK, transform=axB.get_xaxis_transform(), va='bottom', ha='right'); axB.text(KSTAR*0.30, 0.03, 'extrapolated', fontsize=7, color=INK, transform=axB.get_xaxis_transform(), va='bottom')
else:
    axB.text(0.5,0.5,'B  memory size vs. trajectory length\n(pending: length phase)', ha='center', va='center', fontsize=9, color=INK, transform=axB.transAxes); axB.set_xticks([]); axB.set_yticks([]); axB.grid(False)
panel_title(axB, 'B', 'Memory size versus trajectory length')
out=os.path.join(ROOT,'figures','fig2_cliff_footprint.pdf')
save(fig, out); print('wrote', os.path.normpath(out))
