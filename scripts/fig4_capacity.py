#!/usr/bin/env python3
"""Fig. 4. The store holds more places than it has cells.
  A  synthetic capacity test (results/capacity_test.json): recall accuracy (within 0.10 m) vs places stored for the deployed
     pattern-separated code (sparse patterns, random 8-cell codes, coincidence read-out), the one-cell code with places
     added to shared cells, the dense-pattern variant and appearance-derived codes; mean over the three code draws.
  B  in the SLAM system (capacity phase and core:deployed, 12 seeds): returns to evicted places served per run with memory
     size N = 32 / 48 / 256 place cells, one-cell code vs pattern-separated code (k = 8), with the window-alone loss line;
     catastrophic recalls / recalls of evicted places annotated on each bar.
Writes figures/fig4_capacity.pdf and .png."""
import os, json, numpy as np
import matplotlib.pyplot as plt
from figstyle import FULL_W, BLUE, ORANGE, PURPLE, TEAL, INK, MUTED, GRID, panel_title, save
from scipy import stats
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); RES=os.path.join(ROOT,'results','paper2_results.jsonl')
PIL=os.path.join(ROOT,'results','capacity_test.json')
OUT=os.path.join(ROOT,'figures','fig4_capacity')
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42})
fig,(a,b)=plt.subplots(1,2,figsize=(FULL_W,2.8), gridspec_kw={'width_ratios':[1.2,1],'wspace':0.28})
# --- A: memory tested in isolation ---
pil=json.load(open(PIL)); import collections; agg=collections.defaultdict(list)
for r in pil: agg[(r['arm'],r['M'])].append(r['prec10'])
loads=sorted({r['M'] for r in pil})
for arm,col,lab,lw,mk,ms,zo in (('sparse_rand_k8_coinc',BLUE,'pattern-separated code (deployed)',1.8,'o',3.2,4),('onehot_add',ORANGE,'one-cell code (cells shared)',1.3,'s',2.6,3),('dense_rand_k8_coinc',PURPLE,'unsparsified patterns',1.2,'^',2.9,3),('sparse_appear_k8_coinc',TEAL,'appearance-derived codes',1.2,'D',2.4,3)):
    ys=[np.mean(agg[(arm,M)]) for M in loads]; a.plot(loads,ys,marker=mk,ms=ms,lw=lw,color=col,label=lab,zorder=zo)   # deployed code drawn on top
a.axvline(256,color=MUTED,lw=0.8,ls=':',zorder=1); a.text(248,0.80,'$N=256$ cells',fontsize=7,color=INK,va='center',ha='right')
a.set_xscale('log'); a.set_xlabel('Places stored $M$'); a.set_ylabel('Recall accuracy (within 10 cm)'); a.set_ylim(-0.02,1.03); a.legend(fontsize=7,loc='lower left',bbox_to_anchor=(0.0,0.02),frameon=True,facecolor='white',edgecolor='none',framealpha=1.0); panel_title(a, 'A', 'Memory tested in isolation')
from matplotlib.ticker import NullFormatter, FixedLocator
a.xaxis.set_major_locator(FixedLocator([64,128,256,512,768])); a.set_xticklabels(['64','128','256','512','768']); a.xaxis.set_minor_formatter(NullFormatter()); a.xaxis.set_minor_locator(FixedLocator([]))
# --- B: capacity in the SLAM system ---
rows=[json.loads(l) for l in open(RES) if l.strip()]
def arm(ph,nm): return {r['seed']:r for r in rows if r['phase']==ph and r['arm']==nm and r['cap']==25}
D=arm('core','deployed'); L=arm('core','l1only'); sd=sorted(set(D)&set(L))
def ci(v): v=np.asarray(v,float); return v.mean(), stats.t.ppf(0.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
Ns=[32,48,256]; oh=[arm('capacity','onehot_N32'),arm('capacity','onehot_N48'),arm('capacity','onehot_N256')]; kh=[arm('capacity','khot_N32'),arm('capacity','khot_N48'),D]
x=np.arange(3); w=0.36
for i,(A,col,lab) in enumerate(((oh,ORANGE,'one-cell code (least recently used cell reassigned)'),(kh,BLUE,'pattern-separated code, $k=8$'))):
    m=[ci([A[j][s]['n_served'] for s in sd])[0] for j in range(3)]; h=[ci([A[j][s]['n_served'] for s in sd])[1] for j in range(3)]
    if i==1: m[2]=ci([D[s]['n_served'] for s in sd])[0]; h[2]=ci([D[s]['n_served'] for s in sd])[1]
    bars=b.bar(x+(i-0.5)*w, m, w, color=col, yerr=h, capsize=2.5, error_kw={'lw':0.9,'color':INK}, label=lab, edgecolor='white', linewidth=0.8, zorder=2)
    for j,bar in enumerate(bars):
        cat=sum(A[j][s]['catastrophic'] for s in sd); tot=sum(A[j][s]['n_served'] for s in sd)
        if bar.get_height() > 35: b.text(bar.get_x()+bar.get_width()/2, 2.5, f"{cat} / {tot}", ha='center', va='bottom', rotation=90, fontsize=7, color=INK, zorder=4)
        else: b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+h[j]+2, f"{cat} / {tot}", ha='center', va='bottom', rotation=90, fontsize=7, color=INK, zorder=4)
lost=ci([L[s]['n_lost'] for s in sd])[0]; b.axhline(lost,color=MUTED,lw=0.9,ls='--',label='returns to evicted places per run\n(all lost by the window alone)')
b.set_xticks(x); b.set_xticklabels(['$N=32$', '$N=48$', '$N=256$\n(deployed)']); b.set_xlabel('Place cells $N$ (course has $52\\pm2$ places)'); b.set_ylabel('Returns served per run'); b.set_ylim(0,max(150,lost*1.75)); b.grid(axis='x',visible=False); b.legend(fontsize=6.8,loc='upper left',bbox_to_anchor=(0.0,1.02),frameon=True,facecolor='white',edgecolor='none',framealpha=1.0); panel_title(b, 'B', 'In the SLAM system')
save(fig, OUT+'.pdf'); print('wrote',OUT+'.pdf')
