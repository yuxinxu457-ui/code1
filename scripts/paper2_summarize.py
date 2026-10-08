#!/usr/bin/env python3
"""Summarise the campaign results: mean with 95% t-interval per (phase, arm, window size) for the main metrics,
per-seed paired differences deployed - l1only, the deletion test (poison_evicted vs deployed trajectories), and
the error of every W = 25 arm at a common reference set of evicted revisits.

Usage: python scripts/paper2_summarize.py [RESULTS_DIR]   (default: results/, containing paper2_results.jsonl and npz/)"""
import os, sys, json
import numpy as np
from scipy import stats
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results')
RES = os.path.join(RDIR, 'paper2_results.jsonl'); NPZ = os.path.join(RDIR, 'npz')
rows = [json.loads(l) for l in open(RES) if l.strip()] if os.path.exists(RES) else []
def ci(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0: return "  --  "
    if len(v) == 1: return f"{v[0]:6.1f} (n=1)"
    h = stats.t.ppf(0.975, len(v)-1) * v.std(ddof=1) / np.sqrt(len(v)); return f"{v.mean():6.1f} ±{h:4.1f} (n={len(v)})"
by = {}
for r in rows: by.setdefault((r['phase'], r['arm'], r.get('cap')), []).append(r)
print(f"{len(rows)} runs.\n")
cols = ['raw','ate','n_served','n_lost','before','after','delta','catastrophic','at_lost','tail_served','tail_lost','alias_mean','alias_max','row_mean','row_max','row_first','row_last','landrow_mean','consol_target','consol_after','W_gt','BAT_gt','vsEMA','vsMED','vsBATCH','nullgap','nodes','K','loops','relaxes','l2_only','l1_gate_rej']
for (p, a, c), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
    print(f"=== {p}:{a} cap{c}  (n={len(rs)}) ===")
    for c in cols:
        v = [r.get(c) for r in rs]
        if any(x is not None and (isinstance(x,(int,float)) and np.isfinite(x)) for x in v): print(f"  {c:<14} {ci(v)}")
# paired
def paired(pa, pb, col):
    A = {r['seed']: r.get(col) for r in by.get(pa, [])}; B = {r['seed']: r.get(col) for r in by.get(pb, [])}
    d = [B[s]-A[s] for s in A if s in B and A[s] is not None and B[s] is not None and np.isfinite(A[s]) and np.isfinite(B[s])]
    return ci(d)
for cap in sorted({k[2] for k in by if k[0]=='core'}, key=lambda x: x or 0):
    A=('core','l1only',cap); B=('core','deployed',cap)
    if A in by and B in by:
        print(f"\n=== PAIRED deployed - l1only, cap{cap} (per seed) ===")
        for c in ('raw','ate'): print(f"  Δ{c:<12} {paired(A,B,c)}")
        print(f"  at revisits: l1only at_lost {ci([r.get('at_lost') for r in by[A]])} | deployed before {ci([r.get('before') for r in by[B]])} -> after {ci([r.get('after') for r in by[B]])}")
        print(f"  tails: l1only {ci([r.get('tail_lost') for r in by[A]])} | deployed {ci([r.get('tail_served') for r in by[B]])}")
        cat=[r['catastrophic'] for r in by[B]]; srv=[r['n_served'] for r in by[B]]; print(f"  catastrophic k'/M pooled: {sum(cat)} / {sum(srv)}")
        print(f"  seeds with evicted revisits: {sum(1 for r in by[A] if r['n_lost']>0)}/{len(by[A])}")
# deletion test: symbolic positions of evicted places deleted; the trajectory must equal the deployed one
for parm in ('poison_evicted',):
    devs = []
    for r in by.get(('store', parm, 25), []):
        try:
            a = np.load(os.path.join(NPZ, f"store_{parm}_cap{r['cap']}_s{r['seed']}.npz"))['est']; b = np.load(os.path.join(NPZ, f"core_deployed_cap{r['cap']}_s{r['seed']}.npz"))['est']
            devs.append(float(np.nanmax(np.hypot(*(a-b).T)))*100 if a.shape == b.shape else float('nan'))
        except Exception: pass
    if devs: print(f"\n  deletion test {parm}: max |est - deployed| per seed (cm): {' '.join(f'{d:.3f}' for d in devs)}")

# ---------------------------------------------------------------------------------------------------------------------
# Error at a common reference set of evicted revisits. 'after' averages each arm's own served closures, which differ
# between arms; here every W = 25 arm is scored at the same keyframes, the evicted revisits the window alone loses
# (core:l1only lost_kfs, per seed; keyframes are placed identically in every arm, so indices align).
# ref_evicted = mean unaligned position error (cm) at those keyframes.
# ---------------------------------------------------------------------------------------------------------------------
def _npz(phase, arm, cap, seed):
    p = os.path.join(NPZ, f"{phase}_{arm}_cap{cap}_s{seed}.npz"); return np.load(p) if os.path.exists(p) else None
ref = {}
for r in by.get(('core', 'l1only', 25), []):
    d = _npz('core', 'l1only', 25, r['seed'])
    if d is not None and len(d['lost_kfs']): ref[r['seed']] = d['lost_kfs'].astype(int)
if ref:
    print("\n=== ERROR AT THE REFERENCE EVICTED REVISITS (keyframes = core:l1only lost_kfs, per seed; raw position error, cm) ===")
    tab = {}
    for (p, a, c), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if c != 25 or rs[0].get('steps', 3500) != 3500: continue
        vals = {}
        for r in rs:
            d = _npz(p, a, 25, r['seed'])
            if d is None or r['seed'] not in ref: continue
            e = d['est']; g = d['gt']; k = ref[r['seed']]; k = k[k < len(e)]
            err = np.hypot(*(e[k] - g[k]).T) * 100; vals[r['seed']] = float(np.nanmean(err))
        if vals: tab[(p, a)] = vals; print(f"  {p}:{a:<16} ref_evicted {ci(list(vals.values()))}")
    D = tab.get(('core', 'deployed'))
    if D:
        print("  paired vs deployed (arm - deployed):")
        for (p, a), vals in tab.items():
            if (p, a) == ('core', 'deployed'): continue
            d = [vals[s] - D[s] for s in vals if s in D]
            if d: print(f"    {p}:{a:<16} Δ {ci(d)}")
