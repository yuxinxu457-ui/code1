#!/usr/bin/env python3
"""Print the numbers of Supplementary Tables S-II to S-V from the shipped results.

  S-II   memory capacity test, deployed code, summed over the three code draws, per load M
  S-III  alternative memories (one-cell, unsparsified, appearance-derived) and code sizes k = 2, 4
  S-IV   window sweep, window alone (cliff phase and core:l1only), 12 seeds
  S-V    Layer-2 synaptic operations per event and share of the Layer-2 budget (core:deployed)

Capacity-test rates are counts summed over the code draws with Wilson 95% bounds: accuracy
= recalls landing within 0.10 m of the place / recalls evaluated (prec10_all x n);
identification and catastrophic (landing more than 0.20 m farther away than the arrival) are fractions of
the recalls that fired (every recall fires for these additive stores). Window-sweep values are means with
Student-t 95% half-widths over seeds.

Usage: python scripts/supplement_tables.py [RESULTS_DIR]   (default: results/)
"""
import os, sys, json, math, collections
import numpy as np
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results')
cap_rows = json.load(open(os.path.join(RDIR, 'capacity_test.json')))
rows = [json.loads(l) for l in open(os.path.join(RDIR, 'paper2_results.jsonl')) if l.strip()]


def wilson(x, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = x / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


agg = collections.defaultdict(list)
for r in cap_rows: agg[(r['arm'], r['M'])].append(r)
LOADS = sorted({r['M'] for r in cap_rows})


def pooled(arm, M):
    """Counts summed over the code draws: n evaluated, fired, accurate, identified, catastrophic."""
    rs = agg[(arm, M)]
    n = sum(r['n'] for r in rs); fired = sum(r['fired'] for r in rs)
    cnt = lambda key, den: sum(int(round(r[key] * r[den])) for r in rs if r[den] and np.isfinite(r[key]))
    return dict(n=n, fired=fired, acc=cnt('prec10_all', 'n'), ident=cnt('ident', 'fired'), cat=cnt('cat', 'fired'))


def rate(x, n, ci=False):
    if not n: return '   --  '
    s = f"{x / n:.3f}"
    if ci: lo, hi = wilson(x, n); s += f" [{lo:.3f}, {hi:.3f}]"
    return s


# ------------------------------------------------------------------ S-II
def meets(p):
    """The three acceptance criteria: accuracy >= 0.95 on its Wilson lower bound, catastrophic rate <= 0.02 on its Wilson
    upper bound, and identification >= 0.95 on its measured fraction."""
    return (wilson(p['acc'], p['n'])[0] >= 0.95 and p['fired'] and p['ident'] / p['fired'] >= 0.95
            and wilson(p['cat'], p['fired'])[1] <= 0.02)


def capacity(arm):
    """Largest load up to which every tested load meets the criteria (None if the smallest load fails)."""
    best = None
    for M in LOADS:
        if not meets(pooled(arm, M)): break
        best = M
    return best


print("Table S-II. Memory capacity test: deployed code (256 cells, k = 8, three code draws)")
print(f"{'M':>4} {'n':>5} | {'accuracy [95% CI]':<22} {'ident':>6} | {'catastrophic [95% CI]':<22} | criteria met")
for M in LOADS:
    d = pooled('sparse_rand_k8_coinc', M)
    print(f"{M:>4} {d['n']:>5} | {rate(d['acc'], d['n'], True):<22} {rate(d['ident'], d['fired']):>6} | "
          f"{rate(d['cat'], d['fired'], True):<22} | {'yes' if meets(d) else 'no'}")
print(f"capacity: {capacity('sparse_rand_k8_coinc')} places")

# ------------------------------------------------------------------ S-III
ALT = [('one-cell', 'onehot_add'), ('unsparsified', 'dense_rand_k8_coinc'), ('appearance', 'sparse_appear_k8_coinc'),
       ('k = 2', 'sparse_rand_k2_coinc'), ('k = 4', 'sparse_rand_k4_coinc')]
print("\nTable S-III. Memory capacity test: alternative memories and code sizes (accuracy / catastrophic rate)")
print(f"{'M':>4} | " + ' | '.join(f"{lab:^11}" for lab, _ in ALT))
for M in LOADS:
    cells = []
    for _, arm in ALT:
        p = pooled(arm, M)
        cells.append(f"{rate(p['acc'], p['n'])} {rate(p['cat'], p['fired'])}")
    print(f"{M:>4} | " + ' | '.join(cells))
caps = []
for _, arm in ALT:
    c = capacity(arm)
    caps.append(f"{('below ' + str(LOADS[0])) if c is None else str(c):^11}")
print(f"{'cap.':>4} | " + ' | '.join(caps))

# ------------------------------------------------------------------ S-IV


def tci(v, fmt):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0: return '---', 0
    if len(v) == 1: return f"{v[0]:{fmt}}", 1
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    if h == 0: return f"{v.mean():{fmt}}", len(v)
    return f"{v.mean():{fmt}}+-{h:{fmt}}", len(v)


print("\nTable S-IV. Window sweep: bounded window alone (12 seeds); mean +- 95% t half-width")
sweep = {25: [r for r in rows if r['phase'] == 'core' and r['arm'] == 'l1only']}
for W in (30, 35, 40, 60):
    sweep[W] = [r for r in rows if r['phase'] == 'cliff' and r['arm'] == f'l1only_cap{W}']
print(f"{'W':>3} {'seeds':>5} {'closures/run':>13} {'lost/run':>9} {'error at lost (cm) (seeds)':>27} {'gate rej./run':>14} {'position error (cm)':>20} {'ATE (cm)':>11}")
for W in sorted(sweep):
    rs = sweep[W]
    lost = tci([r['n_lost'] for r in rs], '.0f')[0]
    atl, n_atl = tci([r['at_lost'] for r in rs], '.1f')
    atl = f"{atl} ({n_atl})" if n_atl else atl
    print(f"{W:>3} {len(rs):>5} {tci([r['loops'] for r in rs], '.1f')[0]:>13} {lost:>9} {atl:>27} "
          f"{tci([r['l1_gate_rej'] for r in rs], '.0f')[0]:>14} {tci([r['raw'] for r in rs], '.1f')[0]:>20} {tci([r['ate'] for r in rs], '.1f')[0]:>11}")
print("  closures = in-window loop closures served (loops); lost = evicted revisits the window cannot close (n_lost);")
print("  error at lost = position error at those revisits (at_lost) over the seeds that have them; gate rej. = candidate")
print("  closures rejected by the 0.25 m metric gate (l1_gate_rej); position error = unaligned per-keyframe error (raw).")

# ------------------------------------------------------------------ S-V
K_HOT, N_GRID, N_SETTLE, PASSES = 8, 579, 12, 3
per_event = {
    'recall':  ('Recall (read rows, inject)', N_GRID * N_SETTLE + K_HOT * N_GRID),
    'hebb':    ('In-window instar write', K_HOT * N_GRID),
    'consol':  ('Consolidation (three passes)', PASSES * (N_GRID * N_SETTLE + K_HOT * N_GRID)),
    'correct': ('Post-relaxation correction (per place)', N_GRID * N_SETTLE + K_HOT * N_GRID),
    'create':  ('Creation write', K_HOT * N_GRID),
}
D = [r for r in rows if r['phase'] == 'core' and r['arm'] == 'deployed']
tot = sum(r['sop_total'] for r in D)
print(f"\nTable S-V. Layer-2 operations per event and share of the Layer-2 budget (core:deployed, {len(D)} seeds pooled)")
print(f"{'update':<40} {'ops/event':>10} {'share':>7}")
for k, (name, ops) in per_event.items():
    print(f"{name:<40} {ops:>10,d} {100 * sum(r['sop_' + k] for r in D) / tot:>6.0f}%")
# consistency of the ledger with the per-event counts
chk_recall = {r['sop_recall'] / r['loops'] for r in D if r['loops']}
chk_consol = {r['sop_consol'] / r['consol_n'] for r in D if r.get('consol_n')}
chk_create = {r['sop_create'] / r['nodes'] for r in D if r['nodes']}
secs = 3500 * 0.02
print(f"  ledger check: sop_recall/closure = {sorted(chk_recall)}, sop_consol/eviction = {sorted(chk_consol)}, "
      f"sop_create/place = {sorted(chk_create)} (two creation writes per place)")
print(f"  Layer-2 total: {np.mean([r['sop_total'] for r in D]):,.0f} operations per run = {np.mean([r['sop_total'] for r in D]) / secs:,.0f} per second of simulated time")

# Settling steps that Layer 2 adds to the grid attractor. Every recall settles the live attractor for N_SETTLE steps,
# and every consolidation pass and post-relaxation correction settles an attractor copy for N_SETTLE steps. A settling
# step applies each module's recurrent matrix (N_m x N_m, N_m = 121, 169, 289 cells) but not the two velocity-shift
# matrices of a live update. Operations are counted as in the grid-layer accounting of the navigation stack the
# estimator is built on: presynaptic rate x fan-out, with the grid layer's measured mean rate of 4.596 Hz per cell
# (peak-normalised at 100 Hz).
GRID_RATE_HZ, DT = 4.596, 0.02
REC_SYN = sum((s * s) ** 2 for s in (11, 13, 17))                     # synapses of the three recurrent matrices
per_step = GRID_RATE_HZ * REC_SYN * DT                                # operations per settling step
unit = N_GRID * N_SETTLE + K_HOT * N_GRID                             # ledger operations per settle event
settles = np.array([(r['sop_recall'] + r['sop_consol'] + r['sop_correct']) / unit for r in D])
steps = settles * N_SETTLE
print(f"  settling steps added to the grid attractor: {steps.mean():,.0f} per run (recall "
      f"{np.mean([r['sop_recall'] / unit for r in D]) * N_SETTLE:,.0f}, consolidation {np.mean([r['sop_consol'] / unit for r in D]) * N_SETTLE:,.0f}, "
      f"correction {np.mean([r['sop_correct'] / unit for r in D]) * N_SETTLE:,.0f}), against {int(secs / DT):,d} live attractor steps")
print(f"  at {per_step:,.0f} recurrent operations per settling step: {steps.mean() * per_step / secs:,.0f} per second of simulated time; "
      f"Layer 2 in total {(steps.mean() * per_step + np.mean([r['sop_total'] for r in D])) / secs:,.0f} per second")
