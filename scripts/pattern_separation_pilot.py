#!/usr/bin/env python3
"""Synthetic capacity test of the Layer-2 store (Fig. 4A and Supplementary Tables S-II and S-III).

M places are stored in an N = 256 cell x 579 grid-cell store and each is read back through the deployed
read-out: the grid attractor is seeded with a bump at live amplitude displaced 0.05-0.45 m from the place,
the recalled pattern is injected (gain K_REC, N_SETTLE settle steps) and the position is decoded by
residue consensus.

Arena: places uniform in a 5.5 x 5.5 m square, at least 0.085 m apart (the in-loop nearest-neighbour place
spacing); the decoder span is widened to [-0.5, 6.5) m for this test (module periods 0.6 / 0.8 / 1.06 m are
unambiguous to ~7 m). Loads M = 64 ... 768 over three independent code draws; every place is evaluated at
M <= 128 and at M = 256, 512, 640, and 160 places per draw at the other loads.

Arms (stored pattern | code, k | storage | read-out):
  onehot_lru              dense  | one cell per place, overwritten least-recently-used beyond N
  onehot_add              dense  | one cell per place, additive (cells shared once M > N)
  dense_rand_k8_sum / dense_rand_k8_coinc     dense patterns, random 8-cell codes
  sparse_onehot_add       sparse | one cell per place, additive
  sparse_rand_k{2,4,8}_coinc                  sparse patterns, random k-cell codes, coincidence read-out
                                              (sparse_rand_k8_coinc is the deployed code)
  sparse_rand_k8_sum      sparse | random 8-cell code, sum read-out
  sparse_appear_k8_coinc  sparse | 8-cell codes derived from a smooth appearance-like feature of position
  dense_rand_k8_ema_sum   dense  | random 8-cell code, EMA storage (eta = 0.3), sum read-out
'Dense' = the settled bump; 'sparse' = cells >= 0.5 of each module's peak. Coincidence read-out = elementwise
minimum over the k per-module-normalised rows.

Metrics per (arm, M, draw): prec10 = P(error < 0.10 m | fired) and prec10_all = the same over all evaluated
places (landing precision); ident = P(decode nearer the stored place than any other | fired) (identification);
cat = P(error > arrival distance + 0.20 m | fired) (catastrophic, the in-loop criterion); nomove, err_med,
residue; mf_prec10 = prec10 of a matched filter on the same stored pattern; prec10_oldest / prec10_newest.

Capacity bar (printed for the full run): Wilson 95% lower bound of prec10 >= C - 0.05 (C = one-cell LRU
precision at M <= N), Wilson upper bound of cat <= 0.02, and ident >= 0.95.

Usage:
  python scripts/pattern_separation_pilot.py                          # full run -> results/capacity_test.json (~30 min)
  python scripts/pattern_separation_pilot.py --loads 64,768 --seeds 1 --out /tmp/check.json
"""
import os, sys, json, time, argparse, math, collections
os.environ.setdefault('SLAM_OBS_SLICE', '800'); os.environ.setdefault('JAX_PLATFORMS', 'cpu'); os.environ['MPLBACKEND'] = 'Agg'
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, ROOT)
import numpy as np
import jax.numpy as jnp
from slam import weight_l2_slam as WL
from slam.weight_l2_slam import bump_pattern_flat, inject_flat, crt_decode_res, bump_flat, make_system, CANN_SIZES, K_REC, N_SETTLE, N_PLACE_CELLS
from slam.attractor_fusion import S, jax
N = N_PLACE_CELLS; G = WL.N_GRID
OUT = os.path.join(ROOT, 'results')
ARENA = (0.25, 5.75); MINSEP = 0.085; DISP = (0.05, 0.45); ETA = 0.3; N_EVAL = 160; MF_STEP = 0.04
WL._CRT_XS = np.arange(-0.5, 6.5, 0.002)           # decoder span for this test (in-loop: [-0.5, 2.5) for the 2 x 2 m arena)
LOADS = [64, 128, 192, 256, 320, 384, 448, 512, 640, 768]; DECISION = {256, 512, 640}; SEEDS = [0, 1, 2]   # every place is evaluated at the DECISION loads
OFFS = np.cumsum([0] + [c * c for c in CANN_SIZES])

def blocks(flat): return [np.asarray(flat[OFFS[i]:OFFS[i + 1]], np.float32).reshape(c, c) for i, c in enumerate(CANN_SIZES)]
def per_module_norm(flat):
    out = np.zeros(G, np.float32)
    for i in range(3):
        b = np.asarray(flat[OFFS[i]:OFFS[i + 1]], np.float32); m = float(b.max()); out[OFFS[i]:OFFS[i + 1]] = b / m if m > 1e-9 else 0.0
    return out

_SCR = None; A_LIVE = None
def scratch():
    global _SCR
    if _SCR is None: _SCR = make_system(); _SCR.reset(1)
    return _SCR
def _set_state(s, flat_u):
    for i, blk in enumerate(blocks(flat_u)):
        s.pose._u_canns[i] = jnp.asarray(blk[None]); s.pose._r_canns[i] = jnp.asarray(np.maximum(blk, 0.0)[None])
def measure_live_amplitude(n_steps=60):
    """Live bump amplitude: steady-state u peak per module of a path-integrating pose (make_system, initialize_pose and
    n_steps forward steps on the 'explore' course)."""
    s = make_system(); env = S.LiveEnvironment(jax.random.PRNGKey(42), chunk_size=200, course_type='explore')
    ev, kin, tof, p0, th0, _ = env.step(); s.initialize_pose(jnp.array([p0]), jnp.array([th0]))
    for _ in range(n_steps):
        ev, kin, tof, *_ = env.step(); s.forward_step(jnp.array([ev]), jnp.array([kin]), jnp.array([tof]))
    return np.array([float(np.asarray(s.pose._u_canns[i]).max()) for i in range(3)])
def seed_live(xy):
    """A live-amplitude bump at xy: per module, A_LIVE x the rendered bump."""
    s = scratch(); flat = bump_pattern_flat(np.asarray(xy, float)).copy()
    for i in range(3): flat[OFFS[i]:OFFS[i + 1]] *= A_LIVE[i]
    _set_state(s, flat); return s
def live_pattern(xy):
    """The pattern a consolidated row holds: the live bump at xy settled under its own rendered target for N_SETTLE steps,
    max-normalised."""
    s = seed_live(xy); inject_flat(s.pose, bump_pattern_flat(np.asarray(xy, float)), K=K_REC, n_settle=N_SETTLE)
    f = np.asarray(bump_flat(s.pose)).ravel(); return (f / max(1e-9, f.max())).astype(np.float32)
def sparsify(flat, frac=0.5):
    out = np.zeros_like(flat)
    for i in range(3):
        b = flat[OFFS[i]:OFFS[i + 1]]; m = b.max(); out[OFFS[i]:OFFS[i + 1]] = np.where(b >= frac * m, b, 0.0) if m > 0 else 0.0
    return out

def recall(pattern, true_xy, rng):
    """Deployed read-out on a displaced live bump: per-module-normalised pattern injected with K_REC for N_SETTLE steps; residue decode."""
    d = rng.uniform(*DISP); ang = rng.uniform(0, 2 * np.pi); seed_xy = true_xy + d * np.array([np.cos(ang), np.sin(ang)])
    s = seed_live(seed_xy); p = per_module_norm(pattern)
    if p.max() <= 0: return dict(err=np.nan, d=d, res=np.nan, xy=None, fired=False)
    inject_flat(s.pose, p, K=K_REC, n_settle=N_SETTLE); xy, (rx, ry) = crt_decode_res(bump_flat(s.pose))
    return dict(err=float(np.hypot(*(xy - true_xy))), d=float(d), res=float(np.hypot(rx, ry)), xy=xy, fired=True)

# ---------------- matched filter on the stored pattern (store ceiling, independent of the attractor) ----------------
_MF = None
def mf_templates():
    global _MF
    if _MF is None:
        g = np.arange(ARENA[0], ARENA[1] + 1e-9, MF_STEP); X, Y = np.meshgrid(g, g, indexing='xy'); P = np.c_[X.ravel(), Y.ravel()]
        T = np.stack([per_module_norm(bump_pattern_flat(p)) for p in P]).astype(np.float32); _MF = (P, T)
    return _MF
def mf_decode(pattern):
    P, T = mf_templates(); return P[int(np.argmax(T @ per_module_norm(pattern)))]

# ---------------- places, codes, storage, read-outs ----------------
def sample_places(n, rng, max_batches=200):
    """Random sequential packing at MINSEP separation, capped at max_batches candidate batches."""
    pts = []; P = np.zeros((0, 2))
    for _ in range(max_batches):
        cand = rng.uniform(ARENA[0], ARENA[1], size=(4 * n, 2))
        for c in cand:
            if len(P) == 0 or np.min(np.hypot(*(P - c).T)) >= MINSEP: pts.append(c); P = np.asarray(pts)
            if len(pts) >= n: return P
    print(f"  WARNING: only {len(pts)} of {n} places fit at {MINSEP} m separation; loads above that are skipped", flush=True); return P
def codes_onehot(M): return [np.array([p % N]) for p in range(M)]
def codes_random(M, k, rng): return [np.sort(rng.choice(N, k, replace=False)) for _ in range(M)]
def codes_appearance(M, k, rng, pos):
    """Appearance-like codes: a smooth 64-D random-Fourier feature of position (+ view noise) -> random projection -> top-k.
    Nearby places share cells."""
    Wf = rng.normal(scale=2.0, size=(2, 32)); bf = rng.uniform(0, 2 * np.pi, 32)
    feat = np.concatenate([np.cos(pos @ Wf + bf), np.sin(pos @ Wf + bf)], axis=1) + 0.15 * rng.normal(size=(M, 64))
    P = rng.normal(size=(N, 64)); return [np.sort(np.argsort(-(P @ f))[:k]) for f in feat]
def store(codes, pats, rule):
    W = np.zeros((N, G), np.float32); owner = np.full(N, -1)
    for p, (cs, pat) in enumerate(zip(codes, pats)):
        for c in cs:
            if rule == 'additive': W[c] += pat
            elif rule == 'ema':    W[c] = (1.0 - ETA) * W[c] + ETA * pat
            elif rule == 'lru':    W[c] = pat
            owner[c] = p
    return W, owner
def readout(W, cs, how):
    R = W[cs]
    if how == 'sum': return R.sum(axis=0)
    # coincidence: per-module-normalise each row, then the elementwise minimum over the k rows (a cell survives only if
    # it is active in all k rows)
    Rn = np.stack([per_module_norm(r) for r in R]); return Rn.min(axis=0)

def wilson(x, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = x / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))

ARMS = [  # name, pattern, code, k, storage, read-out
    ('onehot_lru', 'dense', 'onehot', 1, 'lru', 'sum'), ('onehot_add', 'dense', 'onehot', 1, 'additive', 'sum'),
    ('dense_rand_k8_sum', 'dense', 'random', 8, 'additive', 'sum'), ('dense_rand_k8_coinc', 'dense', 'random', 8, 'additive', 'coinc'),
    ('sparse_onehot_add', 'sparse', 'onehot', 1, 'additive', 'sum'),
    ('sparse_rand_k2_coinc', 'sparse', 'random', 2, 'additive', 'coinc'), ('sparse_rand_k4_coinc', 'sparse', 'random', 4, 'additive', 'coinc'),
    ('sparse_rand_k8_coinc', 'sparse', 'random', 8, 'additive', 'coinc'), ('sparse_rand_k8_sum', 'sparse', 'random', 8, 'additive', 'sum'),
    ('sparse_appear_k8_coinc', 'sparse', 'appear', 8, 'additive', 'coinc'),
    ('dense_rand_k8_ema_sum', 'dense', 'random', 8, 'ema', 'sum')]
DECISION_ARM = 'sparse_rand_k8_coinc'

def evaluate(name, W, owner, codes, pos, pats, rule, how, rng, n_eval):
    M = len(codes); idx = np.sort(rng.choice(M, min(n_eval, M), replace=False)); out = []
    for p in idx:
        cs = codes[p]
        if rule == 'lru' and owner[cs[0]] != p: out.append(dict(p=int(p), fired=False, err=np.nan, d=np.nan, res=np.nan, mf=np.nan, ident=False)); continue
        pat = readout(W, cs, how); r = recall(pat, pos[p], rng)
        if not r['fired']: out.append(dict(p=int(p), fired=False, err=np.nan, d=r['d'], res=np.nan, mf=np.nan, ident=False)); continue
        dd = np.hypot(*(pos - r['xy']).T); ident = bool(np.argmin(dd) == p)
        mf_err = float(np.hypot(*(mf_decode(readout(W, cs, 'sum')) - pos[p])))
        out.append(dict(p=int(p), fired=True, err=r['err'], d=r['d'], res=r['res'], mf=mf_err, ident=ident))
    f = [o for o in out if o['fired']]; nf = len(f); n = len(out)
    err = np.array([o['err'] for o in f]); d = np.array([o['d'] for o in f]); mf = np.array([o['mf'] for o in f])
    q = idx / max(1, M - 1); old = np.array([q[i] < 0.25 and out[i]['fired'] for i in range(n)]); new = np.array([q[i] >= 0.75 and out[i]['fired'] for i in range(n)])
    def prec_mask(mask): e = np.array([out[i]['err'] for i in range(n) if mask[i]]); return float(np.mean(e < 0.10)) if len(e) else np.nan
    prec10 = float(np.mean(err < 0.10)) if nf else np.nan; cat = float(np.mean(err > d + 0.20)) if nf else np.nan
    m = dict(arm=name, M=M, n=n, fired=nf, lost=n - nf, prec10=prec10, prec10_all=float(np.sum(err < 0.10) / n) if n else np.nan,
             prec10_ci=wilson(int(np.sum(err < 0.10)), nf), cat=cat, cat_ci=wilson(int(np.sum(err > d + 0.20)), nf),
             nomove=float(np.mean(np.abs(err - d) < 0.05)) if nf else np.nan, ident=float(np.mean([o['ident'] for o in f])) if nf else np.nan,
             err_med=float(np.median(err)) if nf else np.nan, residue=float(np.nanmean([o['res'] for o in f])) if nf else np.nan,
             mf_prec10=float(np.mean(mf < 0.10)) if nf else np.nan, prec10_oldest=prec_mask(old), prec10_newest=prec_mask(new))
    return m

def main():
    global A_LIVE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true', help='one load (48), one draw, 6 places')
    ap.add_argument('--loads', type=str, default='', help='comma-separated loads M (the place pool is drawn for the largest)')
    ap.add_argument('--seeds', type=int, default=0, help='number of code draws (default 3)')
    ap.add_argument('--neval', type=int, default=0, help='places evaluated per (arm, load) outside the decision loads')
    ap.add_argument('--out', type=str, default='', help='output JSON (default: results/capacity_test.json for the full run)')
    a = ap.parse_args()
    loads, seeds, n_eval = LOADS, SEEDS, N_EVAL
    if a.smoke: loads, seeds, n_eval = [48], [0], 6
    if a.loads: loads = [int(x) for x in a.loads.split(',')]
    if a.seeds: seeds = list(range(a.seeds))
    if a.neval: n_eval = a.neval
    full = not (a.smoke or a.loads or a.seeds or a.neval)
    out_path = a.out or os.path.join(OUT, 'capacity_test.json' if full else 'capacity_test_partial.json')
    t0 = time.time(); A_LIVE = measure_live_amplitude(); print(f"A_LIVE (steady-state u peak per module) = {np.round(A_LIVE, 2)}  [{time.time()-t0:.0f}s]", flush=True)
    # ---------- positive control: a full-amplitude row relocates a live bump, a 0.30-amplitude row does not ----------
    rng = np.random.RandomState(12345); x0 = np.array([2.5, 2.5]); row = live_pattern(x0); ok1 = []; ok0 = []
    for _ in range(16 if not a.smoke else 4):
        ang = rng.uniform(0, 2 * np.pi); sxy = x0 + 0.30 * np.array([np.cos(ang), np.sin(ang)])
        s = seed_live(sxy); inject_flat(s.pose, row, K=K_REC, n_settle=N_SETTLE); ok1.append(np.hypot(*(crt_decode_res(bump_flat(s.pose))[0] - x0)) < 0.10)
        s = seed_live(sxy); inject_flat(s.pose, 0.30 * row, K=K_REC, n_settle=N_SETTLE); ok0.append(np.hypot(*(crt_decode_res(bump_flat(s.pose))[0] - x0)) < 0.10)
    print(f"positive control: full-amplitude row relocates live bump {np.mean(ok1):.2f} (need >= 0.90); 0.30-amplitude row relocates {np.mean(ok0):.2f} (need <= 0.20)", flush=True)
    if not a.smoke and not (np.mean(ok1) >= 0.90 and np.mean(ok0) <= 0.20): print("ABORT: seeding does not reproduce the in-loop amplitude sensitivity"); sys.exit(2)
    mf_templates(); print(f"MF templates ready [{time.time()-t0:.0f}s]", flush=True)
    rows = []
    for sd in seeds:
        rng = np.random.RandomState(1000 + sd); pool = sample_places(max(loads), rng)
        nn = np.array([np.sort(np.hypot(*(pool - p).T))[1] for p in pool]); print(f"seed {sd}: {len(pool)} places, NN spacing median {np.median(nn):.3f} m (min {nn.min():.3f})", flush=True)
        dense = [live_pattern(p) for p in pool]; sparse = [sparsify(f) for f in dense]
        print(f"  patterns built: dense mass/module1 {np.mean([f[:121].sum() for f in dense]):.1f} cells, sparse {np.mean([f[:121].sum() for f in sparse]):.1f} cells  [{time.time()-t0:.0f}s]", flush=True)
        crng = np.random.RandomState(7 + sd)
        ks_used = sorted({k for _, _, code, k, _, _ in ARMS if code == 'random'}); codes_all = {('random', k): codes_random(max(loads), k, crng) for k in ks_used}; codes_all[('onehot', 1)] = codes_onehot(max(loads))
        if any(code == 'appear' for _, _, code, _, _, _ in ARMS):
            codes_all[('appear', 8)] = codes_appearance(max(loads), 8, crng, pool)
            used = len(set(int(c) for cs in codes_all[('appear', 8)] for c in cs)); print(f"  appearance codes use {used}/{N} cells", flush=True)
        for M in loads:
            if M > len(pool): print(f'  skip M={M} (pool has {len(pool)})', flush=True); continue
            pos = pool[:M]
            for name, pat, code, k, rule, how in ARMS:
                pats = (dense if pat == 'dense' else sparse)[:M]; codes = codes_all[(code, k)][:M]
                W, owner = store(codes, pats, rule)
                m = evaluate(name, W, owner, codes, pos, pats, rule, how, np.random.RandomState(99 + sd), M if (M in DECISION and not a.smoke) else n_eval)
                m.update(seed=sd, nn_median=float(np.median(nn[:M]))); rows.append(m)
                print(f"[{time.time()-t0:6.0f}s] s{sd} M={M:4d} {name:24s} prec10 {m['prec10']:.3f} ident {m['ident']:.3f} cat {m['cat']:.3f} nomove {m['nomove']:.3f} med {m['err_med']:.3f} MF {m['mf_prec10']:.3f} lost {m['lost']:3d}", flush=True)
        json.dump(rows, open(out_path, 'w'), indent=1)
    print('results ->', out_path)
    # ---------- summary ----------
    agg = collections.defaultdict(list)
    for r in rows: agg[(r['arm'], r['M'])].append(r)
    def g(arm, M, key): v = [x[key] for x in agg[(arm, M)] if x[key] is not None and np.isfinite(x[key])]; return float(np.mean(v)) if v else np.nan
    def pooled(arm, Ms, num, den):
        x = sum(int(round(r[num] * r[den])) if np.isfinite(r[num]) else 0 for M in Ms for r in agg[(arm, M)]); n = sum(r[den] for M in Ms for r in agg[(arm, M)]); return x, n
    for key in ('prec10', 'ident', 'cat', 'mf_prec10'):
        print(f"\n=== {key} (mean over code draws) ===\n{'arm':24s} " + ' '.join(f"M={M:<5d}" for M in loads))
        for name, *_ in ARMS: print(f"{name:24s} " + ' '.join(f"{g(name, M, key):.3f}  " for M in loads))
    if full:
        cx, cn = pooled('onehot_lru', [M for M in loads if M <= N], 'prec10', 'fired'); C = cx / max(1, cn)
        dx, dn = pooled(DECISION_ARM, [512], 'prec10', 'fired'); lo_d, hi_d = wilson(dx, dn); pd = dx / max(1, dn)
        kx, kn = pooled(DECISION_ARM, [512], 'cat', 'fired'); lo_c, hi_c = wilson(kx, kn)
        nx, nn_ = pooled('onehot_add', [512], 'prec10', 'fired'); lo_n, hi_n = wilson(nx, nn_); pn = nx / max(1, nn_)
        ident512 = g(DECISION_ARM, 512, 'ident')
        pass_ = (lo_d >= C - 0.05) and (hi_c <= 0.02) and (ident512 >= 0.95) and (pd - pn > (hi_d - lo_d) / 2 + (hi_n - lo_n) / 2)
        no_cross = all(g(DECISION_ARM, M, 'prec10') <= g('onehot_lru', M, 'prec10_all') for M in loads if M > N)
        store_over = (g(DECISION_ARM, 256, 'prec10') < C - 0.05) and (g(DECISION_ARM, 256, 'mf_prec10') < C - 0.05)
        verdict = 'PASS' if pass_ else ('FAIL' if (no_cross or store_over) else 'PARTIAL')
        def mstar(arm):
            ok = []
            for M in loads:
                px, pn_ = pooled(arm, [M], 'prec10', 'fired'); cx_, cn_ = pooled(arm, [M], 'cat', 'fired')
                if pn_ and wilson(px, pn_)[0] >= C - 0.05 and wilson(cx_, cn_)[1] <= 0.02 and g(arm, M, 'ident') >= 0.95: ok.append(M)
            return max(ok) if ok else None
        print(f"\n=== CAPACITY AT 2N (M = 512): {verdict} ===")
        print(f"  ceiling C (onehot_lru, M<=256) = {C:.3f} | {DECISION_ARM}@512: prec10 {pd:.3f} [{lo_d:.3f},{hi_d:.3f}] cat {kx/max(1,kn):.3f} [{lo_c:.3f},{hi_c:.3f}] ident {ident512:.3f} | onehot_add@512 prec10 {pn:.3f} [{lo_n:.3f},{hi_n:.3f}] | MF@512 {g(DECISION_ARM,512,'mf_prec10'):.3f}")
        print("  largest M meeting the bar (prec >= C-0.05, cat <= 0.02, ident >= 0.95; Wilson bounds): " + ', '.join(f"{name}={mstar(name)}" for name, *_ in ARMS))
        print("  first M with cat > 0.02: " + ', '.join(f"{name}={next((M for M in loads if g(name,M,'cat')>0.02), None)}" for name, *_ in ARMS))
        print(f"  EMA storage horizon H = (N/k) ln k / (-ln(1-eta)) = {(N/8)*math.log(8)/(-math.log(1-ETA)):.0f} places; measured EMA oldest/newest prec10 @512: {g('dense_rand_k8_ema_sum',512,'prec10_oldest'):.2f}/{g('dense_rand_k8_ema_sum',512,'prec10_newest'):.2f}")

if __name__ == '__main__': main()
