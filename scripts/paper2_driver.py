#!/usr/bin/env python3
"""Simulation campaign driver.

Runs every (phase, arm, seed) configuration sequentially in one process and appends one JSON row per run to
<out>/paper2_results.jsonl, plus the per-keyframe and per-closure arrays to <out>/npz/. The driver is
resumable: rows whose key is already in the results file are skipped.

Phases and arms (all derive from BASE, the deployed design):
  core:       deployed | l1only (window alone: recall of evicted places off)
  components: eta0 (no in-window learning) | noverify (distance-free verifier off) | nogate (0.20 m
              relative-displacement sequence check off) | noverify_nogate | noconsol (no consolidating write
              at eviction) | symstore (evicted recall from the retained symbolic position)
  store:      readout (patterns are direct copies of Layer 1's positions) | poison_evicted (deletion test)
  cliff:      window alone at W = 30 / 35 / 40 / 60 (W = 25 is core:l1only)
  capacity:   onehot_N256 | onehot_N32 | khot_N32 | onehot_N48 | khot_N48 (memory size N place cells)
  length:     deployed / l1only on schedules ~1.8x (len2) and ~3.4x (len4) longer, first 3 seeds

Usage:
  python scripts/paper2_driver.py                                   # full campaign (240 runs)
  python scripts/paper2_driver.py --phases core --seeds 42          # one phase, one seed
  python scripts/paper2_driver.py --phases length --arms deployed_len2 --seeds 42 --out /tmp/check
Run one simulation process at a time (each uses ~5 GB RAM).
"""
import os, sys, json, time, argparse, hashlib, signal, traceback
RUN_TIMEOUT_S = int(os.environ.get('RUN_TIMEOUT_S', '900'))   # per-run wall clock; on expiry the stack is printed and the run skipped
os.environ.setdefault('SLAM_OBS_SLICE', '800')
for _k in ('L2P_RA', 'L2P_RB', 'L2P_ALTS'): os.environ.pop(_k, None)
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, ROOT)
import numpy as np
from slam import weight_l2_slam as WL
from slam.weight_l2_slam import weight_l2_stream, DIAG
from slam.attractor_fusion import S

SEEDS = [42 + 111 * i for i in range(12)]
BASE = dict(cap=25, course='l2probe', kf_gt=0.04, use_l2=True, eta=0.3, l2_learn='hebb_correct',
            l2_code='khot', k_hot=8, n_cells=256, sparsify=0.5)   # deployed design
T_TAIL = 20
ARMS = {  # name -> (kwargs override, env override, patch, steps, seeds)
 'core': {
   'deployed': (dict(), {}, None, 3500, SEEDS),
   'l1only':   (dict(use_l2=False), {}, None, 3500, SEEDS)},
 'components': {
   'eta0':            (dict(eta=0.0), {}, None, 3500, SEEDS),
   'noverify':        (dict(), {}, 'noverify', 3500, SEEDS),
   'nogate':          (dict(), {}, 'nogate', 3500, SEEDS),
   'noverify_nogate': (dict(), {}, 'noverify_nogate', 3500, SEEDS),
   'noconsol':        (dict(consol_eta=0.0), {}, None, 3500, SEEDS),
   'symstore':        (dict(recall_source='float'), {}, None, 3500, SEEDS)},
 'store': {
   'readout':        (dict(l2_learn='relax', eta=1.0), {}, None, 3500, SEEDS),
   'poison_evicted': (dict(poison='evicted'), {}, None, 3500, SEEDS)},
 'cliff': {
   'l1only_cap30': (dict(use_l2=False, cap=30), {}, None, 3500, SEEDS),
   'l1only_cap35': (dict(use_l2=False, cap=35), {}, None, 3500, SEEDS),
   'l1only_cap40': (dict(use_l2=False, cap=40), {}, None, 3500, SEEDS),
   'l1only_cap60': (dict(use_l2=False, cap=60), {}, None, 3500, SEEDS)},
 'capacity': {   # ~52 places on the course vs N place cells
   'onehot_N256': (dict(l2_code='onehot', n_cells=256), {}, None, 3500, SEEDS),
   'onehot_N32':  (dict(l2_code='onehot', n_cells=32), {}, None, 3500, SEEDS),
   'khot_N32':    (dict(n_cells=32), {}, None, 3500, SEEDS),
   'onehot_N48':  (dict(l2_code='onehot', n_cells=48), {}, None, 3500, SEEDS),
   'khot_N48':    (dict(n_cells=48), {}, None, 3500, SEEDS)},
 'length': {   # L2P_ALTS excursion/return alternations: 6 -> ~1.8x, 12 -> ~3.4x the base path length
   'deployed_len2': (dict(), {'L2P_ALTS': '6'}, None, 7000, SEEDS[:3]),
   'l1only_len2':   (dict(use_l2=False), {'L2P_ALTS': '6'}, None, 7000, SEEDS[:3]),
   'deployed_len4': (dict(), {'L2P_ALTS': '12'}, None, 14000, SEEDS[:3]),
   'l1only_len4':   (dict(use_l2=False), {'L2P_ALTS': '12'}, None, 14000, SEEDS[:3])},
}
ORDER = ['core', 'components', 'store', 'cliff', 'capacity', 'length']
RES = os.path.join(ROOT, 'results', 'paper2_results.jsonl'); NPZ = os.path.join(ROOT, 'results', 'npz')


def done_keys():
    if not os.path.exists(RES): return set()
    return {json.loads(l)['key'] for l in open(RES) if l.strip()}


def f(x):
    try: return float(x)
    except Exception: return float('nan')


def ate_umeyama(E, G):
    """ATE (cm): mean position error after the optimal rigid (rotation + translation) alignment of E onto G."""
    R, t = S.get_optimal_alignment_2d(E, G); al = (np.array(R) @ E.T).T + np.array(t)
    return float(np.mean(np.hypot(al[:, 0] - G[:, 0], al[:, 1] - G[:, 1]))) * 100


def metrics(e, g, cap, steps):
    """Per-run metrics (errors in cm). raw = unaligned per-keyframe position error; ate = aligned error."""
    e = np.asarray(e, float); g = np.asarray(g, float); err = np.hypot(*(e - g).T) * 100
    fin = np.isfinite(err); nonfin = int(np.sum(~fin))
    try: ate = f(ate_umeyama(e[fin], g[fin])) if fin.sum() > 10 else float('nan')
    except Exception: ate = float('nan')
    m = dict(K=int(len(e)), steps=int(steps), cap=int(cap), raw=f(np.nanmean(err)), ate=ate, n_nonfinite=nonfin)
    for k in ('loops', 'relaxes', 'l2_only', 'lost', 'l1_gate_rej', 'nodes', 'l2_evictions', 'l2_bytes',
              'l2_cell_lost', 'l2_cell_lost_inwin', 'stale_instar', 'l2_verify_rej', 'seq_rej', 'seq_rej_ev', 'seq_short', 'rej_margin', 'rej_short', 'rej_order', 'rej_noprog', 'float_recalls',
              'sop_recall', 'sop_hebb', 'sop_consol', 'sop_correct', 'sop_create', 'sop_total'):
        m[k] = int(DIAG.get(k, 0) or 0)
    def tail(kfs): v = [err[b+1:b+1+T_TAIL].mean() for b in kfs if b + 1 < len(err)]; return f(np.mean(v)) if v else float('nan')
    # closures served from Layer 2 (evicted places): error before (pose before recall) and after recall
    ev = [s for s in DIAG.get('snaps', []) if not s[3]]; bs = [int(s[0]) for s in ev]
    if bs:
        before = np.array([np.hypot(e[b,0]-s[1]-g[b,0], e[b,1]-s[2]-g[b,1])*100 for s, b in zip(ev, bs)]); after = err[bs]
        m.update(n_served=len(bs), before=f(before.mean()), after=f(after.mean()), delta=f((before-after).mean()),
                 frac_improved=f(np.mean(after < before)), catastrophic=int(np.sum(after-before > 20)), tail_served=tail(bs))
    else:
        m.update(n_served=0, before=float('nan'), after=float('nan'), delta=float('nan'), frac_improved=float('nan'), catastrophic=0, tail_served=float('nan'))
    lost = [int(b) for b in (DIAG.get('lost_kfs') or [])] if isinstance(DIAG.get('lost_kfs'), list) else []
    m.update(n_lost=len(lost), at_lost=f(err[lost].mean()) if lost else float('nan'), tail_lost=tail(lost))
    al = [a for b, a, iw in (DIAG.get('alias_list') or []) if not iw] if isinstance(DIAG.get('alias_list'), list) else []
    m.update(alias_mean=f(np.mean(al)) if al else float('nan'), alias_max=f(np.max(al)) if al else float('nan'))
    rr = [r for r in (DIAG.get('recall_rows') or []) if not r[6]] if isinstance(DIAG.get('recall_rows'), list) else []
    if rr:
        rs = np.array([r[4] for r in rr]); lr = np.array([r[5] for r in rr]); tg = np.array([r[2] for r in rr])
        m.update(row_mean=f(rs.mean()), row_max=f(rs.max()), row_first=f(rs[:max(1,len(rs)//4)].mean()), row_last=f(rs[-max(1,len(rs)//4):].mean()),
                 landrow_mean=f(lr.mean()), landrow_median=f(np.median(lr)), target_mean=f(np.nanmean(tg)))
    n = int(DIAG.get('consol_n', 0) or 0)
    if n: m.update(consol_n=n, consol_target=f(DIAG['consol_tgt_sum']/n), consol_before=f(DIAG['consol_bef_sum']/n), consol_after=f(DIAG['consol_aft_sum']/n))
    try:   # end-of-run store check: settled pattern decode (W) vs ground truth and vs the symbolic reference estimators
        W = np.asarray(DIAG['gc_wset']); G = np.asarray(DIAG['gc_plgt']); EMA = np.asarray(DIAG['gc_plema']); MED = np.asarray(DIAG['gc_plmed']); BAT = np.asarray(DIAG['gc_plbatch']); X = np.asarray(DIAG['gc_plxy'])
        pm = lambda a, b: f(np.nanmean(np.hypot(*(a - b).T)) * 100)
        m.update(W_gt=pm(W, G), EMA_gt=pm(EMA, G), MED_gt=pm(MED, G), BAT_gt=pm(BAT, G), plxy_gt=pm(X, G), vsEMA=pm(W, EMA), vsMED=pm(W, MED), vsBATCH=pm(W, BAT))
        ev_mask = np.asarray(DIAG.get('gc_evicted', []), bool)
        if ev_mask.size == len(W) and ev_mask.any(): m.update(W_gt_evicted=pm(W[ev_mask], G[ev_mask]), n_evicted_places=int(ev_mask.sum()))
    except Exception as ex: m['store_err'] = str(ex)[:80]
    m['seq_tol'] = str(DIAG.get('seq_tol')); m['recall_source'] = DIAG.get('recall_source'); m['desc_eta'] = DIAG.get('desc_eta')
    ng = int(DIAG.get('nullgap_n', 0) or 0); m['nullgap'] = f(DIAG['nullgap_sum']/ng) if ng else float('nan')
    nl = np.asarray(DIAG.get('nullgap_list') or [], float)
    if nl.size: m.update(nullgap_median=f(np.median(nl)), nullgap_p95=f(np.percentile(nl, 95)), nullgap_max=f(nl.max()))
    return m


def run_one(phase, arm, seed):
    kw, env, patch, steps, _ = ARMS[phase][arm]
    for k in ('L2P_RA', 'L2P_RB', 'L2P_ALTS'): os.environ.pop(k, None)
    os.environ.update(env)
    saved = WL._l2_verify; saved_tol = WL.SEQ_TOL
    if patch in ('noverify', 'noverify_nogate'): WL._l2_verify = lambda b, mp, best, second: True
    if patch in ('nogate', 'noverify_nogate'): WL.SEQ_TOL = float('inf')       # relative-displacement sequence check off
    cfg = dict(BASE); cfg.update(kw)
    t0 = time.time()
    class RunTimeout(Exception): pass
    def _on_alarm(signum, frame):
        print(f"[driver] RUN_TIMEOUT {phase}:{arm}:{seed} after {RUN_TIMEOUT_S}s -- Python stack at expiry:", flush=True)
        print(''.join(traceback.format_stack(frame)[-14:]), flush=True); raise RunTimeout()
    old_h = signal.signal(signal.SIGALRM, _on_alarm); signal.alarm(RUN_TIMEOUT_S)
    try:
        st, e, g, H, fired = weight_l2_stream(seed, steps, **cfg)
    except RunTimeout:
        print(f"[driver] SKIPPED {phase}:{arm}:cap{cfg['cap']}:{seed} (timeout)", flush=True); return None
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_h)
        WL._l2_verify = saved; WL.SEQ_TOL = saved_tol
        for k in env: os.environ.pop(k, None)
    m = metrics(e, g, cfg['cap'], steps); m.update(key=f"{phase}:{arm}:cap{cfg['cap']}:{seed}", phase=phase, arm=arm, seed=seed, cfg={k: (v if isinstance(v,(int,float,str,bool)) else str(v)) for k, v in cfg.items()}, env=env, secs=round(time.time()-t0))
    _L = lambda k: np.asarray(DIAG.get(k) if isinstance(DIAG.get(k), list) else [], dtype=float)
    np.savez_compressed(os.path.join(NPZ, f"{phase}_{arm}_cap{cfg['cap']}_s{seed}.npz"), est=np.asarray(e), gt=np.asarray(g),
                        wset=np.asarray(DIAG.get('gc_wset', [])), plgt=np.asarray(DIAG.get('gc_plgt', [])), plxy=np.asarray(DIAG.get('gc_plxy', [])),
                        snaps=_L('snaps'), alias_list=_L('alias_list'), recall_rows=_L('recall_rows'), lost_kfs=_L('lost_kfs'), nullgap_list=_L('nullgap_list'))
    m['est_sha'] = hashlib.sha1(np.asarray(e, float).round(6).tobytes()).hexdigest()[:12]
    with open(RES, 'a') as fh: fh.write(json.dumps(m) + '\n')
    return m


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--phases', default=','.join(ORDER), help='comma-separated phases (default: all, in campaign order)')
    ap.add_argument('--arms', default='', help='comma-separated arm names to keep within the selected phases (default: all)')
    ap.add_argument('--seeds', default='', help='comma-separated seeds to keep (default: each arm\'s seed list)')
    ap.add_argument('--out', default='', help='output directory for paper2_results.jsonl and npz/ (default: results/)')
    ap.add_argument('--smoke', action='store_true', help='400-step core arms, seed 42, written to results/smoke/')
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, 'results', 'smoke' if a.smoke else '')
    RES = os.path.join(out, 'paper2_results.jsonl'); NPZ = os.path.join(out, 'npz'); os.makedirs(NPZ, exist_ok=True)
    if a.smoke:
        ARMS = {'core': {k: (v[0], v[1], v[2], 400, [42]) for k, v in ARMS['core'].items()}}; phases = ['core']
        if os.path.exists(RES): os.remove(RES)
    else:
        phases = [p for p in a.phases.split(',') if p in ARMS]
    arms = set(x for x in a.arms.split(',') if x); seeds = set(int(x) for x in a.seeds.split(',') if x)
    todo = [(p, arm, s) for p in phases for arm, v in ARMS[p].items() for s in v[4]
            if (not arms or arm in arms) and (not seeds or s in seeds)]
    def _cap(p, arm):
        kw = ARMS[p][arm][0]; return kw.get('cap', BASE['cap'])
    done = done_keys(); todo = [t for t in todo if f"{t[0]}:{t[1]}:cap{_cap(t[0],t[1])}:{t[2]}" not in done]
    print(f"[driver] {len(todo)} runs to do ({len(done)} already in {RES}).", flush=True)
    t_start = time.time(); n_done = 0
    for p, arm, s in todo:
        m = run_one(p, arm, s); n_done += 1
        if m is None: continue                      # timed-out run: skipped, no row written
        el = time.time() - t_start; eta = el / n_done * (len(todo) - n_done)
        print(f"[{time.strftime('%H:%M')}] {m['key']:<32} RAW {m['raw']:5.1f} ATE {m['ate']:5.1f} | served {m['n_served']:>2} lost {m['n_lost']:>2} "
              f"before {m['before']:5.1f} after {m['after']:5.1f} cat {m['catastrophic']} | row {m.get('row_mean', float('nan')):5.1f} landrow {m.get('landrow_mean', float('nan')):5.1f} "
              f"| {m['secs']}s | ETA {eta/3600:.1f} h", flush=True)
    print("[driver] SMOKE_DONE" if a.smoke else "[driver] CAMPAIGN_DONE", flush=True)
