"""Two-layer bounded-memory SLAM: a bounded symbolic pose-graph window (Layer 1) coupled to a fixed-size
synaptic place->grid memory (Layer 2).

Layer 1 holds the `cap` most-recently-used places (LRU eviction) with odometry-chain and loop edges, and
relaxes the window (translation only, headings fixed) on every accepted in-window loop closure.

Layer 2 is a fixed [n_cells x 579] weight matrix from place cells to the grid attractor (256 x 579 by
default). With the pattern-separated code (l2_code='khot') each place is assigned k random cells and its
sparsified grid pattern is added into those k rows. A place's pattern is learned while the place is in the
window (instar writes from the live grid bump, plus a correction toward Layer 1's relaxed estimate after
each relaxation), consolidated once when the place is evicted, and read-only afterwards. A revisit to an
evicted place is recalled by injecting the coincidence of the place's rows (elementwise minimum over its k
per-module-normalised rows) into the live grid attractor, letting the attractor settle, and decoding the
position by residue (CRT) consensus. With the one-cell code (l2_code='onehot') each place owns one row and,
once all cells are taken, the least-recently-used cell of a place outside the window is reassigned.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'; os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ.setdefault('SLAM_OBS_SLICE', '800'); os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np
from .attractor_fusion import make_system, bump_flat, cos, S, jnp, jax, MIN_TOPO, SEQ_W, SEQ_TOL
from src.snn_pose_cann import neural_field_update, CANN_SIZES, WRAP_SCALES
from .place_graph_slam import se2_between, relax_xy

K_REC = 40.0; N_SETTLE = 12   # recall injection gain and number of attractor settle steps
GATE_R = 0.25                 # metric gate for in-window closures (m)
L2_SEQ_L = 4                  # evicted-place verifier: keyframes whose matches must advance along the stored route
L2_MARGIN = 0.04              # evicted-place verifier: appearance margin of the best match over the runner-up
DIAG = {'nodes': 0, 'loops': 0, 'relaxes': 0, 'l2_only': 0, 'lost': 0,
        'l1_gate_rej': 0, 'l2_verify_rej': 0,
        'rej_margin': 0, 'rej_short': 0, 'rej_order': 0, 'rej_noprog': 0,
        'l2_cell_lost': 0, 'l2_evictions': 0, 'l2_bytes': 0, 'stale_instar': 0}   # per-run counters and records


def bump_pattern(target_xy, sigma=1.2):
    """Per-module Gaussian bump (peak 1) that represents world position target_xy on each grid module."""
    pats = []
    for c_size, scale in zip(CANN_SIZES, WRAP_SCALES):
        lx = float(target_xy[0]) % scale; ly = float(target_xy[1]) % scale
        cx = (lx / scale) * c_size; cy = (ly / scale) * c_size
        xx, yy = np.meshgrid(np.arange(c_size), np.arange(c_size), indexing='xy')
        dx = np.minimum(np.abs(xx - cx), c_size - np.abs(xx - cx))
        dy = np.minimum(np.abs(yy - cy), c_size - np.abs(yy - cy))
        pats.append(np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)).astype(np.float32))
    return pats


def inject_pattern(pose, W_mods, K=K_REC, n_settle=N_SETTLE):
    """Inject the per-module patterns W_mods as a current (gain K) into the grid attractor and let it settle for
    n_settle steps (the substrate's field update with divisive normalisation)."""
    for i, c_size in enumerate(CANN_SIZES):
        Igauss = (K * jnp.asarray(W_mods[i])).reshape(1, -1)
        for _ in range(n_settle):
            u = pose._u_canns[i].reshape(1, -1); r = jnp.maximum(0.0, u)
            u = neural_field_update(u, r + 1e-8, pose.W_cann_list[i], Igauss, dt=0.02, tau=pose.TAU_U)
            pose._u_canns[i] = u.reshape(1, c_size, c_size)
            r_raw = jnp.maximum(0.0, pose._u_canns[i]); rs = r_raw.sum(axis=(1, 2), keepdims=True)
            gi = 1.0 + pose.k_global_cann * rs; base = 1.0 + pose.k_global_cann * pose.k_global_cann_scale
            pose._r_canns[i] = (r_raw / gi) * base


def relax_window(pl_xy, pl_th, chain, loops, wl):
    """Relax the Layer-1 graph restricted to the window `wl` (a list of place ids) and write the result back
    into pl_xy. Membership is arbitrary under LRU, so place ids are re-indexed locally; the lowest id in the
    window is the fixed gauge node. Returns the relaxed place ids."""
    idx = sorted(wl)
    pos = {g: l for l, g in enumerate(idx)}
    sub_xy = [pl_xy[i] for i in idx]; sub_th = [pl_th[i] for i in idx]
    sub_chain = {pos[i]: (pos[chain[i][0]],) + tuple(chain[i][1:])
                 for i in idx if i in chain and chain[i][0] in pos}
    sub_loops = {(pos[a], pos[b]): v for (a, b), v in loops.items() if a in pos and b in pos}
    rel = relax_xy(np.array(sub_xy), np.array(sub_th), sub_chain, sub_loops)
    for r, i in enumerate(idx):
        pl_xy[i] = rel[r]
    return idx


# =================================================================================================
#  Layer 2: fixed-size place->grid synaptic memory
# =================================================================================================
N_GRID = int(sum(c * c for c in CANN_SIZES))            # 11^2 + 13^2 + 17^2 = 579 grid cells
N_PLACE_CELLS = int(S.FLYHASH_CONFIG['num_bits'])       # 256 place cells
_SCRATCH = None    # scratch grid attractor for consolidation, correction and row read-out (built on demand)


def bump_pattern_flat(target_xy, sigma=1.2):
    """The 579-dim grid pattern representing target_xy (the three module patterns concatenated)."""
    return np.concatenate([m.ravel() for m in bump_pattern(target_xy, sigma)]).astype(np.float32)


def inject_flat(pose, w_row, K=K_REC, n_settle=N_SETTLE):
    """Inject one 579-dim row (split into the three modules) into the grid attractor and settle."""
    mods, o = [], 0
    for c in CANN_SIZES:
        mods.append(np.asarray(w_row[o:o + c * c], np.float32).reshape(c, c)); o += c * c
    inject_pattern(pose, mods, K=K, n_settle=n_settle)


def _row_decode_settled(w_row):
    """Position encoded by a stored row: seed the scratch attractor from the row, inject the row, settle and
    decode. Recurrent competition resolves any multimodality in the row."""
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = make_system(); _SCRATCH.reset(1)
    o = 0
    for _i, _c in enumerate(CANN_SIZES):
        blk = np.asarray(w_row[o:o + _c * _c], np.float32).reshape(1, _c, _c); o += _c * _c
        _SCRATCH.pose._u_canns[_i] = jnp.asarray(blk)
        _SCRATCH.pose._r_canns[_i] = jnp.asarray(np.maximum(blk, 0.0))
    inject_flat(_SCRATCH.pose, w_row, K=K_REC, n_settle=N_SETTLE)
    return crt_decode(bump_flat(_SCRATCH.pose))


class L2WeightMemory:
    """Fixed-size place->grid memory. W is [n_cells x N_GRID] float32 (256 x 579 by default); its size is
    set by the neuron counts, not by the number of places stored.

    k_hot == 1 (one-cell code): each place owns one row. When no cell is free, the least-recently-used cell
    whose place is outside the Layer-1 window is reassigned, and its previous place loses its memory
    (valid() checks ownership before any read or write).
    k_hot > 1 (pattern-separated code): each place is assigned k random cells (`codes`); its sparsified grid
    pattern is added into those k rows. contrib[place] holds the place's current term while the place is
    writable, so an update replaces only that term; freeze(place) drops it and the place is read-only from
    then on. row() returns the coincidence of the place's k rows.
    Appearance descriptors H and creation keyframes kf are stored per place alongside W."""

    def __init__(self, n_cells=N_PLACE_CELLS, n_grid=N_GRID, k_hot=1, sparsify=0.0, code_seed=0):
        self.W = np.zeros((n_cells, n_grid), np.float32)   # the Layer-2 synaptic store
        self.k_hot = int(k_hot); self.sparsify = float(sparsify); self.codes = {}; self.contrib = {}
        self.code_rng = np.random.RandomState(1000003 + int(code_seed))
        # Synaptic-operation ledger, one operation per nonzero synapse per update: recall = injected current per
        # grid cell per settle step (+ the k-row read for k > 1); hebb/correct/consol/create = synapses written
        # (correct and consol also count the scratch injection).
        self.sops = dict(recall=0, hebb=0, consol=0, correct=0, create=0)
        self.owner = np.full(n_cells, -1, np.int64)        # one-cell code: place occupying each cell
        self.last_use = np.full(n_cells, -1, np.int64)     # one-cell code: LRU clock per cell
        self.H = []                                        # appearance descriptor per place
        self.kf = []                                       # creation keyframe per place
        self.n_cells = n_cells
        self.evictions = 0                                 # one-cell code: cell reassignments
        self.stale_instar = 0                              # writes refused (frozen place or reassigned cell)

    def allocate(self, place, t, protect=()):
        """Assign cells to a new place. k-hot: k random cells. One-cell: a free cell, else the least-recently-used
        cell whose place is not in `protect` (the places currently in the Layer-1 window)."""
        if self.k_hot > 1:
            self.codes[place] = np.sort(self.code_rng.choice(self.n_cells, self.k_hot, replace=False)); return place
        free = np.flatnonzero(self.owner < 0)
        if len(free):
            c = int(free[0])
        else:
            cand = [i for i in range(self.n_cells) if self.owner[i] not in protect]
            pool = cand if cand else list(range(self.n_cells))          # every cell is in-window (N < window)
            c = int(min(pool, key=lambda i: self.last_use[i])); self.evictions += 1
        self.owner[c] = place; self.last_use[c] = t
        return c

    def consolidate_via_substrate(self, c, place, target_xy, eta, t):
        """Eviction-time consolidation: seed the scratch attractor from this place's own pattern, inject the
        target position as a current, settle, and write the settled bump (eta=1 replaces the pattern)."""
        global _SCRATCH
        if _SCRATCH is None:
            _SCRATCH = make_system(); _SCRATCH.reset(1)
        o = 0
        _own = self.row(c) if self.k_hot == 1 else self.contrib.get(c, self.row(c))
        for _i, _c in enumerate(CANN_SIZES):
            blk = np.asarray(_own[o:o + _c * _c], np.float32).reshape(1, _c, _c); o += _c * _c
            _SCRATCH.pose._u_canns[_i] = jnp.asarray(blk)
            _SCRATCH.pose._r_canns[_i] = jnp.asarray(np.maximum(blk, 0.0))
        inject_flat(_SCRATCH.pose, bump_pattern_flat(target_xy), K=K_REC, n_settle=N_SETTLE)
        self.sops['consol'] += self.W.shape[1] * N_SETTLE
        self.hebb(c, place, bump_flat(_SCRATCH.pose), eta, t, bucket='consol')

    def hebb_correct_via_substrate(self, c, place, pose, target_xy, eta, t):
        """Layer-1 -> Layer-2 correction delivered through the substrate: seed the scratch attractor from the
        current bump, inject Layer 1's relaxed position as a current, settle, and write the settled bump, so the
        correction has the same representation as the observation-driven writes."""
        global _SCRATCH
        if self.k_hot > 1 and place not in self.contrib: self.stale_instar += 1; return   # frozen place
        if _SCRATCH is None:
            _SCRATCH = make_system(); _SCRATCH.reset(1)
        for _i, _c in enumerate(CANN_SIZES):
            pose_u = np.asarray(pose._u_canns[_i])
            _SCRATCH.pose._u_canns[_i] = jnp.asarray(pose_u)
            _SCRATCH.pose._r_canns[_i] = jnp.asarray(np.maximum(pose_u, 0.0))
        inject_flat(_SCRATCH.pose, bump_pattern_flat(target_xy), K=K_REC, n_settle=N_SETTLE)
        self.sops['correct'] += self.W.shape[1] * N_SETTLE
        self.hebb(c, place, bump_flat(_SCRATCH.pose), eta, t, bucket='correct')

    def learn_descriptor(self, place, h, desc_eta):
        """Instar update of a place's appearance descriptor at rate desc_eta (independent of eta and l2_learn)."""
        if h is not None and place < len(self.H) and self.H[place] is not None:
            self.H[place] = (1.0 - desc_eta) * self.H[place] + desc_eta * np.asarray(h, np.float32)

    def hebb(self, c, place, g_live, eta, t, bucket='hebb'):
        """Instar write toward the live grid population state g_live (max-normalised): the place's pattern moves a
        fraction eta toward the current grid activity. Frozen places and reassigned cells are not written."""
        if not self.valid(c, place):
            self.stale_instar += 1
            return
        g = np.asarray(g_live, np.float32).ravel()
        g = g / (g.max() + 1e-8)                       # store a shape, not a gain
        if self.k_hot > 1:
            old = self.contrib.get(place)
            if old is None: self.stale_instar += 1; return          # frozen (evicted) place
            self._write_contrib(place, (1.0 - eta) * old + eta * self._sparse(g)); self.sops[bucket] += self.k_hot * self.W.shape[1]
        else:
            self.W[c] = (1.0 - eta) * self.W[c] + eta * g
            self.sops[bucket] += self.W.shape[1]
        if self.k_hot == 1: self.last_use[c] = t

    def burn(self, c, place, xy, t, h=None, kf=None):
        """Creation write. `c` is the cell (one-cell code) or the place id (k-hot handle). Also stores the place's
        appearance descriptor and creation keyframe."""
        self.set_pattern(c, bump_pattern_flat(xy), t)
        while len(self.H) <= place: self.H.append(None); self.kf.append(-1)
        if h is not None: self.H[place] = np.asarray(h, np.float32)
        self.kf[place] = kf if kf is not None else t

    # ---------------- pattern-separated code (k_hot > 1) ----------------
    def _sparse(self, flat):
        """Keep grid cells >= sparsify x each module's peak (stored patterns must be sparse to superpose)."""
        if self.sparsify <= 0.0: return np.asarray(flat, np.float32)
        out = np.zeros(len(flat), np.float32); o = 0
        for cs in CANN_SIZES:
            b = np.asarray(flat[o:o + cs * cs], np.float32); m = float(b.max())
            out[o:o + cs * cs] = np.where(b >= self.sparsify * m, b, 0.0) if m > 0 else 0.0; o += cs * cs
        return out

    def _write_contrib(self, place, target):
        """Replace this place's term in its k rows: W[cs] += target - old. Other places sharing the rows are untouched."""
        old = self.contrib.get(place); cs = self.codes[place]
        self.W[cs] += (target - (old if old is not None else 0.0)).astype(np.float32); self.contrib[place] = np.asarray(target, np.float32)

    def set_pattern(self, c, pattern, t):
        pattern = np.asarray(pattern, np.float32)
        if self.k_hot > 1: self._write_contrib(c, self._sparse(pattern)); self.sops['create'] += self.k_hot * self.W.shape[1]
        else: self.W[c] = pattern; self.last_use[c] = t; self.sops['create'] += self.W.shape[1]

    def row(self, c):
        """The pattern recall injects: the row itself (one-cell code) or the k-input coincidence of the place's rows."""
        if self.k_hot == 1: return self.W[c]
        R = self.W[self.codes[c]]; out = np.zeros((len(R), R.shape[1]), np.float32); o = 0
        for cs in CANN_SIZES:                               # per-module normalise each row, then elementwise min
            blk = R[:, o:o + cs * cs]; m = blk.max(axis=1, keepdims=True); out[:, o:o + cs * cs] = np.where(m > 1e-9, blk / np.maximum(m, 1e-9), 0.0); o += cs * cs
        mn = out.min(axis=0); o = 0                          # re-normalise each module of the coincidence pattern to
        for cs in CANN_SIZES:                               # peak 1 so the injected current is at full amplitude
            blk = mn[o:o + cs * cs]; m = float(blk.max())
            if m > 1e-9: mn[o:o + cs * cs] = blk / m
            o += cs * cs
        return mn

    def freeze(self, place):
        """After eviction-time consolidation the place's term is final: drop it, making the place read-only."""
        if self.k_hot > 1: self.contrib.pop(place, None)

    def instar(self, c, place, xy, eta, t):
        """Instar write toward the rendered bump at position xy (used by the readout control, l2_learn='relax')."""
        if not self.valid(c, place):
            self.stale_instar += 1
            return
        if self.k_hot > 1:
            old = self.contrib.get(place)
            if old is None: self.stale_instar += 1; return
            self._write_contrib(place, (1.0 - eta) * old + eta * self._sparse(bump_pattern_flat(xy))); self.sops['correct'] += self.k_hot * self.W.shape[1]; return
        self.W[c] = (1.0 - eta) * self.W[c] + eta * bump_pattern_flat(xy); self.last_use[c] = t; self.sops['correct'] += self.W.shape[1]

    def valid(self, c, place):
        if self.k_hot > 1: return place in self.codes                  # k-hot cells are never reassigned
        return c >= 0 and self.owner[c] == place                        # one-cell: cell not reassigned

    def recall(self, pose, c, t, K=K_REC, n_settle=N_SETTLE):
        inject_flat(pose, self.row(c), K=K, n_settle=n_settle)
        if self.k_hot == 1: self.last_use[c] = t
        self.sops['recall'] += self.W.shape[1] * n_settle + (self.k_hot * self.W.shape[1] if self.k_hot > 1 else 0)

    def nbytes(self):
        return self.W.nbytes


def _module_phases(grid_flat):
    """Per-module (x, y) phase in metres from the population circular mean."""
    g = np.asarray(grid_flat, float).ravel()
    s1, s2 = CANN_SIZES[0] ** 2, CANN_SIZES[1] ** 2
    mods = [g[:s1].reshape(CANN_SIZES[0], CANN_SIZES[0]),
            g[s1:s1 + s2].reshape(CANN_SIZES[1], CANN_SIZES[1]),
            g[s1 + s2:].reshape(CANN_SIZES[2], CANN_SIZES[2])]
    px, py = [], []
    for m, size, scale in zip(mods, CANN_SIZES, WRAP_SCALES):
        ang = np.arange(size) * (2 * np.pi / size)
        pm = m / (m.sum() + 1e-8)
        ax, ay = pm.sum(axis=0), pm.sum(axis=1)          # same axis convention as the substrate's decoder
        cx = np.arctan2((ax * np.sin(ang)).sum(), (ax * np.cos(ang)).sum()) % (2 * np.pi)
        cy = np.arctan2((ay * np.sin(ang)).sum(), (ay * np.cos(ang)).sum()) % (2 * np.pi)
        px.append(cx / (2 * np.pi) * scale); py.append(cy / (2 * np.pi) * scale)
    return np.array(px), np.array(py)


_CRT_LO, _CRT_HI, _CRT_STEP = -0.5, 2.5, 0.002        # search span: the 2 x 2 m arena plus margin
_CRT_XS = np.arange(_CRT_LO, _CRT_HI, _CRT_STEP)
_CRT_SC = np.array(WRAP_SCALES)


def crt_decode(grid_flat):
    """Prior-free position read-out of the grid code by residue (CRT) consensus: the position in the search
    span most consistent with all three module phases (periods 0.6 / 0.8 / 1.06 m)."""
    return crt_decode_res(grid_flat)[0]


def crt_decode_res(grid_flat):
    """crt_decode plus its consistency residual per axis: RMS over modules of the wrapped distance between the
    winning position and each module's phase (zero when the three modules agree)."""
    px, py = _module_phases(grid_flat)

    def _best(ph):
        d = _CRT_XS[:, None] - ph[None, :]
        d = (d + _CRT_SC / 2.0) % _CRT_SC - _CRT_SC / 2.0     # signed wrap into +-scale/2
        cost = (d ** 2).sum(axis=1); i = int(np.argmin(cost))
        return float(_CRT_XS[i]), float(np.sqrt(cost[i] / d.shape[1]))
    (x, rx), (y, ry) = _best(px), _best(py)
    return np.array([x, y]), (rx, ry)


def _l2_verify(b, match_place, best, second):
    """Distance-free verification of a closure to an evicted place.

    Accepts only if (i) the best appearance match beats the runner-up by at least L2_MARGIN and (ii) the
    matched places of the last L2_SEQ_L keyframes are non-decreasing in place index and advance overall,
    i.e. the robot is retracing the stored route. Neither test uses metric distance, so the check remains
    valid at any accumulated drift."""
    if (best - second) < L2_MARGIN:
        DIAG['rej_margin'] += 1
        return False
    seq = [match_place[k] for k in range(b - L2_SEQ_L + 1, b + 1) if k in match_place]
    if len(seq) < L2_SEQ_L:
        DIAG['rej_short'] += 1
        return False
    if not all(seq[i + 1] >= seq[i] for i in range(len(seq) - 1)):   # several keyframes can fall in one place
        DIAG['rej_order'] += 1
        return False
    if seq[-1] <= seq[0]:                                              # must actually advance
        DIAG['rej_noprog'] += 1
        return False
    return True


def weight_l2_stream(seed, steps, thr=0.55, eta=0.3, cap=25, course='l2probe', kf_gt=0.04,
                     use_l2=True, poison='none', l2_learn='hebb_correct', med_lr=0.02,
                     consol_eta=1.0, consol_passes=3,
                     l2_code='khot', k_hot=8, n_cells=N_PLACE_CELLS, sparsify=0.5, desc_eta=0.3, recall_source='row'):
    """Run the two-layer system for `steps` simulator steps. Returns (step, est, gt, H, fired).

    Defaults are the deployed configuration. Options used by the paper's controls:
      use_l2=False          window alone: revisits to evicted places are not served (in-window behaviour,
                            learning and consolidation are unchanged).
      eta                   in-window learning rate of the synaptic pattern (eta=0: no in-window learning).
      consol_eta            eviction-time consolidation rate (0: no consolidating write; the pattern is the last
                            in-window write, then frozen); consol_passes substrate passes per eviction.
      l2_learn              'hebb_correct' (deployed: instar writes from the live grid bump at recognition, and
                            post-relaxation corrections delivered through the substrate) or 'relax' (readout
                            control: after each relaxation the pattern is overwritten by the rendered bump at
                            Layer 1's relaxed position; no observation-driven writes).
      recall_source         'row' (deployed: synaptic recall) or 'float' (symbolic-store control: an evicted
                            recall injects a bump rendered at the place's retained symbolic position).
      poison                'none' or 'evicted' (deletion test: a place's symbolic position is set to NaN at
                            eviction, so any later use of it would surface as NaN).
      l2_code, k_hot, n_cells, sparsify
                            'khot' = pattern-separated code (k_hot random cells per place, sparsified patterns,
                            coincidence read-out); 'onehot' = one-cell code. n_cells = memory size in place cells.
    Per-run counters and per-closure records are left in the module-level DIAG dict."""
    if l2_learn not in ('hebb_correct', 'relax'):
        raise ValueError(f"l2_learn must be 'hebb_correct' or 'relax', not {l2_learn!r}")
    if poison not in ('none', 'evicted'):
        raise ValueError(f"poison must be 'none' or 'evicted', not {poison!r}")
    for k in list(DIAG): DIAG[k] = 0
    DIAG['lost_kfs'] = []
    DIAG['snaps'] = []
    win = []          # Layer-1 window: place ids, most recently used last; |win| <= cap
    def _win_touch(pid):
        """Bring place `pid` into the window (or refresh it); if the window is full, evict the LRU place."""
        ev = None
        if pid in win:
            win.remove(pid)
        elif len(win) >= cap:
            ev = win.pop(0)
        win.append(pid)
        return ev                           # the evicted place id, or None
    def _consolidate(pid, t):
        """Consolidation at eviction: write Layer 1's final relaxed estimate of the place into its pattern,
        delivered through the substrate (consol_passes passes at rate consol_eta). Records the target's error
        and the pattern's decode error before and after, all against ground truth."""
        if pid is None or consol_eta <= 0.0:
            return
        c = cell_of[pid]
        if not L2MEM.valid(c, pid):
            return
        if L2MEM.k_hot > 1 and pid not in L2MEM.contrib:      # already frozen
            return
        tgt = np.asarray(pl_xy[pid], float)
        if not np.all(np.isfinite(tgt)):
            return
        g_true = np.asarray(gt[int(L2MEM.kf[pid])], float)
        e_tgt = float(np.hypot(*(tgt - g_true))) * 100
        e_bef = float(np.hypot(*(np.asarray(crt_decode(L2MEM.row(c)), float) - g_true))) * 100
        for _ in range(int(consol_passes)):
            L2MEM.consolidate_via_substrate(c, pid, tgt, consol_eta, t)
        e_aft = float(np.hypot(*(np.asarray(crt_decode(L2MEM.row(c)), float) - g_true))) * 100
        DIAG['consol_n'] = DIAG.get('consol_n', 0) + 1
        DIAG['consol_tgt_sum'] = DIAG.get('consol_tgt_sum', 0.0) + e_tgt
        DIAG['consol_bef_sum'] = DIAG.get('consol_bef_sum', 0.0) + e_bef
        DIAG['consol_aft_sum'] = DIAG.get('consol_aft_sum', 0.0) + e_aft
    def _on_evict(pid, t):
        """A place leaves the window: consolidate its pattern, freeze it, and (deletion test only) delete its
        symbolic position."""
        if pid is None:
            return
        _consolidate(pid, t)
        L2MEM.freeze(pid)
        if poison == 'evicted':
            pl_xy[pid] = np.array([np.nan, np.nan])
    s = make_system()
    env = S.LiveEnvironment(jax.random.PRNGKey(seed), chunk_size=steps + 100, course_type=course)
    ev, kin, tof, p0, th0, _ = env.step(); s.initialize_pose(jnp.array([p0]), jnp.array([th0]))
    est, gt, H, stamps = [], [], [], []
    pl_hash, pl_xy, pl_th, pl_kf = [], [], [], []
    pl_med = []; pl_obs = []   # per-place online median and observation list (symbolic reference estimators)
    L2MEM = L2WeightMemory(n_cells=int(n_cells), k_hot=(int(k_hot) if l2_code == 'khot' else 1),
                           sparsify=(float(sparsify) if l2_code == 'khot' else 0.0), code_seed=seed); cell_of = []   # place -> cell (one-cell) or place id (k-hot)
    chain = {}; loops = {}; match_of = {}; match_place = {}
    p_cur = None; at_bump = None; fired = []; last_gt = None
    for t in range(steps):
        ev, kin, tof, gp, gth, _ = env.step()
        pose, _, _, _, _, dbg = s.forward_step(jnp.array([ev]), jnp.array([kin]), jnp.array([tof]))
        cx, cy, cth = float(pose[0, 0]), float(pose[0, 1]), float(pose[0, 2])
        gx, gy = float(gp[0]), float(gp[1])
        if last_gt is not None and np.hypot(gx - last_gt[0], gy - last_gt[1]) <= kf_gt:
            continue
        last_gt = (gx, gy)
        b = len(est)                                  # keyframe index
        h = (np.asarray(dbg['Visual_Barcode'][0]).ravel() > 0.5).astype(np.float64)   # 256-bit appearance key
        est.append([cx, cy]); gt.append([gx, gy]); H.append(h); stamps.append(t)
        # ---- recognition: best and runner-up stored place by appearance, created > MIN_TOPO keyframes ago ----
        best, second, ak, aplace = thr, 0.0, -1, -1
        for p in range(len(pl_kf)):
            kf = int(L2MEM.kf[p])
            if (b - kf) > MIN_TOPO:
                c = cos(h, L2MEM.H[p])
                if c > best: second = best if best > thr else second; best, ak, aplace = c, kf, p
                elif c > second: second = c
        if ak >= 0:
            match_of[b] = ak; match_place[b] = aplace
            idxs = [k for k in range(b - SEQ_W, b + 1) if k >= 0 and k in match_of]
            ok = len(idxs) >= SEQ_W
            if not ok: DIAG['seq_short'] = DIAG.get('seq_short', 0) + 1
            if ok:
                for k in idxs[:-1]:
                    dd = (np.array(est[b]) - np.array(est[k])) - (np.array(est[match_of[b]]) - np.array(est[match_of[k]]))
                    if np.hypot(*dd) > SEQ_TOL:                       # relative-displacement sequence check
                        DIAG['seq_rej'] = DIAG.get('seq_rej', 0) + 1; DIAG['seq_rej_ev'] = DIAG.get('seq_rej_ev', 0) + int(aplace not in win)
                        ok = False; break
            if ok and p_cur is not None and aplace != p_cur:
                # Place `aplace` is recognised. The raw pose estimate is an observation of its position; the
                # symbolic reference estimators (online median, observation list) are updated in every arm and
                # are measurement only.
                _obs = np.array([cx, cy], float)
                pl_med[aplace] = pl_med[aplace] + med_lr * np.sign(_obs - pl_med[aplace])
                pl_obs[aplace].append(_obs)
                _gd = float(np.hypot(*(np.asarray(crt_decode(bump_flat(s.pose)), float) - _obs))) * 100   # grid decode vs pose estimate
                DIAG['nullgap_sum'] = DIAG.get('nullgap_sum', 0.0) + _gd
                DIAG['nullgap_n'] = DIAG.get('nullgap_n', 0) + 1
                if not isinstance(DIAG.get('nullgap_list'), list): DIAG['nullgap_list'] = []
                DIAG['nullgap_list'].append(float(_gd))
                # Patterns are written only while the place is in the window; evicted places are read-only.
                if aplace in win: L2MEM.learn_descriptor(aplace, h, desc_eta)
                if l2_learn == 'hebb_correct' and aplace in win:
                    L2MEM.hebb(cell_of[aplace], aplace, bump_flat(s.pose), eta, b)
            if ok and p_cur is not None and aplace != p_cur and aplace < p_cur:
                # ---- loop closure to an earlier place ----
                _edge = se2_between(at_bump, (cx, cy, cth))
                do_recall = True; in_window = (aplace in win)
                if aplace in win and np.hypot(cx - pl_xy[aplace][0], cy - pl_xy[aplace][1]) > GATE_R:
                    DIAG['l1_gate_rej'] += 1          # in-window closure rejected by the metric gate
                    do_recall = False
                elif aplace not in win and not L2MEM.valid(cell_of[aplace], aplace):
                    DIAG['l2_cell_lost'] += 1          # one-cell code: the place's cell was reassigned
                    do_recall = False
                elif aplace not in win and not _l2_verify(b, match_place, best, second):
                    DIAG['l2_verify_rej'] += 1        # evicted place: distance-free verifier rejected
                    do_recall = False
                elif in_window:
                    # ---- Layer 1: add the loop edge, relax the window, correct the window's patterns ----
                    _on_evict(_win_touch(aplace), b)         # in-window use refreshes LRU recency (may evict)
                    loops[(p_cur, aplace)] = _edge
                    updated = relax_window(pl_xy, pl_th, chain, loops, win)
                    for i in updated:
                        _t = np.asarray(pl_xy[i], float)
                        if l2_learn == 'hebb_correct':
                            L2MEM.hebb_correct_via_substrate(cell_of[i], i, s.pose, _t, eta, b)
                        else:
                            L2MEM.instar(cell_of[i], i, _t, eta, b)
                    DIAG['relaxes'] += 1
                elif not use_l2:
                    # Window alone: the place has been evicted and there is no Layer 2, so the revisit is lost.
                    DIAG['lost'] += 1; DIAG['lost_kfs'].append(b)
                    do_recall = False
                else:
                    DIAG['l2_only'] += 1            # served by Layer-2 recall
                if do_recall and in_window and not L2MEM.valid(cell_of[aplace], aplace):
                    DIAG['l2_cell_lost_inwin'] = DIAG.get('l2_cell_lost_inwin', 0) + 1   # one-cell code: in-window place
                    do_recall = False                                                   # without a row; the window was
                    p_cur = aplace; at_bump = (cx, cy, cth)                              # relaxed but the pose is not reset
                if do_recall:
                    loops[(p_cur, aplace)] = _edge
                    DIAG['loops'] += 1
                    cx_pre, cy_pre = cx, cy
                    # ---- recall: inject the place's pattern into the live grid attractor, settle, decode ----
                    if recall_source == 'float' and not in_window:           # symbolic-store control
                        inject_flat(s.pose, bump_pattern_flat(np.asarray(pl_xy[aplace], float)), K=K_REC, n_settle=N_SETTLE); DIAG['float_recalls'] = DIAG.get('float_recalls', 0) + 1
                    else:
                        L2MEM.recall(s.pose, cell_of[aplace], b)
                    corr = crt_decode(bump_flat(s.pose))
                    # per-closure record: (keyframe, place, target error, row raw-decode error, row settled-decode
                    # error, landing vs row decode, in_window); errors in cm against the place's ground truth
                    _gp = np.asarray(gt[int(L2MEM.kf[aplace])], float)
                    _row = L2MEM.row(cell_of[aplace])
                    _rd_raw = np.asarray(crt_decode(_row), float)
                    _rd_set = np.asarray(_row_decode_settled(_row), float)
                    _tg = np.asarray(pl_xy[aplace], float)
                    if not isinstance(DIAG.get('recall_rows'), list): DIAG['recall_rows'] = []
                    DIAG['recall_rows'].append((int(b), int(aplace),
                        float(np.hypot(*(_tg - _gp))) * 100 if np.all(np.isfinite(_tg)) else float('nan'),
                        float(np.hypot(*(_rd_raw - _gp))) * 100,
                        float(np.hypot(*(_rd_set - _gp))) * 100,
                        float(np.hypot(*(np.asarray(corr, float) - _rd_set))) * 100,
                        bool(in_window)))
                    # recognition check: true distance between the vehicle now and the recalled place (cm)
                    _al = float(np.hypot(*(np.asarray(gt[b], float)
                                           - np.asarray(gt[int(L2MEM.kf[aplace])], float)))) * 100
                    if not isinstance(DIAG.get('alias_list'), list): DIAG['alias_list'] = []
                    DIAG['alias_list'].append((int(b), float(_al), bool(in_window)))
                    s.last_decoded_xy = jnp.array([corr])
                    # The correction is not motion: the substrate's online velocity-gain calibration measures the
                    # bump's velocity from successive positions, so it also continues from the corrected position.
                    s.pose.prev_pose_xy = jnp.array([corr])
                    est[b] = [float(corr[0]), float(corr[1])]; fired.append((p_cur, aplace, b))
                    DIAG['snaps'].append((b, float(corr[0] - cx_pre), float(corr[1] - cy_pre),
                                          bool(in_window)))
                    cx, cy = float(corr[0]), float(corr[1])
                    # The current place advances only on in-window closures: an evicted place is not in the
                    # window graph, so the next chain edge must still start from a window node.
                    if in_window:
                        p_cur = aplace
                    at_bump = (cx, cy, cth)
            elif ok and p_cur is not None and aplace != p_cur:
                p_cur = aplace; at_bump = (cx, cy, cth)                   # forward re-traversal
        # ---- Layer-1 bound: drop graph entries outside the window and match records no longer read ----
        if len(pl_hash) > cap:
            _wset = set(win)
            for _k in [k for k in chain if k not in _wset or chain[k][0] not in _wset]:
                del chain[_k]
            for _k in [k for k in loops if k[0] not in _wset or k[1] not in _wset]:
                del loops[_k]
        _keep = b - max(L2_SEQ_L, SEQ_W) - 2
        if _keep > 0:
            for _k in [k for k in match_place if k < _keep]:
                del match_place[_k]
            for _k in [k for k in match_of if k < _keep]:
                del match_of[_k]
        # ---- place creation (appearance novelty) and creation write ----
        if all(cos(h, ph) < thr for ph in pl_hash):
            newp = len(pl_hash)
            pl_hash.append(h); pl_xy.append(np.array([cx, cy])); pl_th.append(cth); pl_kf.append(b)
            _c = L2MEM.allocate(newp, b, protect=set(win)); cell_of.append(_c)
            assert L2MEM.k_hot > 1 or _c == newp or L2MEM.evictions > 0, "place index != cell index without reassignment"
            L2MEM.burn(_c, newp, [cx, cy], b, h=h, kf=b)
            if l2_learn == 'hebb_correct':                  # the stored pattern is the live grid activity
                _g = np.asarray(bump_flat(s.pose)).ravel()
                L2MEM.set_pattern(_c, (_g / (_g.max() + 1e-8)).astype(np.float32), b)
            pl_med.append(np.array([cx, cy], float)); pl_obs.append([])
            DIAG['nodes'] += 1
            if p_cur is not None:
                chain[p_cur] = (newp,) + se2_between(at_bump, (cx, cy, cth))
            p_cur = newp; at_bump = (cx, cy, cth)
            _on_evict(_win_touch(newp), b)          # a new place enters the window (may evict the LRU place)
    DIAG['l2_evictions'] = L2MEM.evictions; DIAG['l2_bytes'] = int(L2MEM.nbytes())
    DIAG['seq_tol'] = float(SEQ_TOL); DIAG['recall_source'] = recall_source; DIAG['desc_eta'] = float(desc_eta); DIAG['stale_instar'] = int(L2MEM.stale_instar)
    for _k, _v in L2MEM.sops.items(): DIAG[f'sop_{_k}'] = int(_v)           # synaptic-operation ledger
    DIAG['sop_total'] = int(sum(L2MEM.sops.values()))
    if pl_xy:
        # end-of-run store check: each place's settled pattern decode vs its symbolic and reference estimates
        DIAG['gc_plxy'] = np.array(pl_xy)
        DIAG['gc_plema'] = np.array([np.mean(np.array(o), axis=0) if o else pl_xy[k] for k, o in enumerate(pl_obs)])   # mean of observations
        DIAG['gc_plmed'] = np.array(pl_med)
        DIAG['gc_plbatch'] = np.array([np.median(np.array(o), axis=0) if o else pl_xy[k]
                                       for k, o in enumerate(pl_obs)])
        DIAG['gc_wset'] = np.array([_row_decode_settled(L2MEM.row(cell_of[k])) for k in range(len(pl_xy))])
        DIAG['gc_plgt'] = np.array([gt[int(L2MEM.kf[k])] for k in range(len(pl_xy))])
        DIAG['gc_evicted'] = np.array([k not in win for k in range(len(pl_xy))])   # places outside the window at run end
    return np.array(stamps), np.array(est), np.array(gt), np.array(H), fired
