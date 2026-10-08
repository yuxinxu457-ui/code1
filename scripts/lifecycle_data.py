#!/usr/bin/env python3
"""Record the lifecycle of every place on one deployed run (for Supplementary Fig. S2).

Runs the deployed configuration for one seed exactly as the campaign driver does, and records, without changing the
model, when each place is created (Layer 2 allocates its code) and when it is evicted (its pattern is consolidated).
Recall events come from the run's own diagnostics. The run is checked against the campaign result: the trajectory
hash and the headline metrics must match the stored row. Output: results/lifecycle_s<seed>.npz
Usage: SLAM_OBS_SLICE=800 python scripts/lifecycle_data.py [seed]   (default: 930, the median-gain seed used in Fig. 3)
"""
import os, sys, json, hashlib
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slam import weight_l2_slam as WL
from slam.weight_l2_slam import weight_l2_stream, DIAG
import paper2_driver as PD

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 930
created, evicted = {}, {}
_alloc, _consol = WL.L2WeightMemory.allocate, WL.L2WeightMemory.consolidate_via_substrate


def allocate(self, place, t, protect=()):
    created.setdefault(int(place), int(t)); return _alloc(self, place, t, protect)


def consolidate(self, c, place, target_xy, eta, t):
    evicted.setdefault(int(place), int(t)); return _consol(self, c, place, target_xy, eta, t)


WL.L2WeightMemory.allocate = allocate; WL.L2WeightMemory.consolidate_via_substrate = consolidate
cfg = dict(PD.BASE)
st, est, gt, H, fired = weight_l2_stream(seed, 3500, **cfg)
est = np.asarray(est, float); gt = np.asarray(gt, float)
sha = hashlib.sha1(est.round(6).tobytes()).hexdigest()[:12]
ref = [json.loads(l) for l in open(os.path.join(ROOT, 'results', 'paper2_results.jsonl'))
       if '"core:deployed:cap25:%d"' % seed in l][0]
m = PD.metrics(est, gt, cfg['cap'], 3500)
print(f"trajectory hash {sha} vs campaign {ref['est_sha']}; raw {m['raw']:.3f} vs {ref['raw']:.3f}; served {m['n_served']} vs {ref['n_served']}")
assert sha == ref['est_sha'] and abs(m['raw'] - ref['raw']) < 1e-9 and m['n_served'] == ref['n_served'], 'run does not reproduce the campaign'
rr = np.asarray(DIAG.get('recall_rows') or [], float)          # (keyframe, place, ..., in_window flag)
P = sorted(created)
np.savez_compressed(os.path.join(ROOT, 'results', f'lifecycle_s{seed}.npz'), seed=seed, est=est, gt=gt,
                    place=np.array(P), created=np.array([created[p] for p in P]),
                    evicted=np.array([evicted.get(p, -1) for p in P]),
                    recall_kf=rr[:, 0].astype(int), recall_place=rr[:, 1].astype(int), recall_inwin=rr[:, 6].astype(int))
print(f"places {len(P)}, evicted {sum(1 for p in P if p in evicted)}, recalls {len(rr)} ({int((rr[:,6]==0).sum())} of evicted places)")
