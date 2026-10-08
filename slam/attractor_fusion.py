"""Interface to the bundled spiking substrate (``src/``).

Builds the path-integrating grid attractor used by the two-layer estimator and holds the appearance
recognition constants shared by both layers.
"""
import os, sys
os.environ['MPLBACKEND'] = 'Agg'; os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import jax, jax.numpy as jnp
import src.snn_slam_system as S

MIN_TOPO = 10      # a candidate place must have been created more than this many keyframes ago
SEQ_W = 3          # keyframes in the relative-displacement sequence check
SEQ_TOL = 0.20     # relative-displacement sequence check tolerance (m)


def bump_flat(pose):
    """The grid attractor's firing rates, the three modules concatenated (1 x 579)."""
    return jnp.concatenate([g.reshape(1, -1) for g in pose._r_canns], axis=1)


def make_system(seed_sys=43):
    """Substrate with pure grid-attractor path integration: no pin to the integrated IMU position
    (K_IMU_POS = 0), and the cerebellar translational velocity gains initialised at the calibrated 0.0120."""
    s = S.SNNSLAMSystem(jax.random.PRNGKey(seed_sys), n_depth=S.N_DEPTH); s.reset(1)
    s.pose.K_IMU_POS = 0.0
    s.pose.VEL_GAIN_XY = 0.0120
    g = jnp.ones((1, s.pose.n_speed_neurons)) * 0.0120
    s.pose.W_cereb_xy_imu = g; s.pose.W_cereb_xy_vis = g
    return s


def cos(a, b):
    """Overlap of two binary appearance keys, normalised by the geometric mean of their active counts."""
    return float(a @ b) / (np.sqrt(a.sum() * b.sum()) + 1e-9)
