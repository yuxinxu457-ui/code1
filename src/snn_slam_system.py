#!/usr/bin/env python3
"""Spiking SLAM substrate: sensor simulation and the neuromorphic front end.

  * LiveEnvironment: streams events, IMU (gyro, accelerometer), visual-odometry velocity and ToF readings
    along a course of the simulated arena (src/sparse_forest.py), with MEMS gyro bias and random walk,
    wingbeat vibration and a visual-odometry velocity-error model.
  * SNNSLAMSystem: event-camera vision (frozen CSNN + STDP stream), the pose attractors (three grid-cell
    modules of 11^2 + 13^2 + 17^2 = 579 cells and a heading ring, PoseCANN) with a gravity-referenced
    complementary attitude filter, and the place-cell network whose sparse appearance key
    ('Visual_Barcode', 256 bits) is used for place recognition.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import sys


# ============================================================================
# Imports from the package (the repository root must be importable)
# ============================================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sparse_forest import (
    generate_fixed_room_dataset,
    N_PIXELS, DT, FOV_DEG,
    VX_RANGE, VY_RANGE, OMEGA_RANGE,
)
from src.snn_vision_fusion import DualStreamVisionCortex
from src.snn_pose_cann import (
    PoseCANN,
    build_2d_cann_weights,
    build_1d_ring_weights,
    build_asymmetric_ring_weights,
    build_asymmetric_cann_weights_x,
    build_asymmetric_cann_weights_y,
    CANN_SIZES, WRAP_SCALES,
)
from src.snn_place_cells import PlaceCellNetwork

# ============================================================================
#  Sparse appearance key (place-cell code) configuration
# ============================================================================
FLYHASH_CONFIG = {
    "num_bits": 256,         # key length (number of place cells)
    "active_spikes_k": 8,    # active bits per key (~3% sparsity)
    "match_threshold": 5,    # minimum bit overlap for a key match
}

# ============================================================================
#  Parameters
# ============================================================================


N_DEPTH_PER_RAY = 64
N_DEPTH         = N_DEPTH_PER_RAY * 3  # 192 depth units (3 ToF rays)

TOF_MIN         = 0.1    # meters
TOF_MAX         = 2.83   # meters, diagonal of the 2 x 2 m room
TOF_SIGMA       = 0.25   # tof precision

# Gyro error model, injected into the gyro stream by LiveEnvironment.generate_new_chunk: a constant
# per-trial bias plus a slow rate random walk (on top of the wingbeat vibration below). DRIFT_OMEGA is an
# additional constant rate offset applied only when forward_step(..., inject_drift=True); it is 0.
DRIFT_OMEGA     = 0.0
GYRO_BIAS_STD   = 0.025   # rad/s : std of the per-trial constant gyro bias (cheap MEMS)
GYRO_RW_SIGMA   = 0.0015  # rad/s : per-step gyro rate random-walk increment std

# Wingbeat vibration on the IMU (gyro and accelerometer). The IMU samples at ~1 kHz (IMU_OVERSAMPLE x the
# 50 Hz estimator rate), where the 115 Hz wingbeat is represented without aliasing; the estimator consumes
# the mean rate / specific force over each step, which attenuates a zero-mean tone by |sinc(f*DT)|
# (0.112 at 115 Hz).
IMU_OVERSAMPLE  = 20      # IMU samples per estimator step (20 / 0.02 s = 1 kHz MEMS IMU)
VIB_ENABLE      = True
VIB_FREQ        = 115.0   # Hz wingbeat frequency (properly sampled by the 1 kHz IMU)
VIB_GYRO_AMP    = 0.20    # rad/s gyro vibration amplitude (at the IMU, before rate-averaging)
VIB_ACC_AMP     = 0.50    # m/s^2 accelerometer vibration amplitude (at the IMU)


def imu_rate_average(freq, amp, dt, steps, phase0=0.0, t0=0.0, oversample=None):
    """The estimator-rate signal an IMU delivers for a vibration tone.

    The IMU samples at `oversample`/dt (default IMU_OVERSAMPLE/DT = 1 kHz), where `freq` is
    properly represented; the estimator consumes the MEAN over each step (what integrating the
    rate over the interval gives). Returns a length-`steps` array at the estimator rate.

    """
    OS = int(oversample or IMU_OVERSAMPLE)
    t = t0 + np.arange(steps * OS, dtype=np.float64) * (dt / OS)
    return (amp * np.sin(2.0 * np.pi * freq * t + phase0)).reshape(steps, OS).mean(axis=1)

# Visual-odometry (event + ToF) translational-velocity error model. The body-frame velocity that drives
# position path integration is a modelled VO estimate, not ground truth: a per-trial metric-scale bias, a
# per-trial constant bias with a slow random walk, and per-step magnitude and direction noise, all applied
# to [vx, vy] in LiveEnvironment.generate_new_chunk. The accelerometer (attitude reference) is not affected.
# Levels are env-overridable (SLAM_VO_*).
VO_NOISE_ENABLE = os.environ.get('SLAM_VO_NOISE', '1') == '1'                 # on by default
VO_SCALE_STD    = float(os.environ.get('SLAM_VO_SCALE_STD',   '0.04'))        # per-trial metric-scale bias (frac)
VO_MAG_REL      = float(os.environ.get('SLAM_VO_MAG_REL',     '0.08'))        # per-step relative magnitude noise
VO_DIR_STD      = float(os.environ.get('SLAM_VO_DIR_STD',     '0.05'))        # per-step direction noise (rad, ~3 deg)
VO_BIAS_REL     = float(os.environ.get('SLAM_VO_BIAS_REL',    '0.03'))        # per-trial constant bias (frac of RMS speed)
VO_BIAS_RW_REL  = float(os.environ.get('SLAM_VO_BIAS_RW_REL', '0.0015'))      # slow bias random-walk (frac of RMS speed / step)

# Complementary-filter accelerometer-correction gain: theta_new = theta_gyro + ALPHA_FUSE*(theta_accel - theta_gyro).
# Env-overridable (SLAM_ALPHA_FUSE).
ALPHA_FUSE      = float(os.environ.get('SLAM_ALPHA_FUSE', '0.02'))

# ============================================================================
#  ToF population coder (Gaussian RBF)
# ============================================================================

class ToFPopulationCoder:
    """Convert 3-ray ToF depth to a Gaussian population code."""

    def __init__(self, n_depth_per_ray=N_DEPTH_PER_RAY, tof_min=TOF_MIN, tof_max=TOF_MAX, sigma=TOF_SIGMA):
        self.n_depth_per_ray = n_depth_per_ray
        self.sigma = sigma
        self.centers = jnp.linspace(tof_min, tof_max, n_depth_per_ray)

    def __call__(self, tof_array):
        B = tof_array.shape[0]
        diff = tof_array[:, :, None] - jnp.array(self.centers)[None, None, :]
        activations = jnp.exp(-(diff ** 2) / (2 * self.sigma ** 2))
        activations = activations / (activations.max(axis=2, keepdims=True) + 1e-8)
        return activations.reshape(B, -1)

# ============================================================================
#  Grid-code read-out
# ============================================================================

@jax.jit
def decode_grid_to_xy(grid_key_flat, prior_xy):
    """
    Decode the 579-dim grid activity to (x, y) by local phase unwrapping: each module's phase is resolved
    to the branch closest to prior_xy (the previous position) and the three modules are averaged.
    """
    B = grid_key_flat.shape[0]
    
    # 1. Slice the 579-dim vector back into 121, 169, 289 arrays
    s1, s2 = CANN_SIZES[0]**2, CANN_SIZES[1]**2
    r1 = grid_key_flat[:, :s1].reshape(B, CANN_SIZES[0], CANN_SIZES[0])
    r2 = grid_key_flat[:, s1:s1+s2].reshape(B, CANN_SIZES[1], CANN_SIZES[1])
    r3 = grid_key_flat[:, s1+s2:].reshape(B, CANN_SIZES[2], CANN_SIZES[2])
    
    modules = [r1, r2, r3]
    phases_x, phases_y = [], []
    
    # 2. Extract local phase (in meters) for each module
    for i, (size, scale) in enumerate(zip(CANN_SIZES, WRAP_SCALES)):
        angles = jnp.arange(size, dtype=jnp.float32) * (2 * jnp.pi / size)
        sin_a, cos_a = jnp.sin(angles), jnp.cos(angles)
        
        p = modules[i] / (modules[i].sum(axis=(1, 2), keepdims=True) + 1e-8)
        cx_angle = jnp.arctan2((p.sum(axis=1) * sin_a).sum(axis=1), (p.sum(axis=1) * cos_a).sum(axis=1)) % (2 * jnp.pi)
        cy_angle = jnp.arctan2((p.sum(axis=2) * sin_a).sum(axis=1), (p.sum(axis=2) * cos_a).sum(axis=1)) % (2 * jnp.pi)
        
        phases_x.append((cx_angle / (2 * jnp.pi)) * scale)
        phases_y.append((cy_angle / (2 * jnp.pi)) * scale)
        
    phases_x = jnp.stack(phases_x, axis=1) # [B, 3]
    phases_y = jnp.stack(phases_y, axis=1)
    
    scale_arr = jnp.array(WRAP_SCALES)[None, :] # [1, 3]
    
    # 3. Local phase unwrapping around the prior
    def unwrap_closest(phases, prior):
        prior_exp = prior[:, None] # [B, 1]
        # Find the shortest distance from the current phase to the expected prior phase
        delta = phases - (prior_exp % scale_arr)
        # Wrap delta cleanly between -scale/2 and +scale/2
        delta_wrapped = (delta + scale_arr / 2.0) % scale_arr - scale_arr / 2.0
        
        # Apply the shortest-path delta to the prior and average the three modules
        unwrapped_candidates = prior_exp + delta_wrapped 
        return jnp.mean(unwrapped_candidates, axis=1) 

    global_x = unwrap_closest(phases_x, prior_xy[:, 0])
    global_y = unwrap_closest(phases_y, prior_xy[:, 1])
    
    return jnp.stack([global_x, global_y], axis=1)

def wrap_angle(theta):
    """Wrap angles to [-pi, pi)."""
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

# ============================================================================
#  Event-camera rotation estimate (phase correlation)
# ============================================================================
@jax.jit
def get_phase_correlation(live_csnn, mem_csnn):
    """
    Scale/contrast-invariant 1-D phase correlation, zero-padded to 2N so that shifts do not wrap around.
    """
    N = live_csnn.shape[-1]
    
    # Zero-pad to 2N (linear rather than circular correlation)
    pad_width = [(0, 0)] * (live_csnn.ndim - 1) + [(0, N)]
    
    live_padded = jnp.pad(live_csnn, pad_width)
    mem_padded = jnp.pad(mem_csnn, pad_width)
    
    # Transform to Frequency Domain
    F_live = jnp.fft.fft(live_padded)
    F_mem = jnp.fft.fft(mem_padded)
    
    # Calculate the Cross-Power Spectrum (Mem * conj(Live))
    cross_power = F_mem * jnp.conj(F_live)
    
    # Normalize by magnitude to extract pure Phase
    cross_power_norm = cross_power / (jnp.abs(cross_power) + 1e-8)
    
    # Inverse FFT back to Spatial Domain
    r = jnp.fft.ifft(cross_power_norm)
    return jnp.abs(r)


@jax.jit
def get_dvs_rotation_shift(curr_ts, prev_ts):
    """
    Computes visual rotation shift and Peak-to-Sidelobe Ratio (PSR) confidence
    between consecutive time surfaces using 1D phase correlation.
    """
    curr_on = curr_ts[:, :N_PIXELS]
    curr_off = curr_ts[:, N_PIXELS:]
    prev_on = prev_ts[:, :N_PIXELS]
    prev_off = prev_ts[:, N_PIXELS:]
    
    r_on = get_phase_correlation(curr_on, prev_on)
    r_off = get_phase_correlation(curr_off, prev_off)
    r_real = r_on + r_off
    
    N_PAD = N_PIXELS * 2
    search_radius = 15
    
    idx = jnp.arange(N_PAD)
    mask = (idx <= search_radius) | (idx >= N_PAD - search_radius)
    r_masked = jnp.where(mask[None, :], r_real, -1e9)
    
    peak_idx = jnp.argmax(r_masked, axis=1)
    
    y2 = jnp.take_along_axis(r_real, peak_idx[:, None], axis=1)[:, 0]
    y1 = jnp.take_along_axis(r_real, ((peak_idx - 1) % N_PAD)[:, None], axis=1)[:, 0]
    y3 = jnp.take_along_axis(r_real, ((peak_idx + 1) % N_PAD)[:, None], axis=1)[:, 0]
    
    denom = 2.0 * (y1 - 2.0 * y2 + y3)
    sub_pixel_offset = jnp.clip((y1 - y3) / (denom - 1e-8), -1.0, 1.0)
    
    shift_int = jnp.where(peak_idx <= search_radius, peak_idx, peak_idx - N_PAD)
    pixel_shift = shift_int + sub_pixel_offset
    
    pixel_ang_res = jnp.radians(FOV_DEG) / N_PIXELS
    sub_pixel_th = -pixel_shift * pixel_ang_res
    
    # Calculate PSR (Peak-to-Sidelobe Ratio) as confidence
    mean_val = jnp.mean(r_real, axis=1)
    psr = y2 / (mean_val + 1e-8)
    
    return sub_pixel_th, psr


# ============================================================================
#  Trajectory alignment (Umeyama / Kabsch, rotation + translation)
# ============================================================================

def get_optimal_alignment_2d(P_est, P_gt):
    """
    Computes the optimal rotation (R) and translation (t) to align P_est to P_gt.
    SVD solution (Umeyama), used for the ATE.
    """
    if len(P_est) < 5:
        return np.eye(2), np.zeros(2)

    # 1. Find centroids
    mu_est = np.mean(P_est, axis=0)
    mu_gt = np.mean(P_gt, axis=0)

    # 2. Center the points
    P_est_c = P_est - mu_est
    P_gt_c = P_gt - mu_gt

    # 3. Calculate Covariance Matrix & SVD
    H = P_est_c.T @ P_gt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 4. Handle reflection (ensure it's a pure rotation)
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    # 5. Calculate final translation
    t = mu_gt - R @ mu_est
    return R, t


# ============================================================================
#  Live environment (sensor streams along a course)
# ============================================================================
class LiveEnvironment:
    def __init__(self, key, chunk_size=2000, course_type='l2probe'):
        self.key = key
        self.chunk_size = chunk_size
        self.course_type = course_type   # 'l2probe' or 'explore' (see src/sparse_forest.py)
        self.obstacles = None
        self.generate_new_chunk()

    def generate_new_chunk(self):
        self.key, subkey = random.split(self.key)
        events, labels, tof_dists, positions, headings, obs, segments, intensities = \
            generate_fixed_room_dataset(subkey, 1, time_steps=self.chunk_size,
                                        obstacles=self.obstacles, course_type=self.course_type)
        
        self.obstacles = np.array(obs)
        self.ev = np.array(events[0])
        kin3 = np.array(jnp.stack([labels[0, :, 0] * abs(VX_RANGE[1]),
                                   labels[0, :, 1] * abs(VY_RANGE[1]),
                                   labels[0, :, 2] * abs(OMEGA_RANGE[1])], axis=1))
        self.tof = np.array(tof_dists[0])
        self.pos = np.array(positions[0, :, :2])
        self.th = np.array(headings[0])
        self.intensities = np.array(intensities[0])

        # --- Proper-acceleration accelerometer (vertical-plane / pitch model) ---
        # The body attitude theta is a pitch in a sagittal (forward x, vertical z) plane, so
        # gravity provides an absolute reference. The body-frame proper acceleration is
        #   a_proper = R(theta)^T (a_world + [0, g]),   g = 9.81 along the vertical (z) axis,
        # giving arctan2(ax, az) = theta at low linear acceleration -- the signal the spiking
        # complementary filter fuses with the (drift-prone) gyro to anchor the ring attractor.
        G = 9.81
        pw = self.pos.astype(np.float32)
        N = pw.shape[0]
        aw = np.zeros((N, 2), dtype=np.float32)
        if N > 2:
            aw[1:-1] = (pw[2:] - 2.0 * pw[1:-1] + pw[:-2]) / (DT ** 2)   # world linear accel
        th = self.th.astype(np.float32)
        c, s = np.cos(th), np.sin(th)
        gx = aw[:, 0]
        gz = aw[:, 1] + G                                                # gravity on the vertical axis
        ax = c * gx + s * gz
        az = -s * gx + c * gz
        seed_acc = int(random.randint(subkey, (), 0, 2**31 - 1))
        rng_acc = np.random.RandomState(seed_acc)
        ACC_MEMS_NOISE = 0.05                                            # m/s^2 white MEMS noise (vibration added below)
        ax = ax + rng_acc.normal(0.0, ACC_MEMS_NOISE, N).astype(np.float32)
        az = az + rng_acc.normal(0.0, ACC_MEMS_NOISE, N).astype(np.float32)

        # --- Wingbeat vibration on the IMU (gyro + accelerometer), rate-averaged per step ---
        # (see imu_rate_average)
        if VIB_ENABLE:
            vib_g = imu_rate_average(VIB_FREQ, VIB_GYRO_AMP, DT, N)                 # gyro (rad/s)
            vib_ax = imu_rate_average(VIB_FREQ, VIB_ACC_AMP, DT, N)                 # accel x
            vib_az = imu_rate_average(VIB_FREQ, VIB_ACC_AMP, DT, N, phase0=np.pi / 2)  # accel z (cos)
            kin3[:, 2] = kin3[:, 2] + vib_g.astype(np.float32)
            ax = ax + vib_ax.astype(np.float32)
            az = az + vib_az.astype(np.float32)

        # --- MEMS gyro drift: constant per-trial bias + rate random walk ---
        # Applied to the gyro only (the accelerometer remains the absolute attitude reference).
        # Seeded per chunk from the JAX key stream.
        self.key, subkey_gyro = random.split(self.key)
        seed_gyro = int(random.randint(subkey_gyro, (), 0, 2**31 - 1))
        rng_gyro = np.random.RandomState(seed_gyro)
        gyro_bias = float(rng_gyro.normal(0.0, GYRO_BIAS_STD))
        gyro_rw = np.cumsum(rng_gyro.normal(0.0, GYRO_RW_SIGMA, N)).astype(np.float32)
        kin3[:, 2] = kin3[:, 2] + gyro_bias + gyro_rw

        # --- Visual-odometry velocity-error model on [vx, vy] ---
        # The accelerometer is built from true position and attitude, so it is unaffected; only the
        # translational velocity that drives position path integration is perturbed. Seeded per chunk
        # from the JAX key stream (after the gyro split).
        if VO_NOISE_ENABLE:
            self.key, subkey_vo = random.split(self.key)
            seed_vo = int(random.randint(subkey_vo, (), 0, 2**31 - 1))
            rng_vo = np.random.RandomState(seed_vo)
            v_body = kin3[:, :2].astype(np.float64)
            speed_rms = float(np.sqrt(np.mean(v_body[:, 0] ** 2 + v_body[:, 1] ** 2))) + 1e-9
            scale = 1.0 + rng_vo.normal(0.0, VO_SCALE_STD)                       # per-trial metric-scale bias
            mag = 1.0 + rng_vo.normal(0.0, VO_MAG_REL, N)                        # per-step relative magnitude noise
            dth = rng_vo.normal(0.0, VO_DIR_STD, N)                             # per-step direction noise (rad)
            bias0 = rng_vo.normal(0.0, VO_BIAS_REL * speed_rms, 2)             # per-trial constant velocity bias
            bias_rw = np.cumsum(rng_vo.normal(0.0, VO_BIAS_RW_REL * speed_rms, (N, 2)), axis=0)  # slow bias drift
            cd, sd = np.cos(dth), np.sin(dth)
            vx_r = cd * v_body[:, 0] - sd * v_body[:, 1]                        # rotate by direction noise
            vy_r = sd * v_body[:, 0] + cd * v_body[:, 1]
            kin3[:, 0] = (scale * mag * vx_r + bias0[0] + bias_rw[:, 0]).astype(np.float32)
            kin3[:, 1] = (scale * mag * vy_r + bias0[1] + bias_rw[:, 1]).astype(np.float32)

        acc = np.stack([ax, az], axis=1).astype(np.float32)
        # kin now carries [vx, vy, omega, ax, az]; forward_step reads acc from columns 3:5.
        self.kin = np.concatenate([kin3, acc], axis=1).astype(np.float32)
        self.t = 0

    def step(self):
        if self.t >= self.chunk_size:
            print("\n Robot reached end of planned trajectory. Generating next path chunk...")
            self.generate_new_chunk()
            
        frame = (self.ev[self.t], self.kin[self.t], self.tof[self.t], 
                 self.pos[self.t], self.th[self.t], self.intensities[self.t])
        self.t += 1
        return frame


# ============================================================================
#  System
# ============================================================================

class SNNSLAMSystem:
    def __init__(self, key, n_depth=N_DEPTH):
        self.n_depth = n_depth

        # Recurrent weight matrices for the three grid modules
        W_cann_list = [build_2d_cann_weights(size) for size in CANN_SIZES]
        W_cann_asym_x_list = [build_asymmetric_cann_weights_x(size) for size in CANN_SIZES]
        W_cann_asym_y_list = [build_asymmetric_cann_weights_y(size) for size in CANN_SIZES]
        
        # Heading ring attractor (single scale)
        W_ring        = build_1d_ring_weights()
        W_ring_asym   = build_asymmetric_ring_weights()

        k_vision, key = random.split(key)
        self.vision = DualStreamVisionCortex(k_vision, n_pixels=N_PIXELS)

        self.tof_coder = ToFPopulationCoder(n_depth_per_ray=self.n_depth // 3)

        k_pose, key = random.split(key)
        self.pose = PoseCANN(k_pose, W_cann_list, W_ring,
                              W_cann_asym_x_list, W_cann_asym_y_list, W_ring_asym)

        # Place-cell network producing the sparse appearance key
        k_place, key = random.split(key)
        self.place = PlaceCellNetwork(
            key=k_place, 
            n_csnn=256, 
            n_stdp=256, 
            n_depth=self.n_depth, 
            fov_deg=FOV_DEG,
            n_place=FLYHASH_CONFIG["num_bits"],        # 256 place cells (FLYHASH_CONFIG num_bits)
            k_spikes=FLYHASH_CONFIG["active_spikes_k"] # 8 active bits
        )

        self.vision_state = None
        self.place_state = None
        self._initialized = False
        self._step = 0
        
        # Learned gyro-bias estimate (cerebellar calibration; stays 0 unless calibrated)
        self.learned_omega_bias = 0.0
        self.last_decoded_xy = None   # temporal anchor for phase unwrapping
        self.prev_decoded_xy = None   # one step earlier
        self.prev_time_surface = None # previous event time surface for visual odometry

        # Sensory pre-processing and fusion parameters
        self.v_x_scale = 0.01000
        self.v_z_scale = 0.22070
        self.psr_thresh = 4.49000
        self.psr_range = 4.84275
        self.vis_act_thresh = 0.01154
        # Complementary-filter accelerometer-correction gain (ALPHA_FUSE).
        # new = (1-a)*gyro + a*accel; a=0.02 keeps the gyro for high-frequency motion while the
        # gravity (accel) estimate corrects low-frequency drift -- 0.98 gyro / 0.02 accel.
        self.alpha_fuse = ALPHA_FUSE
        self.alpha_acc = 0.18630   # EMA low-pass on raw accel to suppress wingbeat vibration
        self._smooth_omega = None

    def reset(self, B):
        self.vision_state = self.vision.init_state(B)
        self.place_state = self.place.init_state(B)
        self.pose.reset(B)
        self._initialized = False
        self._step = 0
        self.last_decoded_xy = None
        self.prev_decoded_xy = None
        self.prev_time_surface = None
        self._theta_gravity = jnp.zeros((B,))
        self._smooth_acc = None
        self._smooth_omega = None

    def initialize_pose(self, gt_pos, gt_heading):
        self.pose.initialize_pose(gt_pos, gt_heading)
        pose_bump = self.pose.get_state_flat()
        ring_bump = self.pose.get_ring_activity()
        self.place_state = self.place.initialize_from_pose(self.place_state, pose_bump, ring_bump=ring_bump)
        self.last_decoded_xy = gt_pos[:, :2]
        self.prev_decoded_xy = gt_pos[:, :2]  # bootstrap: prev == last so first-frame step = 0
        self._theta_gravity = gt_heading
        self._smooth_acc = None
        self._initialized = True

    def phase_perception(self, events_t, tof_t, learn=True):
        self.vision_state, dual_vis_features = self.vision(self.vision_state, events_t, tof_t[:, 1], learn=learn)
        tof_pop = self.tof_coder(tof_t)
        return dual_vis_features, tof_pop

    def phase_inference(self, dual_vis_features, tof_features, pose_bump, current_heading_rads, ring_bump): 
        vis_csnn, vis_stdp = dual_vis_features
        self.place_state, is_confident, peak_idx_place, debug_gates = self.place.compute_confidence_with_gates(
            self.place_state, vis_csnn, vis_stdp, tof_features, pose_bump, current_heading_rads, ring_bump 
        )
        return is_confident, peak_idx_place, debug_gates

    def phase_odometry(self, kin_t, events_t=None, tof_t=None, theta_gravity=None, inject_drift=False, dt=DT):
        # Subtract the learned gyro bias
        corrected_omega = kin_t[:, 2] - self.learned_omega_bias
        
        # Event-camera visual odometry: rotation shift by phase correlation of the time surfaces of
        # consecutive frames.
        curr_ts = self.vision_state.time_surface
        
        B = kin_t.shape[0]
        if self.prev_time_surface is not None:
            # Grab current visual activity level from vis_csnn
            csnn_clean = jnp.maximum(0.0, self.vision_state.csnn_trace)
            vis_csnn = csnn_clean / (jnp.linalg.norm(csnn_clean, axis=-1, keepdims=True) + 1e-8)
            top16_vis = jnp.sort(vis_csnn, axis=1)[:, -16:]
            vis_act = jnp.mean(top16_vis, axis=1)
            
            # Compute phase correlation, sub-pixel shift, and peak PSR confidence
            sub_pixel_th, psr = get_dvs_rotation_shift(curr_ts, self.prev_time_surface)
            omega_vis = jnp.clip(sub_pixel_th / dt, -6.0, 6.0)
            
            # Calculate dynamic blending weight based on PSR confidence
            vis_trust_raw = jnp.clip((psr - self.psr_thresh) / self.psr_range, 0.0, 1.0)
            vis_trust = jnp.where(vis_act >= self.vis_act_thresh, vis_trust_raw, 0.0)
        else:
            omega_vis = jnp.zeros((B,))
            vis_trust = jnp.zeros((B,))
            
        self.prev_time_surface = curr_ts
        
        # Visual translation velocity [vx_vis, vz_vis] from event rate x ToF distance
        if events_t is not None and tof_t is not None:
            events_left = events_t[:, :128]
            events_right = events_t[:, 128:]
            
            F_left = jnp.mean(jnp.abs(events_left), axis=1)
            F_right = jnp.mean(jnp.abs(events_right), axis=1)
            
            d_left = tof_t[:, 0]
            d_right = tof_t[:, 2]
            
            v_x_vis = self.v_x_scale * (F_left * d_left + F_right * d_right)
            v_z_vis = self.v_z_scale * jnp.abs(F_left * d_left - F_right * d_right)
            
            # Sign the visual velocities by the odometry velocities so the two currents do not oppose
            v_x_vis = jnp.sign(kin_t[:, 0]) * v_x_vis
            v_z_vis = jnp.sign(kin_t[:, 1]) * v_z_vis
            
            v_vis_trans = jnp.stack([v_x_vis, v_z_vis], axis=1)
        else:
            v_vis_trans = jnp.zeros((B, 2))
            
        kin_corrected = jnp.stack([kin_t[:, 0], kin_t[:, 1], corrected_omega], axis=1)

        if inject_drift:
            # Optional constant rate offset (DRIFT_OMEGA) on the bias-corrected rate
            omega_drift = kin_corrected[:, 2] + DRIFT_OMEGA
            kin_injected = jnp.stack([kin_corrected[:, 0], kin_corrected[:, 1], omega_drift], axis=1)
        else:
            kin_injected = kin_corrected

        # Capture the pre-update heading so the predicted displacement is in the
        # correct global frame (matches what the CANN __call__ uses internally).
        theta_pre = self.pose.estimate_heading()  # ring readout BEFORE CANN update

        # Pass visual inputs directly into PoseCANN for direct current injection
        pose_est = self.pose(
            kin_injected,
            omega_vis=omega_vis,
            vis_trust=vis_trust,
            v_vis_trans=v_vis_trans,
            theta_gravity=theta_gravity,
            dt=dt
        )
        
        # Decode the 579-dim grid activity to (x, y)
        pose_bump = self.pose.get_state_flat()
        
        predicted_xy = self.last_decoded_xy  # unwrapping prior: the previous position (no extrapolation)
        
        decoded_xy = decode_grid_to_xy(pose_bump, predicted_xy)
        
        # Advance the two-frame history
        self.prev_decoded_xy = self.last_decoded_xy
        self.last_decoded_xy = decoded_xy
        
        # Update the cerebellar velocity-gain model (IMU and vision inputs)
        self.pose.update_cerebellum(
            kin_injected,
            decoded_xy,
            pose_est[:, 2],
            omega_vis=omega_vis,
            vis_trust=vis_trust,
            v_vis_trans=v_vis_trans,
            dt=dt
        )

        ring_bump = self.pose.get_ring_activity()
        
        # Pose estimate: decoded (x, y) and the ring heading
        final_pose_est = jnp.stack([decoded_xy[:, 0], decoded_xy[:, 1], pose_est[:, 2]], axis=1)
        
        return final_pose_est, pose_bump, ring_bump

    def phase_mapping(self, dual_vis_features, tof_features, pose_bump, ring_bump, heading, angular_vel, confidence=None):
        vis_csnn, vis_stdp = dual_vis_features
        
        self.place_state, (r_place, r_ring) = self.place.forward_mapping(
            self.place_state, vis_csnn, vis_stdp, tof_features, pose_bump, ring_bump=ring_bump, heading=heading, angular_vel=angular_vel, learn=True, confidence=confidence
        )
        
        return r_place, r_ring

    def forward_step(self, events_t, kin_t, tof_t, acc_t=None, inject_drift=False, autopilot_on=True, dt=DT):
        # Accelerometer is carried in kin columns 3:5 ([vx, vy, omega, ax, az]); extract it so
        # the spiking complementary gravity filter runs (closed-loop / anchored configuration).
        if acc_t is None and kin_t.shape[1] >= 5:
            acc_t = kin_t[:, 3:5]
        # Gate STDP learning by kinematic stability: learn only while the low-pass-filtered turn rate is small
        # (the filter suppresses wingbeat vibration)
        if self._smooth_omega is None or self._smooth_omega.shape[0] != kin_t.shape[0]:
            self._smooth_omega = jnp.zeros_like(kin_t[:, 2])
        else:
            alpha_omega = 0.3
            self._smooth_omega = (1.0 - alpha_omega) * self._smooth_omega + alpha_omega * kin_t[:, 2]

        is_stable = jnp.abs(self._smooth_omega) < self.place.dynamic_saccade_thresh
        learn_gate = autopilot_on & is_stable[0]
        dual_vis_features, tof_features = self.phase_perception(events_t, tof_t, learn=learn_gate)

        # Decode the grid activity before this step's update
        pose_bump_prior = self.pose.get_state_flat()
        pose_xy = decode_grid_to_xy(pose_bump_prior, self.last_decoded_xy)
        
        current_heading_rads = self.pose.estimate_heading()
        
        # Heading-ring activity before inference
        ring_bump_prior = self.pose.get_ring_activity()

        is_confident, peak_idx_place, debug_gates = self.phase_inference(
            dual_vis_features, tof_features, pose_bump_prior, current_heading_rads, ring_bump_prior
        )

        # Complementary Filter state estimator for gravity direction (pitch correction)
        if acc_t is not None:
            # 1. Low-pass filter the accelerometer readings (EMA) to suppress high-frequency flapping vibration
            alpha_acc = self.alpha_acc
            if self._smooth_acc is None or self._smooth_acc.shape[0] != acc_t.shape[0]:
                self._smooth_acc = acc_t
            else:
                self._smooth_acc = (1.0 - alpha_acc) * self._smooth_acc + alpha_acc * acc_t
            
            # 2. Extract pitch angle from proper acceleration (acc_x, acc_z)
            ax = self._smooth_acc[:, 0]
            az = self._smooth_acc[:, 1]
            # Extract the absolute pitch from the (low-passed) proper acceleration.
            theta_accel = wrap_angle(jnp.arctan2(ax, az))
            
            # 3. Integrate gyroscope rate (corrected for learned bias)
            corrected_omega = kin_t[:, 2] - self.learned_omega_bias
            theta_gyro = self._theta_gravity + corrected_omega * dt
            theta_gyro = wrap_angle(theta_gyro)
            
            # 4. Fuse using Complementary Filter
            alpha_fuse = self.alpha_fuse
            diff = wrap_angle(theta_accel - theta_gyro)
            self._theta_gravity = wrap_angle(theta_gyro + alpha_fuse * diff)
            
            theta_gravity_val = self._theta_gravity
        else:
            theta_gravity_val = None

        pose_est, pose_bump, ring_bump = self.phase_odometry(
            kin_t,
            events_t=events_t,
            tof_t=tof_t,
            theta_gravity=theta_gravity_val,
            inject_drift=inject_drift,
            dt=dt
        )
        
        # The next frame unwraps around the new position
        self.last_decoded_xy = pose_est[:, :2]

        # Mapping phase with the continuous heading estimate
        r_place, r_ring = self.phase_mapping(dual_vis_features, tof_features, pose_bump, ring_bump, pose_est[:, 2], self._smooth_omega, confidence=None)

        debug_gates = {**debug_gates, "Smooth_Omega": self._smooth_omega, "Learn_Gate": learn_gate}

        self._step += 1
        return pose_est, r_place, r_ring, is_confident, peak_idx_place, debug_gates
