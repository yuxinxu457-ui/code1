#!/usr/bin/env python3
"""Simulated 2 x 2 m arena with textured obstacles, a 1-D event camera (256 pixels, 90 deg field of view) and
three time-of-flight rays.

Two courses are provided: 'l2probe', the two-scale revisit course used by the SLAM runs (a figure-eight of a
small anchor loop A and a larger excursion loop B), and 'explore', a room-filling repeated Lissajous path
(used by the capacity test to measure the live bump amplitude).
"""

import os
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOM_W = 2.0
ROOM_H = 2.0

N_PIXELS = 256
FOV_DEG = 90.0
DT = 0.02
TIME_STEPS = 2000

BARCODE_RESOLUTION = 512
THRESHOLD = 0.015           # event threshold on the per-pixel intensity change

OBS_SIZE_MIN = 0.02         # obstacle size range (m)
OBS_SIZE_MAX = 0.14
OBS_MARGIN = 0.4            # obstacle-free wall margin (m)

VX_RANGE = (-0.5, 0.5)      # forward speed range (m/s)
VY_RANGE = (-0.15, 0.15)    # lateral speed range (m/s)
OMEGA_RANGE = (-1.0, 1.0)   # turn-rate range (rad/s)

TEX_FREQS = [0.5, 1.0, 2.0]
TEX_AMPS  = [0.8, 0.4, 0.2]

# =============================================================================
# Arena geometry
# =============================================================================
def obstacles_to_segments(obstacles):
    room_segs = jnp.array([
        [[0, 0], [ROOM_W, 0]], [[ROOM_W, 0], [ROOM_W, ROOM_H]],
        [[ROOM_W, ROOM_H], [0, ROOM_H]], [[0, ROOM_H], [0, 0]],
    ], dtype=jnp.float32)
    def _rect(r):
        x0, y0, x1, y1 = r
        return jnp.array([
            [[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]],
            [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]],
        ], dtype=jnp.float32)
    obs_segs = jax.vmap(_rect)(obstacles).reshape(-1, 2, 2)
    return jnp.concatenate([room_segs, obs_segs], axis=0)

def _build_observations(positions, headings, segments, surface_textures, jax_obstacles, tex_t):
    """Event stream, ToF distances and pixel intensities along a path.

    SLAM_OBS_SLICE>0 evaluates the per-timestep ray casting in slices of that many steps (the
    T x N_PIXELS x N_SEGMENTS intermediate grows with T and exhausts memory on long paths); each slice
    is materialised to numpy at once, so the peak footprint is one slice. Results are identical to the
    unsliced evaluation (SLAM_OBS_SLICE=0).
    """
    _slice = int(os.environ.get('SLAM_OBS_SLICE', '0'))
    T = int(positions.shape[0])
    if _slice > 0 and T > _slice:
        inten_parts, dist_parts, tof_parts = [], [], []
        for t0 in range(0, T, _slice):
            t1 = min(t0 + _slice, T)
            r = jax.vmap(
                lambda p, h: compute_pixel_readings(p, h, segments, surface_textures, jax_obstacles, tex_t)
            )(positions[t0:t1], headings[t0:t1])
            inten_parts.append(np.asarray(r[0])); dist_parts.append(np.asarray(r[1]))
            tof_parts.append(np.asarray(
                jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions[t0:t1], headings[t0:t1], segments)))
        intensities = jnp.asarray(np.concatenate(inten_parts, axis=0))
        distances = jnp.asarray(np.concatenate(dist_parts, axis=0))
        tof_dists = jnp.asarray(np.concatenate(tof_parts, axis=0))
    else:
        readings = jax.vmap(
            lambda p, h: compute_pixel_readings(p, h, segments, surface_textures, jax_obstacles, tex_t)
        )(positions, headings)
        intensities = readings[0]
        distances = readings[1]
        tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions, headings, segments)
    prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
    delta = intensities - prev
    events = jnp.where(delta > THRESHOLD, 1.0, jnp.where(delta < -THRESHOLD, -1.0, 0.0))
    events = events.at[0].set(0.0)
    return events, intensities, distances, tof_dists

def _make_explore_trajectory(key, time_steps, dt):
    """'explore' course: a room-filling, self-crossing Lissajous path repeated lap after lap.

    The closed path revisits every point once per lap at the same heading (the heading is tangent to
    the path). Amplitude, frequencies (default 3:2) and speed are env-tunable (EXPLORE_*). Velocities
    are derived from the path so vx and omega stay consistent (the robot faces its direction of motion)."""
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)
    A   = float(os.environ.get('EXPLORE_AMP', '0.80'))
    fa  = float(os.environ.get('EXPLORE_FA', '3.0'))
    fb  = float(os.environ.get('EXPLORE_FB', '2.0'))
    speed = float(os.environ.get('EXPLORE_SPEED', '0.45'))       # cruising speed (m/s)
    phx = rng.uniform(0, 2*np.pi); direction = 1.0 if rng.uniform() > 0.5 else -1.0
    cx = ROOM_W/2.0 + rng.uniform(-0.03, 0.03); cy = ROOM_H/2.0 + rng.uniform(-0.03, 0.03)

    # Arc-length parametrisation: sample the closed Lissajous densely in its parameter U and resample it
    # in time so the robot moves at a physical speed. One period (U: 0 -> 2 pi) is one lap; the resampling
    # wraps around it, so laps repeat.
    U = np.linspace(0.0, 2*np.pi, 20000, endpoint=False); dU = U[1] - U[0]
    xp  =  A*fa*np.cos(fa*U + phx);      yp  =  A*fb*np.cos(fb*U)          # dp/dU
    xpp = -A*fa*fa*np.sin(fa*U + phx);   ypp = -A*fb*fb*np.sin(fb*U)       # d2p/dU2
    spU = np.hypot(xp, yp)                                                 # |dp/dU|
    kappa = np.abs(xp*ypp - yp*xpp) / (spU**3 + 1e-9)                      # path curvature
    ds = spU * dU                                                         # arc-length increments
    # Curvature-limited speed: slow down in tight turns so that omega = v * kappa <= omega_max, and
    # cruise at `speed` elsewhere.
    omega_max = float(os.environ.get('EXPLORE_OMEGA_MAX', '0.9'))
    v_loc = np.minimum(speed, omega_max / (kappa + 1e-6))
    dt_seg = ds / (v_loc + 1e-9)
    T = np.concatenate([[0.0], np.cumsum(dt_seg)]); lap_time = T[-1]
    want_t = (np.arange(time_steps) * dt * direction) % lap_time          # cumulative time, wrapped
    idx_u = np.interp(want_t, T[:-1], U, period=lap_time)
    x = cx + A*np.sin(fa*idx_u + phx); y = cy + A*np.sin(fb*idx_u)
    pos = np.stack([x, y], 1).astype(np.float32)
    dx = np.gradient(x); dy = np.gradient(y)
    hdg = np.arctan2(dy, dx) % (2*np.pi)
    omega = np.gradient(np.unwrap(hdg), dt)
    vx = (np.hypot(dx, dy) / dt).astype(np.float32); vx[0] = 0.0          # actual (variable) speed
    omega = omega.astype(np.float32); omega[0] = 0.0
    return (jnp.array(pos), jnp.array(hdg.astype(np.float32)),
            jnp.array(vx), jnp.array(omega))


def generate_explore_obstacles(key, path_xy, n=None, clear=None):
    """Scatter `n` landmark obstacles across the room, each at least `clear` m from the path and 0.10 m
    from the other obstacles, so that different places see different obstacle configurations."""
    n = int(os.environ.get('EXPLORE_NOBS', '44')) if n is None else n
    clear = float(os.environ.get('EXPLORE_CLEAR', '0.09')) if clear is None else clear
    rng = np.random.RandomState(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    P = np.asarray(path_xy); obs = []
    tries = 0
    while len(obs) < n and tries < 4000:
        tries += 1
        cxy = rng.uniform(OBS_MARGIN, ROOM_W - OBS_MARGIN, 2)
        if np.min(np.hypot(P[:, 0] - cxy[0], P[:, 1] - cxy[1])) < clear:
            continue
        if obs and np.min([np.hypot(cxy[0]-o[0], cxy[1]-o[1]) for o in obs]) < 0.10:
            continue
        sz = rng.uniform(OBS_SIZE_MIN, OBS_SIZE_MAX)
        # corner format [x0, y0, x1, y1]
        obs.append([cxy[0]-sz/2, cxy[1]-sz/2, cxy[0]+sz/2, cxy[1]+sz/2])
    return jnp.array(np.array(obs, dtype=np.float32))


def _make_l2probe_trajectory(key, time_steps, dt):
    """'l2probe' course: the two-scale revisit course used by the SLAM runs.

    Geometry: a figure-eight of two externally tangent circles, a small anchor loop A (traversed
    counter-clockwise) and a larger excursion loop B (traversed clockwise), meeting at the tangency point
    T. With these senses the tangent direction is continuous at T, so the path is C1 without connector
    segments and its curvature is bounded by 1/rA and 1/rB.

    Schedule: A x L2P_ALAPS (default 2), then (B, A) x L2P_ALTS (default 3).
      * The opening A laps give short-gap revisits at the same heading, which a bounded window still
        holds, so they are closed in-window.
      * Each pass of B creates new places, so the places of A leave the window, and accumulates drift.
      * Each return to A revisits places that have been evicted from the window.
    Every lap of A is traversed in the same sense, so each point of A is seen at the same heading on all
    laps. The seed jitters the radii, the position of A and the A->B axis."""
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    _rng = np.random.RandomState(key_np)
    rA = float(os.environ.get('L2P_RA', '0.32')) * (1.0 + _rng.uniform(-0.08, 0.08))
    rB = float(os.environ.get('L2P_RB', '0.50')) * (1.0 + _rng.uniform(-0.08, 0.08))
    n_alaps = int(os.environ.get('L2P_ALAPS', '2'))       # A laps before the first excursion
    n_blaps = int(os.environ.get('L2P_BLAPS', '1'))       # B laps per excursion
    # Only parameters that preserve the tangency are jittered (cB and the tangency angles are derived
    # from cA and the axis u, so the C1 join at T is kept).
    cAx = 0.58 + _rng.uniform(-0.02, 0.02)                 # per-seed geometry jitter
    cAy = 0.58 + _rng.uniform(-0.02, 0.02)
    _ang = np.pi / 4.0 + _rng.uniform(-0.08, 0.08)         # cA -> cB axis, jittered
    u = np.array([np.cos(_ang), np.sin(_ang)])
    cB = np.array([cAx, cAy]) + (rA + rB) * u             # externally tangent
    thA0 = np.arctan2(u[1], u[0])                         # angle of T on circle A
    thB0 = np.arctan2(-u[1], -u[0])                       # angle of T on circle B

    nA = int(os.environ.get('L2P_NSAMP_A', '4000'))
    nB = int(os.environ.get('L2P_NSAMP_B', '9000'))

    def arcA(laps):                                        # CCW from T
        t = thA0 + np.linspace(0.0, 2*np.pi*laps, nA*laps, endpoint=False)
        return np.stack([cAx + rA*np.cos(t), cAy + rA*np.sin(t)], 1)

    def arcB(laps):                                        # CW from T
        t = thB0 - np.linspace(0.0, 2*np.pi*laps, nB*laps, endpoint=False)
        return np.stack([cB[0] + rB*np.cos(t), cB[1] + rB*np.sin(t)], 1)

    # Schedule A x n_alaps, then (B, A) x n_alts. Every arc starts and ends at T, so any A/B
    # concatenation is C1. Every return to A revisits places evicted from the window.
    n_alts = int(os.environ.get('L2P_ALTS', '3'))
    segs = [arcA(n_alaps)]
    for _ in range(n_alts):
        segs.append(arcB(n_blaps)); segs.append(arcA(1))
    P = np.concatenate(segs, 0)

    # Constant-speed resample over the step budget: v = L / (steps * dt), so omega = v * kappa. The
    # heading is interpolated from the source tangent rather than differenced from the resampled points.
    d1 = np.gradient(P, axis=0)
    sp = np.hypot(d1[:, 0], d1[:, 1]) + 1e-12
    s = np.concatenate([[0.0], np.cumsum(sp)])
    want = np.linspace(0.0, s[-1]*(1 - 1e-9), time_steps)
    idx = np.interp(want, s[:-1], np.arange(len(P)))
    x = np.interp(idx, np.arange(len(P)), P[:, 0])
    y = np.interp(idx, np.arange(len(P)), P[:, 1])
    th_src = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
    hd = np.interp(idx, np.arange(len(P)), th_src)
    positions = jnp.asarray(np.stack([x, y], 1))
    headings = jnp.asarray(hd)
    vx = jnp.asarray(np.hypot(np.gradient(x), np.gradient(y)) / dt)   # the robot faces its direction of travel
    omega = jnp.asarray(np.gradient(hd) / dt)
    return positions, headings, vx, omega


def _explore_room_dataset(rng, n_samples, obstacles, time_steps, dt, traj_fn=None):
    """Build n_samples trajectories with their sensor streams in one room. traj_fn selects the path
    generator (the explore Lissajous by default, or the l2probe schedule); obstacles are generated along
    the first path when not given, and surface textures are seeded from the obstacle geometry."""
    traj_fn = traj_fn or _make_explore_trajectory
    import hashlib
    ev_l, lab_l, tof_l, pos_l, hdg_l, int_l = [], [], [], [], [], []
    for _ in range(n_samples):
        k_traj = jax.random.PRNGKey(rng.randint(0, 2**31))
        positions, headings, vx, omega = traj_fn(k_traj, time_steps, dt)
        vy = jnp.zeros_like(vx)
        obs = obstacles
        if obs is None:
            obs = generate_explore_obstacles(jax.random.PRNGKey(rng.randint(0, 2**31)), np.array(positions))
        segments = obstacles_to_segments(obs)
        obs_np = np.array(obs)
        room_seed = int(hashlib.md5(obs_np.tobytes()).hexdigest(), 16) % (2**31 - 1)
        surface_textures = _generate_surface_textures(obs_np, room_seed)
        tex_t = _precompute_barcode_tensors(surface_textures, obs_np)
        events, intensities, distances, tof_dists = _build_observations(
            positions, headings, segments, surface_textures, jnp.asarray(obs), tex_t)
        min_clear_arr = jnp.min(distances, axis=1)
        clearance_norm = jnp.tanh(min_clear_arr / 2.0)
        labels = jnp.stack([vx / abs(VX_RANGE[1]), vy / abs(VY_RANGE[1]),
                            omega / abs(OMEGA_RANGE[1]), clearance_norm], axis=1)
        ev_l.append(events); lab_l.append(labels); tof_l.append(tof_dists)
        pos_l.append(positions); hdg_l.append(headings); int_l.append(intensities)
    return (jnp.stack(ev_l), jnp.stack(lab_l), jnp.stack(tof_l),
            jnp.stack(pos_l), jnp.stack(hdg_l), obs, segments, jnp.stack(int_l))


# =============================================================================
# Ray–Segment Intersection
# =============================================================================
def cast_rays(origins, directions, segments):
    A = segments[:, 0, :]
    B = segments[:, 1, :]
    E = B - A
    D = directions[:, None, :]
    diff = A[None, :, :] - origins[:, None, :]
    det = (D[:, :, 0] * E[None, :, 1] - D[:, :, 1] * E[None, :, 0])
    safe = jnp.where(jnp.abs(det) > 1e-10, det, 1.0)
    t = (diff[:, :, 0] * E[None, :, 1] - diff[:, :, 1] * E[None, :, 0]) / safe
    s = (diff[:, :, 0] * D[:, :, 1] - diff[:, :, 1] * D[:, :, 0]) / safe
    valid = (jnp.abs(det) > 1e-10) & (t > 0.01) & (s >= 0) & (s <= 1)
    dists = jnp.where(valid, t, 1e6)
    hit_pts = origins[:, None, :] + t[:, :, None] * directions[:, None, :]
    return dists, hit_pts

# =============================================================================
# Surface textures and pixel intensities
# =============================================================================
def _barcode_texture(barcode_key, local_coords):
    rng = np.random.RandomState(int(barcode_key) & 0xFFFFFFFF)
    n_stripes = rng.randint(3, 8) 
    boundaries = sorted([0.0] + list(rng.uniform(0.0, 1.0, n_stripes - 1)) + [1.0])
    boundaries = np.array(boundaries, dtype=np.float32)
    brightness = rng.uniform(0.15, 0.95, n_stripes).astype(np.float32)
    stripe_idx = np.searchsorted(boundaries[1:], local_coords)
    stripe_idx = np.clip(stripe_idx, 0, n_stripes - 1)
    base = brightness[stripe_idx] 

    for freq, amp in zip(TEX_FREQS, TEX_AMPS):
        phase_a = rng.uniform(0, 2 * np.pi)
        along_mod = np.cos(2 * np.pi * freq * local_coords + phase_a)
        base = base * (1.0 + amp * 0.6 * along_mod)

    pattern = np.clip(base, 0.05, 1.5)
    return pattern.astype(np.float32)

def _generate_surface_textures(obstacles, room_seed):
    """Per-segment barcode textures (4 walls, then 4 sides per obstacle), seeded from room_seed."""
    rng = np.random.RandomState(int(room_seed) & 0xFFFFFFFF)
    textures = {}

    wall_seeds = [rng.randint(0, 2**31) for _ in range(4)]
    wall_coords = np.linspace(0, 1, BARCODE_RESOLUTION)
    for i, seed in enumerate(wall_seeds):
        textures[i] = _barcode_texture(seed, wall_coords)

    n_obstacles = obstacles.shape[0]
    coords = np.linspace(0.0, 1.0, BARCODE_RESOLUTION)
    for obs_idx in range(n_obstacles):
        for side in range(4):
            seg_idx = 4 + obs_idx * 4 + side
            seed = int(rng.randint(0, 2**31))
            textures[seg_idx] = _barcode_texture(seed, coords)

    return textures

def compute_tof_distance(robot_pos, robot_heading, segments, include_back=False):
    if include_back:
        angles = robot_heading + jnp.array([-jnp.pi/4, 0.0, jnp.pi/4, jnp.pi])
        n_rays = 4
    else:
        angles = robot_heading + jnp.array([-jnp.pi/4, 0.0, jnp.pi/4])
        n_rays = 3
    origins = jnp.broadcast_to(robot_pos, (n_rays, 2))
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    dists, _ = cast_rays(origins, directions, segments) 
    min_dists = jnp.min(dists, axis=-1) 

    max_range = 2.83  # diagonal of the 2 x 2 m room
    tof_dists = jnp.clip(min_dists, 0.0, max_range)

    return tof_dists

def _precompute_barcode_tensors(surface_textures, obstacles):
    """Pre-compute texture tensors for fast vectorized lookup."""
    seg_ids = sorted(surface_textures.keys())
    n_surf = max(seg_ids) + 1 if seg_ids else 0

    tex_rows = []
    for seg_id in range(n_surf):
        if seg_id in surface_textures:
            tex = np.array(surface_textures[seg_id], dtype=np.float32)
            tex_rows.append(tex)
        else:
            tex_rows.append(np.zeros(BARCODE_RESOLUTION, dtype=np.float32))

    return jnp.stack([jnp.array(t) for t in tex_rows]) if tex_rows else jnp.zeros((0, BARCODE_RESOLUTION))


def _sample_barcode_fast(min_idx, nearest, min_dist, obstacles, tex_tensor):
    n_pix = min_idx.shape[0]
    hx = nearest[:, 0]
    hy = nearest[:, 1]

    is_wall = min_idx < 4
    t_wall = jnp.stack([
        hx / ROOM_W,   
        hy / ROOM_H,   
        hx / ROOM_W,   
        hy / ROOM_H,   
    ], axis=1) 

    obs_seg_ids = min_idx - 4 
    obs_idx = obs_seg_ids // 4   
    side = obs_seg_ids % 4      

    # clip indices so wall hits (obs_idx < 0) index safely; they are masked below
    max_obs_idx = max(0, obstacles.shape[0] - 1)
    safe_obs_idx = jnp.clip(obs_idx, 0, max_obs_idx)
    
    _obs_x0 = jnp.take(obstacles[:, 0], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.zeros(n_pix)
    _obs_y0 = jnp.take(obstacles[:, 1], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.zeros(n_pix)
    _obs_x1 = jnp.take(obstacles[:, 2], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.ones(n_pix)
    _obs_y1 = jnp.take(obstacles[:, 3], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.ones(n_pix)
    
    dx = _obs_x1 - _obs_x0 + 1e-8
    dy = _obs_y1 - _obs_y0 + 1e-8

    t_obs = jnp.where(
        side == 0, (hx - _obs_x0) / dx,
        jnp.where(
            side == 1, (hy - _obs_y0) / dy,
            jnp.where(
                side == 2, (hx - _obs_x0) / dx,
                (hy - _obs_y0) / dy
            )
        )
    ) 

    # Clip min_idx to [0, 3] to prevent out-of-bounds lookup on t_wall (which only has 4 columns)
    safe_min_idx = jnp.clip(min_idx, 0, 3)
    t_wall_selected = jnp.take_along_axis(t_wall, safe_min_idx[:, None], axis=1)[:, 0]
    t_all = jnp.where(is_wall, t_wall_selected, t_obs) 

    tex_for_pix = jnp.take(tex_tensor, min_idx, axis=0, mode='clip') 

    # map the position along the surface (0..1) to the texture resolution
    tex_idx = jnp.clip(jnp.floor(t_all * BARCODE_RESOLUTION).astype(jnp.int32), 0, BARCODE_RESOLUTION - 1) 
    
    batch_idx = jnp.arange(n_pix) 
    intensity = tex_for_pix[batch_idx, tex_idx]

    t_depth = min_dist / 2.83   # distance normalised by the room diagonal
    
    for freq, amp in zip(TEX_FREQS, TEX_AMPS):
        phase = jnp.pi * freq 
        depth_mod = jnp.cos(2.0 * jnp.pi * freq * 6.0 * t_depth + phase)
        intensity = intensity * (1.0 + 0.6 * amp * depth_mod)

    return jnp.clip(intensity, 0.05, 1.5)


def _sample_barcode_textures(min_idx, nearest, min_dist, surface_textures, obstacles=None, tex_tensor=None):
    if tex_tensor is not None:
        return _sample_barcode_fast(min_idx, nearest, min_dist, obstacles, tex_tensor)
    return jnp.zeros(N_PIXELS, dtype=jnp.float32) 


def compute_pixel_readings(robot_pos, robot_heading, segments, surface_textures=None, obstacles=None, tex_tensor=None):
    fov_rad = jnp.radians(FOV_DEG)
    angles = robot_heading + jnp.linspace(-fov_rad/2, fov_rad/2, N_PIXELS)
    origins = jnp.broadcast_to(robot_pos, (N_PIXELS, 2))
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    
    dists, hit_pts = cast_rays(origins, dirs, segments)
    min_idx = jnp.argmin(dists, axis=-1)
    min_dist = jnp.min(dists, axis=-1)
    nearest = hit_pts[jnp.arange(N_PIXELS), min_idx]
    hit_type = (min_idx >= 4).astype(jnp.float32)

    intensities = _sample_barcode_textures(min_idx, nearest, min_dist, surface_textures, obstacles, tex_tensor)

    return intensities, min_dist, hit_type, nearest


# =============================================================================
# Dataset builder
# =============================================================================
def generate_fixed_room_dataset(key, n_samples, obstacles=None, time_steps=TIME_STEPS, dt=DT,
                                course_type='l2probe'):
    """Trajectories and sensor streams for one room. course_type: 'l2probe' (two-scale revisit course,
    see _make_l2probe_trajectory) or 'explore' (repeated Lissajous, see _make_explore_trajectory)."""
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)

    if course_type == 'explore':
        return _explore_room_dataset(rng, n_samples, obstacles, time_steps, dt)

    if course_type == 'l2probe':
        return _explore_room_dataset(rng, n_samples, obstacles, time_steps, dt,
                                     traj_fn=_make_l2probe_trajectory)

    raise ValueError(f"unknown course_type {course_type!r} (expected 'l2probe' or 'explore')")
