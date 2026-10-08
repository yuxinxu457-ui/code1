"""Layer-1 geometry: SE(2) relative transforms and the translation-only pose-graph relaxation run over the
bounded window."""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def se2_between(frm, to):
    """Body-frame relative transform from pose `frm`=(x,y,th) to pose `to`=(x,y,th)."""
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    cc, ss = np.cos(-frm[2]), np.sin(-frm[2])
    return (dx * cc - dy * ss, dx * ss + dy * cc, wrap(to[2] - frm[2]))


def relax_xy(pos, th, chain, loops, n_iter=2000, alpha=0.15, w_loop=0.5, beta=0.9):
    """Translation-only pose-graph relaxation with heavy-ball momentum.

    Headings `th` are held fixed (they come from the ring attractor); only (x, y) are relaxed.
    chain: dict src -> (dst, dx, dy, dth), odometry edge src->dst with weight 1;
    loops: dict (a, b) -> (dx, dy, dth), loop edge with weight w_loop. Offsets are in the body frame of the
    source node. Node 0 is the gauge and stays fixed. Returns the relaxed positions (n, 2)."""
    pos = np.asarray(pos, float).copy(); n = len(pos); p0 = pos[0].copy(); vel = np.zeros((n, 2))
    ea, eb, eox, eoy, ew = [], [], [], [], []
    for src, (dst, dx, dy, dth) in chain.items():
        ea.append(src); eb.append(dst); eox.append(dx); eoy.append(dy); ew.append(1.0)
    for (a, b), (dx, dy, dth) in loops.items():
        ea.append(a); eb.append(b); eox.append(dx); eoy.append(dy); ew.append(w_loop)
    if not ea:
        return pos
    ea = np.array(ea); eb = np.array(eb); ew = np.array(ew)
    th = np.asarray(th, float)
    c = np.cos(th[ea]); s = np.sin(th[ea])
    ex = c * np.array(eox) - s * np.array(eoy)                 # world-frame expected displacement (headings fixed)
    ey = s * np.array(eox) + c * np.array(eoy)
    for _ in range(n_iter):
        rx = pos[eb, 0] - pos[ea, 0] - ex; ry = pos[eb, 1] - pos[ea, 1] - ey
        d = np.zeros((n, 2)); w = np.zeros(n) + 1e-9
        np.add.at(d, eb, -np.stack([ew * rx, ew * ry], 1)); np.add.at(d, ea, np.stack([ew * rx, ew * ry], 1))
        np.add.at(w, eb, ew); np.add.at(w, ea, ew)
        vel = beta * vel + alpha * d / w[:, None]
        pos += vel
        pos[0] = p0; vel[0] = 0.0                              # gauge node fixed
    return pos
