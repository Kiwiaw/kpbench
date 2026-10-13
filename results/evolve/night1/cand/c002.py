# idea: fit the dominant 3D plane by RANSAC and reserve ~35 of the 100 picks for off-plane points (cube round robin), then fill the rest by cube round robin over all points, because off-plane points constrain pose on near-planar scenes.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5
OFF_QUOTA = 35
PLANE_TOL = 0.08


def _round_robin(idx, P, k, taken):
    # one point per CUBE-sized cube per round, in score order (idx is ascending = score order)
    key = np.floor(P / CUBE).astype(int); g = {}
    for jj, j in enumerate(idx):
        if int(j) in taken:
            continue
        g.setdefault(tuple(key[jj]), []).append(int(j))
    groups = list(g.values()); sel = []; r = 0
    while len(sel) < k and any(r < len(x) for x in groups):
        sel += [x[r] for x in groups if r < len(x)]; r += 1
    return sel[:k]


def _dominant_plane(P):
    n = len(P)
    if n < 4:
        return None
    rs = np.random.RandomState(0)
    best = None; best_cnt = -1
    tri = rs.randint(0, n, size=(80, 3))
    for a, b, c in tri:
        nv = np.cross(P[b] - P[a], P[c] - P[a]); nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv = nv / nn
        cnt = int(np.sum(np.abs((P - P[a]) @ nv) < PLANE_TOL))
        if cnt > best_cnt:
            best_cnt = cnt; best = (nv, float(nv @ P[a]))
    return best


def pick(u, v, s, z, K):
    m = min(K, len(u))
    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(m)
    P = to3d(u[ok], v[ok], z[ok]); sel = []; taken = set()
    pl = _dominant_plane(P)
    if pl is not None:
        dist = np.abs(P @ pl[0] - pl[1])
        off = dist >= PLANE_TOL * 1.5
        if np.any(off):
            part = _round_robin(ok[off], P[off], min(OFF_QUOTA, m), taken)
            sel += part; taken.update(part)
    rest = _round_robin(ok, P, m - len(sel), taken)
    sel += rest; taken.update(rest)
    if len(sel) < m:
        sel += [i for i in range(len(u)) if i not in taken][:m - len(sel)]
    return np.array(sel[:m], int)
