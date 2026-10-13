# idea: keep seed_cube's proven flat 3D-cube round-robin exactly as is, but softly re-rank the per-cube pick queue so points clearly off the frame's dominant plane get pulled a bounded number of slots earlier (instead of reserving a hard off-plane quota like c002, which cannibalized strong/repeatable on-plane picks and hurt fail rate).
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5
PLANE_TOL = 0.08
RANK_BONUS = 40
RANSAC_TRIALS = 60


def _dominant_plane(P, rs):
    n = len(P)
    if n < 4:
        return None
    best = None
    best_cnt = -1
    tri = rs.randint(0, n, size=(RANSAC_TRIALS, 3))
    for a, b, c in tri:
        if a == b or b == c or a == c:
            continue
        nv = np.cross(P[b] - P[a], P[c] - P[a])
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv = nv / nn
        d0 = float(nv @ P[a])
        cnt = int(np.sum(np.abs(P @ nv - d0) < PLANE_TOL))
        if cnt > best_cnt:
            best_cnt = cnt
            best = (nv, d0)
    return best


def pick(u, v, s, z, K):
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])
    # ok preserves the original score-descending order, so position == score rank
    rank = np.arange(len(ok), dtype=float)

    pl = _dominant_plane(P, np.random.RandomState(0))
    if pl is not None:
        nv, d0 = pl
        dist = np.abs(P @ nv - d0)
        off = dist > PLANE_TOL * 1.5
        if np.any(off):
            # soft bias: off-plane points jump ahead within their own cube's queue,
            # but never beyond a bounded number of strictly-stronger on-plane points
            rank = rank - RANK_BONUS * off

    order = np.argsort(rank, kind='stable')

    key = np.floor(P / CUBE).astype(int)
    groups = {}
    for pos in order:
        gk = tuple(key[pos])
        groups.setdefault(gk, []).append(int(ok[pos]))

    group_list = list(groups.values())
    sel = []
    r = 0
    while len(sel) < m and any(r < len(g) for g in group_list):
        sel += [g[r] for g in group_list if r < len(g)]
        r += 1
    sel = sel[:m]

    if len(sel) < m:
        used = set(sel)
        sel += [i for i in range(n) if i not in used][:m - len(sel)]

    return np.array(sel[:m], dtype=int)
