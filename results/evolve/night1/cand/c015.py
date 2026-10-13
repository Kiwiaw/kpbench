# idea: replace seed_cube's isotropic 3D cube grid with a PCA-aligned anisotropic "slab" grid (fine within the dominant plane, coarse along its normal) so the proven round-robin collapses redundant near-plane depth noise into few cells while still giving truly off-plane points their own distinct, well-covered cells, directly exploiting the documented benefit of off-plane points and wide coverage without disturbing the score-order round-robin logic that already works.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE_INPLANE = 0.35   # fine spacing within the dominant plane -> good spatial spread
CUBE_NORMAL = 1.0     # coarse spacing along the plane normal -> collapses depth-noise on the
                      # dominant plane, while truly off-plane points still land in their own
                      # distinct, rare cells and are not starved by round-robin


def _dominant_frame(P):
    n = len(P)
    if n < 3:
        return np.eye(3)
    c = P - P.mean(axis=0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.eye(3)
    if Vt.shape[0] < 3 or not np.all(np.isfinite(Vt)):
        return np.eye(3)
    # rows of Vt: directions of decreasing variance; last row = normal of best-fit plane
    R = np.vstack([Vt[0], Vt[1], Vt[2]])
    if not np.all(np.isfinite(R)):
        return np.eye(3)
    return R


def _cube_priority(ok, P):
    R = _dominant_frame(P)
    Pr = P @ R.T
    scale = np.array([CUBE_INPLANE, CUBE_INPLANE, CUBE_NORMAL])
    key = np.floor(Pr / scale).astype(int)
    groups = {}
    order = []
    for jj in range(len(ok)):
        k = (int(key[jj, 0]), int(key[jj, 1]), int(key[jj, 2]))
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(int(ok[jj]))
    pr = []
    r = 0
    while True:
        added = False
        for k in order:
            g = groups[k]
            if r < len(g):
                pr.append(g[r])
                added = True
        if not added:
            break
        r += 1
    return pr


def pick(u, v, s, z, K):
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if len(ok) == 0:
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])
    cube_rank = _cube_priority(ok, P)

    sel = cube_rank[:m]
    if len(sel) < m:
        used = set(sel)
        leftover = [i for i in range(n) if i not in used]
        sel = sel + leftover[: m - len(sel)]

    return np.array(sel[:m], dtype=int)
