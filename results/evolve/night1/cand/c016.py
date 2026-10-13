# idea: replace seed_cube's fixed 0.5 m cube size with a per-frame adaptive isotropic cube size derived from the robust (5th-95th percentile) 3D extent of the camera-frame point cloud, so scenes with very different depth/spatial scale (close-range vs far/near-planar) get a grid density matched to ~2K target cells instead of a one-size-fits-all 0.5 m voxel, improving spatial spread and thus pose stability while keeping the proven round-robin selection and camera-origin anchoring unchanged.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

MIN_CUBE = 0.15      # metres, floor so dense/close scenes don't get degenerate tiny cubes
MAX_CUBE = 1.2       # metres, ceiling so sparse/far scenes don't collapse everything into one cell
DEFAULT_CUBE = 0.5   # seed_cube's original size, used as a safe fallback


def _adaptive_cube(P, K):
    n = len(P)
    if n < 8:
        return DEFAULT_CUBE
    lo = np.percentile(P, 5, axis=0)
    hi = np.percentile(P, 95, axis=0)
    ext = np.maximum(hi - lo, 1e-3)
    vol = float(ext[0] * ext[1] * ext[2])
    target = max(2 * K, 50)  # aim for roughly 2 points-worth of cells so round-robin spreads well
    cube = (vol / target) ** (1.0 / 3.0)
    if not np.isfinite(cube) or cube <= 0:
        return DEFAULT_CUBE
    return float(np.clip(cube, MIN_CUBE, MAX_CUBE))


def pick(u, v, s, z, K):
    u = np.asarray(u, dtype=float)
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])  # (nk, 3), rows in score-descending order (ok preserves input order)
    cube = _adaptive_cube(P, m)

    key = np.floor(P / cube).astype(np.int64)
    g = {}
    order = []
    for jj, j in enumerate(ok):
        kk = (int(key[jj, 0]), int(key[jj, 1]), int(key[jj, 2]))
        lst = g.get(kk)
        if lst is None:
            g[kk] = [int(j)]
            order.append(kk)
        else:
            lst.append(int(j))

    sel = []
    r = 0
    while len(sel) < m:
        added = False
        for kk in order:
            lst = g[kk]
            if r < len(lst):
                sel.append(lst[r])
                added = True
                if len(sel) >= m:
                    break
        if not added:
            break
        r += 1

    sel = sel[:m]
    if len(sel) < m:
        used = set(sel)
        for i in range(n):
            if len(sel) >= m:
                break
            if i not in used:
                sel.append(i)

    return np.array(sel, dtype=int)
