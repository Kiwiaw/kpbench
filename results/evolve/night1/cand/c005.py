# idea: keep the proven fixed-size 3D cube round-robin ranking from seed_cube unchanged, but add a lightweight greedy 2D pixel-space non-maximum-suppression pass on top of that ranking (shrinking the suppression radius over a few passes) so the final 100 picks are explicitly forced to spread across the image plane, directly exploiting the documented fact that wide image coverage helps PnP on near-planar frames, without touching the 3D binning that already works well.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5          # same fixed cube size as the strong seed_cube baseline
NMS_SCALE = 0.45    # fraction of the average "uniform spread" pixel spacing used as base NMS radius


def _cube_priority(ok, P):
    # one point per 3D cube per round, in score order (ok is ascending index = score order)
    key = np.floor(P / CUBE).astype(int)
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

    ok_set = set(int(i) for i in ok)
    leftover = [i for i in range(n) if i not in ok_set]
    full_rank = cube_rank + leftover  # covers every index exactly once, in priority order

    px = np.stack([np.asarray(u, dtype=float), np.asarray(v, dtype=float)], axis=1)

    base_r = NMS_SCALE * np.sqrt(640.0 * 480.0 / max(m, 1))
    radii = (base_r, base_r * 0.5, base_r * 0.25, 0.0)  # progressively relax, last pass accepts anything

    sel = []
    sel_set = set()
    sel_px = np.zeros((m, 2), dtype=float)
    nsel = 0

    for radius in radii:
        if nsel >= m:
            break
        r2 = radius * radius
        for idx in full_rank:
            if nsel >= m:
                break
            if idx in sel_set:
                continue
            p = px[idx]
            if nsel == 0 or r2 <= 0.0:
                ok_dist = True
            else:
                diff = sel_px[:nsel] - p
                ok_dist = np.min(diff[:, 0] ** 2 + diff[:, 1] ** 2) >= r2
            if ok_dist:
                sel.append(idx)
                sel_set.add(idx)
                sel_px[nsel] = p
                nsel += 1

    if nsel < m:
        for i in range(n):
            if nsel >= m:
                break
            if i not in sel_set:
                sel.append(i)
                sel_set.add(i)
                nsel += 1

    return np.array(sel[:m], dtype=int)
