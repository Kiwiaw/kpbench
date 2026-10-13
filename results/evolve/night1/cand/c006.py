# idea: replace the flat cube round-robin with an octree-style progressive refinement, so cells that still hold leftover points after their first pick get halved into finer sub-cells each round instead of yielding a second (spatially redundant) point from the same coarse cell, giving denser point clusters (e.g. the dominant plane) genuinely spread-out picks while leaving sparse off-plane cells untouched.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE0 = 0.5


def pick(u, v, s, z, K):
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])

    # initial grouping into fixed-size cubes, points inside each group kept in
    # original (score-descending) order since `ok` preserves score order.
    key0 = np.floor(P / CUBE0).astype(int)
    groups = {}
    for li in range(len(ok)):
        groups.setdefault(tuple(key0[li]), []).append(li)

    queue = [(idxs, CUBE0) for idxs in groups.values()]
    sel = []

    while queue and len(sel) < m:
        round_groups = queue
        queue = []
        for idxs, csize in round_groups:
            if len(sel) >= m:
                break
            if not idxs:
                continue
            top_local = idxs[0]
            sel.append(int(ok[top_local]))
            rest = idxs[1:]
            if len(rest) == 1:
                queue.append((rest, csize))
            elif len(rest) > 1:
                newc = csize / 2.0
                sub = {}
                for li in rest:
                    subkey = tuple(np.floor(P[li] / newc).astype(int))
                    sub.setdefault(subkey, []).append(li)
                for sidxs in sub.values():
                    queue.append((sidxs, newc))

    sel = sel[:m]
    if len(sel) < m:
        used = set(sel)
        sel += [i for i in range(n) if i not in used][:m - len(sel)]

    return np.array(sel[:m], dtype=int)
