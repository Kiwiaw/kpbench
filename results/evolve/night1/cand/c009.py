# idea: keep seed_cube's exact 3D-cube round-robin and per-cube score ranking unchanged, but when the round-robin must truncate mid-round (the only point where "which points make the final 100" is actually decided for near-planar scenes with many equally-strong cubes), pick that round's subset via greedy farthest-point sampling in 2D image space instead of arbitrary dict order, directly injecting the known "wide image coverage helps pose" benefit exactly where it matters without disturbing the proven 3D spread/repeatability/strength structure that drives the parent's low fail rate.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5


def pick(u, v, s, z, K):
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])
    key = np.floor(P / CUBE).astype(int)

    groups = {}
    for jj, j in enumerate(ok):
        groups.setdefault(tuple(key[jj]), []).append(int(j))
    group_list = list(groups.values())  # insertion order ~ score order of each cube's strongest member

    sel = []
    r = 0
    while len(sel) < m and any(r < len(g) for g in group_list):
        cand = [g[r] for g in group_list if r < len(g)]
        need = m - len(sel)

        if len(cand) <= need:
            sel += cand
        else:
            # This round must be truncated: choose the subset that maximises 2D image
            # coverage via greedy farthest-point sampling, seeded by the round's
            # strongest-cube-first candidate (preserves the parent's score priority).
            cu = u[cand]
            cv = v[cand]
            mind = (cu - cu[0]) ** 2 + (cv - cv[0]) ** 2
            mind[0] = -np.inf
            chosen_idx = [0]
            for _ in range(need - 1):
                nxt = int(np.argmax(mind))
                chosen_idx.append(nxt)
                d = (cu - cu[nxt]) ** 2 + (cv - cv[nxt]) ** 2
                mind = np.minimum(mind, d)
                mind[nxt] = -np.inf
            sel += [cand[i] for i in chosen_idx]
        r += 1

    sel = sel[:m]
    if len(sel) < m:
        used = set(sel)
        sel += [i for i in range(n) if i not in used][:m - len(sel)]

    return np.array(sel[:m], dtype=int)
