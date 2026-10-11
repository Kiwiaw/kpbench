# seed: 3D cube round robin (cube0.5): one point per 0.5 m camera-frame cube in score order, repeated
import numpy as np
from kpbench.data import to3d, DEPTH_MAX


def pick(u, v, s, z, K):
    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(min(K, len(u)))
    key = np.floor(to3d(u[ok], v[ok], z[ok]) / 0.5).astype(int); g = {}
    for jj, j in enumerate(ok):
        g.setdefault(tuple(key[jj]), []).append(int(j))
    groups = list(g.values()); sel = []; r = 0
    while len(sel) < K and any(r < len(x) for x in groups):
        sel += [x[r] for x in groups if r < len(x)]; r += 1
    sel = sel[:K]
    if len(sel) < min(K, len(u)):
        used = set(sel); sel += [i for i in range(len(u)) if i not in used][:min(K, len(u)) - len(sel)]
    return np.array(sel, int)
