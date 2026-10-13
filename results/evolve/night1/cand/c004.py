# idea: make the cube size adaptive to each frame's 3D point-cloud extent (axis-wise floor-clamped so a near-planar frame can't collapse the grid) instead of a fixed 0.5 m, so the round-robin spread auto-tunes to the scene's actual scale and keeps working on both tight and expansive near-planar query frames.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

TARGET_CELLS_PER_PICK = 3.0   # aim for ~3x as many spatial bins as picks requested
MIN_AXIS_EXTENT = 0.3         # metres; floors degenerate (near-planar / tiny) extents
MIN_CUBE = 0.25
MAX_CUBE = 1.0


def pick(u, v, s, z, K):
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])

    if len(P) >= 2:
        bbox = np.maximum(P.max(axis=0) - P.min(axis=0), MIN_AXIS_EXTENT)
    else:
        bbox = np.full(3, MIN_AXIS_EXTENT)

    vol = float(np.prod(bbox))
    target_cells = max(1.0, TARGET_CELLS_PER_PICK * m)
    cube = (vol / target_cells) ** (1.0 / 3.0)
    cube = min(max(cube, MIN_CUBE), MAX_CUBE)

    key = np.floor(P / cube).astype(int)
    g = {}
    for jj, j in enumerate(ok):
        g.setdefault(tuple(key[jj]), []).append(int(j))

    groups = list(g.values())
    sel = []
    r = 0
    while len(sel) < m and any(r < len(x) for x in groups):
        sel += [x[r] for x in groups if r < len(x)]
        r += 1
    sel = sel[:m]

    if len(sel) < m:
        used = set(sel)
        sel += [i for i in range(n) if i not in used][:m - len(sel)]

    return np.array(sel, dtype=int)
