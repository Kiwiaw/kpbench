# idea: fix the parent's joint 3D+2D farthest-point sampling by running it over a de-noised candidate pool (the top-2 strongest points of each proven 0.5 m 3D cube, exactly as in seed_cube) instead of all 2000 raw points, so coverage-maximising FPS cannot be hijacked by noisy/outlier single points and still gets explicit, simultaneous 3D + image-plane spread.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5          # proven voxel size from seed_cube (metres)
REP_PER_CELL = 2     # keep the 2 strongest points per voxel as FPS candidates
IMG_WEIGHT = 4.0     # scales normalised pixel-plane distance into cube units
ALPHA = 1.0          # score-decay rate (seed_dpp convention)


def pick(u, v, s, z, K):
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if len(ok) == 0:
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])          # (nk, 3) camera-frame metres
    nk = len(ok)
    rank = np.arange(nk)                   # ok preserves score-descending order

    # ---- step 1: voxel-bucket the valid-depth points (seed_cube style) ----
    key = np.floor(P / CUBE).astype(np.int64)
    buckets = {}
    for pos in range(nk):
        k3 = (int(key[pos, 0]), int(key[pos, 1]), int(key[pos, 2]))
        lst = buckets.get(k3)
        if lst is None:
            buckets[k3] = [pos]
        elif len(lst) < REP_PER_CELL:
            lst.append(pos)

    rep_pos = np.array(sorted(x for lst in buckets.values() for x in lst), dtype=int)
    if len(rep_pos) == 0:
        rep_pos = np.arange(nk)

    # ---- step 2: joint 3D + 2D feature space over the de-noised pool ----
    Pr = P[rep_pos] / CUBE
    uvr = np.stack(
        [(u[ok][rep_pos] / 640.0 - 0.5), (v[ok][rep_pos] / 480.0 - 0.5)], axis=1
    ) * IMG_WEIGHT
    F = np.concatenate([Pr, uvr], axis=1)
    quality = np.exp(-ALPHA * rank[rep_pos] / 100.0)

    nr = len(rep_pos)
    kmax = min(m, nr)

    # start from the strongest representative (lowest rank = best score)
    start = int(np.argmin(rank[rep_pos]))
    sel = [start]
    mind = np.sum((F - F[start]) ** 2, axis=1)
    mind[start] = -np.inf

    for _ in range(1, kmax):
        metric = np.sqrt(np.maximum(mind, 0.0)) * quality
        j = int(np.argmax(metric))
        if metric[j] <= 0.0:
            break
        sel.append(j)
        d = np.sum((F - F[j]) ** 2, axis=1)
        mind = np.minimum(mind, d)
        mind[j] = -np.inf

    sel_idx = list(ok[rep_pos[np.array(sel, dtype=int)]])

    # ---- step 3: fill any remainder by overall score order ----
    if len(sel_idx) < m:
        used = set(sel_idx)
        fill = [i for i in range(n) if i not in used]
        sel_idx += fill[: m - len(sel_idx)]

    return np.array(sel_idx[:m], dtype=int)
