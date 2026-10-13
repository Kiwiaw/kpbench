# idea: replace the pure 3D cube grid with a greedy farthest-point selection in a joint 3D+2D feature space (camera-frame position normalized by the cube scale, plus pixel coordinates scaled to match), so spread is enforced simultaneously in 3D (helps conditioning) and across the image plane (helps PnP on near-planar scenes), with a score-decay weight to keep preferring strong points -- this combines the two previously-separate "3D spread" and "wide image coverage" ingredients that no prior variant used together.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5          # normalises 3D camera-frame distance (metres -> cube units)
IMG_WEIGHT = 4.0     # scales normalised pixel-plane distance into the same units
ALPHA = 1.0          # score-decay rate (same convention as seed_dpp)


def pick(u, v, s, z, K):
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if len(ok) == 0:
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok]) / CUBE
    uv = np.stack(
        [(u[ok] / 640.0 - 0.5), (v[ok] / 480.0 - 0.5)], axis=1
    ) * IMG_WEIGHT
    F = np.concatenate([P, uv], axis=1)  # (nk, 5) joint 3D+2D feature space

    nk = len(ok)
    # ok preserves original (score-descending) order among valid-depth points
    rank = np.arange(nk)
    quality = np.exp(-ALPHA * rank / 100.0)

    kmax = min(m, nk)
    sel = [0]
    mind = np.sum((F - F[0]) ** 2, axis=1)
    mind[0] = -np.inf

    for _ in range(1, kmax):
        metric = np.sqrt(np.maximum(mind, 0.0)) * quality
        j = int(np.argmax(metric))
        if metric[j] <= 0.0:
            break
        sel.append(j)
        d = np.sum((F - F[j]) ** 2, axis=1)
        mind = np.minimum(mind, d)
        mind[j] = -np.inf

    sel_idx = list(ok[np.array(sel, dtype=int)])

    if len(sel_idx) < m:
        used = set(sel_idx)
        fill = [i for i in range(n) if i not in used]
        sel_idx += fill[: m - len(sel_idx)]

    return np.array(sel_idx[:m], dtype=int)
