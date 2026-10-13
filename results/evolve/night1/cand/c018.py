# idea: keep seed_cube's exact 3D-cube bucketing and round-robin cadence, but reorder the few candidates inside each cube by a blend of score-strength and a fast deterministic PCA/SVD off-plane distance, so off-plane points surface in earlier rounds (and thus survive into the top-100) without any separate reservation phase or extra selection pass that could disturb the proven cube spreading.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5
PLANE_WEIGHT = 0.18  # modest bonus: strength still dominates, off-plane only breaks near-ties / early rounds


def pick(u, v, s, z, K):
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])
    nk = len(ok)

    # --- fast deterministic dominant-plane distance via PCA/SVD (no RANSAC, no randomness) ---
    if nk >= 4:
        c = np.median(P, axis=0)
        Pc = P - c
        try:
            _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
            normal = Vt[-1]
            nrm = np.linalg.norm(normal)
            if nrm > 1e-9:
                normal = normal / nrm
                dist = np.abs(Pc @ normal)
            else:
                dist = np.zeros(nk)
        except np.linalg.LinAlgError:
            dist = np.zeros(nk)
    else:
        dist = np.zeros(nk)

    dmax = dist.max() if nk else 0.0
    ndist = dist / dmax if dmax > 1e-9 else dist  # normalized to [0,1], 0 if degenerate

    # ok is already in score-descending order (original index order == score order)
    strength = 1.0 - (np.arange(nk, dtype=float) / max(nk - 1, 1))  # 1 = strongest, 0 = weakest

    combined = strength + PLANE_WEIGHT * ndist

    key = np.floor(P / CUBE).astype(int)
    g = {}
    for jj in range(nk):
        kk = (int(key[jj, 0]), int(key[jj, 1]), int(key[jj, 2]))
        g.setdefault(kk, []).append(jj)

    # within each cube, order local members by the combined (score + off-plane) key,
    # so off-plane points get picked in earlier rounds when they are competitive on score
    groups = []
    for members in g.values():
        members.sort(key=lambda jj: -combined[jj])
        groups.append([int(ok[jj]) for jj in members])

    sel = []
    r = 0
    while len(sel) < m and any(r < len(x) for x in groups):
        sel += [x[r] for x in groups if r < len(x)]
        r += 1
    sel = sel[:m]

    if len(sel) < m:
        used = set(sel)
        sel += [i for i in range(n) if i not in used][:m - len(sel)]

    return np.array(sel[:m], dtype=int)
