# idea: anchor the cube round-robin grid to the 3D point cloud's robust centroid (median) instead of the camera origin, so that for small inter-frame camera translations the cube index of the same physical point stays stable (centroid-relative coordinates cancel first-order camera translation), directly targeting the documented importance of repeatability, while keeping seed_cube's proven coverage/strength trade-off and c011's small deterministic PCA off-plane quota untouched; this replaces the parent's pixel-space NMS pass, which the scoreboard shows collapses repeatability (rep5 0.2362) and raises fail%.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5          # seed_cube's proven voxel size (metres), left untouched
OFF_QUOTA = 12       # small, conservative reserve for off-plane points (as in c011)
MIN_THRESH = 0.04    # metres, floor on the robust off-plane distance threshold
MAD_K = 2.5          # robust-scale multiplier on median-absolute-deviation


def _cube_round_robin(idx_order, keys, taken, k):
    # one point per cube per round; idx_order/keys rows are in score-descending order
    if k <= 0:
        return []
    groups = {}
    order = []
    for pos in range(len(idx_order)):
        orig = int(idx_order[pos])
        if orig in taken:
            continue
        kk = keys[pos]
        lst = groups.get(kk)
        if lst is None:
            groups[kk] = [orig]
            order.append(kk)
        else:
            lst.append(orig)

    sel = []
    r = 0
    while len(sel) < k:
        added = False
        for kk in order:
            g = groups[kk]
            if r < len(g):
                sel.append(g[r])
                added = True
                if len(sel) >= k:
                    break
        if not added:
            break
        r += 1
    return sel


def pick(u, v, s, z, K):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if len(ok) == 0:
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])   # (nk, 3), rows in score-descending order (ok preserves input order)

    sel = []
    taken = set()

    # ---- robust centroid of the local 3D point cloud, used to anchor the cube grid ----
    centroid = np.median(P, axis=0)
    Pc = P - centroid

    # ---- deterministic dominant-plane fit via PCA (smallest-eigenvalue normal), centred on the same centroid ----
    if len(P) >= 6:
        cov = Pc.T @ Pc
        evals, evecs = np.linalg.eigh(cov)   # ascending eigenvalues
        normal = evecs[:, 0]
        nn = np.linalg.norm(normal)
        if nn > 1e-9:
            normal = normal / nn
            dist = np.abs(Pc @ normal)
            med = float(np.median(dist))
            mad = float(np.median(np.abs(dist - med))) * 1.4826 + 1e-9
            thresh = max(MIN_THRESH, med + MAD_K * mad)
            off_mask = dist > thresh
            if np.any(off_mask):
                off_local = np.flatnonzero(off_mask)   # already score-descending (same order as ok/P)
                quota = min(OFF_QUOTA, m, len(off_local))
                for c in off_local[:quota]:
                    orig = int(ok[c])
                    if orig not in taken:
                        sel.append(orig)
                        taken.add(orig)

    # ---- fill the remainder with a centroid-anchored 0.5 m cube round-robin ----
    remaining = m - len(sel)
    if remaining > 0:
        key = np.floor(Pc / CUBE).astype(np.int64)
        keys = [(int(key[i, 0]), int(key[i, 1]), int(key[i, 2])) for i in range(len(ok))]
        rr = _cube_round_robin(ok, keys, taken, remaining)
        sel += rr
        taken.update(rr)

    # ---- final fallback: any leftover indices (e.g. sky points) by original order ----
    if len(sel) < m:
        for i in range(n):
            if len(sel) >= m:
                break
            if i not in taken:
                sel.append(i)
                taken.add(i)

    return np.array(sel[:m], dtype=int)
