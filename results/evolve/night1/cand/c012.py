# idea: instead of running farthest-point sampling over the whole raw/denoised candidate pool (which let noisy outliers hijack the parent and c008), apply a tiny, bounded 2D-pixel-space FPS only to the top-40 strongest already-off-plane points (from c011's proven PCA/MAD plane fit) to pick the <=12-point off-plane quota, explicitly maximising image coverage exactly where it matters for near-planar pose stability, then fall back unchanged to seed_cube's proven 0.5 m 3D round-robin for the rest.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5          # seed_cube's proven voxel size (metres), left untouched
OFF_QUOTA = 12       # small, conservative reserve for off-plane points (as in c011)
OFF_POOL = 40        # bounded, deterministic pool of strongest off-plane candidates for 2D FPS
MIN_THRESH = 0.04    # metres, floor on the robust off-plane distance threshold
MAD_K = 2.5          # robust-scale multiplier on median-absolute-deviation
ALPHA = 1.0          # score-decay rate used to break FPS ties toward stronger points


def _cube_round_robin(idx_order, P, taken, k):
    # one point per CUBE-sized cube per round; idx_order/P rows are in score-descending order
    if k <= 0:
        return []
    key = np.floor(P / CUBE).astype(np.int64)
    groups = {}
    order = []
    for pos in range(len(idx_order)):
        orig = int(idx_order[pos])
        if orig in taken:
            continue
        kk = (int(key[pos, 0]), int(key[pos, 1]), int(key[pos, 2]))
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


def _fps_2d(idx_pool, u_pool, v_pool, quota):
    # deterministic greedy farthest-point sampling in normalised pixel space,
    # over a small bounded pool only; score-rank weighting breaks ties toward
    # stronger candidates instead of letting outliers dominate.
    n = len(idx_pool)
    if n == 0 or quota <= 0:
        return []
    uv = np.stack([u_pool / 640.0, v_pool / 480.0], axis=1)
    quality = np.exp(-ALPHA * np.arange(n) / max(1, n))

    sel_pos = [0]  # strongest-score candidate in the pool first
    mind = np.sum((uv - uv[0]) ** 2, axis=1)
    mind[0] = -np.inf

    kmax = min(quota, n)
    while len(sel_pos) < kmax:
        metric = np.sqrt(np.maximum(mind, 0.0)) * quality
        j = int(np.argmax(metric))
        if metric[j] <= 0.0:
            break
        sel_pos.append(j)
        d = np.sum((uv - uv[j]) ** 2, axis=1)
        mind = np.minimum(mind, d)
        mind[j] = -np.inf

    return [int(idx_pool[p]) for p in sel_pos]


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

    # ---- deterministic dominant-plane fit via PCA (smallest-eigenvalue normal), as in c011 ----
    if len(P) >= 6:
        centroid = P.mean(axis=0)
        Pc = P - centroid
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
                off_local = np.flatnonzero(off_mask)         # score-descending positions into ok/P
                pool_local = off_local[:OFF_POOL]             # bounded strongest off-plane pool
                pool_idx = ok[pool_local]

                fps_sel = _fps_2d(pool_idx, u[pool_idx], v[pool_idx], min(OFF_QUOTA, m))
                for orig in fps_sel:
                    if orig not in taken:
                        sel.append(orig)
                        taken.add(orig)

    # ---- fill the remainder with seed_cube's exact 0.5 m cube round-robin ----
    remaining = m - len(sel)
    if remaining > 0:
        rr = _cube_round_robin(ok, P, taken, remaining)
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
