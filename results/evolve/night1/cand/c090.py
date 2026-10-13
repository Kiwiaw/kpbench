# idea: in the parent's off-plane quota, replace plain strongest-score selection with a farthest-point-in-image-space sampling drawn from the top POOL_MULT*OFF_QUOTA strongest off-plane candidates, so the quota's wide-baseline points are also spread across the image instead of clustering wherever the single strongest off-plane region happens to be, directly injecting the known "wide image coverage helps PnP" signal into the off-plane-selection stage itself rather than leaving it only to the post-hoc grid repair.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011)
POOL_MULT = 3         # off-plane quota is FPS-sampled from the top POOL_MULT*OFF_QUOTA strongest off-plane candidates
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = 6         # tightly bounded repair budget
SOFT_OFF_FRAC = 0.5   # fraction of the robust threshold used as a "soft off-plane" cutoff in repair


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


def _fps_off_quota(off_local, ok, u, v, quota):
    # off_local: indices into the "ok" / P arrays of off-plane candidates, already score-descending.
    # Build a pool of the top POOL_MULT*quota strongest off-plane candidates, then greedily
    # farthest-point-sample `quota` of them in pixel space (starting from the single strongest),
    # so the quota both stays near the best-scoring off-plane evidence and spreads across the image.
    if quota <= 0 or len(off_local) == 0:
        return []

    poolsize = min(len(off_local), max(quota, quota * POOL_MULT))
    pool = off_local[:poolsize]
    orig_idx = ok[pool]
    pts = np.stack([u[orig_idx], v[orig_idx]], axis=1).astype(float)  # (poolsize, 2)
    n_pool = len(pool)
    quota = min(quota, n_pool)

    selected_mask = np.zeros(n_pool, dtype=bool)
    selected_order = [0]
    selected_mask[0] = True
    if quota == 1:
        return [int(orig_idx[0])]

    min_dist = np.linalg.norm(pts - pts[0], axis=1)

    for _ in range(1, quota):
        cand = np.where(selected_mask, -1.0, min_dist)
        nxt = int(np.argmax(cand))
        if cand[nxt] < 0:
            break
        selected_mask[nxt] = True
        selected_order.append(nxt)
        d_new = np.linalg.norm(pts - pts[nxt], axis=1)
        min_dist = np.minimum(min_dist, d_new)

    return [int(orig_idx[i]) for i in selected_order]


def pick(u, v, s, z, K):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    s = np.asarray(s, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if len(ok) == 0:
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])   # (nk, 3), rows in score-descending order

    sel = []
    taken = set()
    off_quota_picks = set()   # protected: never used as a repair "victim"

    # dist[pos] (pos indexes ok/P) = robust off-plane distance of that candidate; -1 if not computed
    dist_full = np.full(len(ok), -1.0, dtype=float)
    soft_thresh = None

    # ---- deterministic dominant-plane fit via PCA (smallest-eigenvalue normal) ----
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
            dist_full = dist
            med = float(np.median(dist))
            mad = float(np.median(np.abs(dist - med))) * 1.4826 + 1e-9
            thresh = max(MIN_THRESH, med + MAD_K * mad)
            soft_thresh = SOFT_OFF_FRAC * thresh
            off_mask = dist > thresh
            if np.any(off_mask):
                off_local = np.flatnonzero(off_mask)   # already score-descending
                quota = min(OFF_QUOTA, m, len(off_local))
                fps_picks = _fps_off_quota(off_local, ok, u, v, quota)
                for orig in fps_picks:
                    if orig not in taken:
                        sel.append(orig)
                        taken.add(orig)
                        off_quota_picks.add(orig)

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

    sel = sel[:m]

    # ---- bounded image-coverage repair: fill genuinely empty coarse grid cells,
    #      preferring each empty cell's strongest still-off-plane candidate so the
    #      off-plane and wide-coverage signals stack instead of competing, and
    #      never swapping out an original off-plane quota pick ----
    if m >= 2 and len(sel) >= 2:
        cw = 640.0 / GX
        ch = 480.0 / GY
        cx_img, cy_img = 320.0, 240.0

        def cell_of(ui, vi):
            gx = int(min(GX - 1, max(0, np.floor(ui / cw))))
            gy = int(min(GY - 1, max(0, np.floor(vi / ch))))
            return gy * GX + gx

        def cell_center(c):
            gx = c % GX
            gy = c // GX
            return (gx + 0.5) * cw, (gy + 0.5) * ch

        sel_arr = np.array(sel, dtype=int)
        sel_cells = {cell_of(u[i], v[i]) for i in sel_arr}

        ok_set_taken = set(sel_arr.tolist())
        best_any_in_cell = {}
        best_off_in_cell = {}
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            c = cell_of(u[orig], v[orig])
            if c not in best_any_in_cell:
                best_any_in_cell[c] = orig
            if soft_thresh is not None and c not in best_off_in_cell and dist_full[pos] > soft_thresh:
                best_off_in_cell[c] = orig

        empty_cells = [c for c in range(GX * GY) if c not in sel_cells and c in best_any_in_cell]
        # corners-first ordering: farthest from image center processed first so the
        # tightly bounded swap budget preferentially restores corner coverage
        empty_cells.sort(
            key=lambda c: -((cell_center(c)[0] - cx_img) ** 2 + (cell_center(c)[1] - cy_img) ** 2)
        )

        protected = set(off_quota_picks)
        swaps = 0
        sel_list = list(sel_arr)
        for c in empty_cells:
            if swaps >= MAX_SWAPS:
                break
            candidate = best_off_in_cell.get(c, best_any_in_cell[c])
            # find weakest (lowest score) currently selected, unprotected point
            scores = np.array([s[i] for i in sel_list])
            order_by_weak = np.argsort(scores)  # ascending score
            victim_pos = None
            for p in order_by_weak:
                if sel_list[p] not in protected:
                    victim_pos = int(p)
                    break
            if victim_pos is None:
                break
            victim = sel_list[victim_pos]
            sel_list[victim_pos] = candidate
            protected.add(candidate)
            ok_set_taken.discard(victim)
            ok_set_taken.add(candidate)
            swaps += 1

        sel = sel_list

    return np.array(sel[:m], dtype=int)
