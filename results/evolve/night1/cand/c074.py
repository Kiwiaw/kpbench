# idea: replace the parent's global-PCA dominant-plane estimate (smallest eigenvalue of the whole cloud's covariance, which can be biased by any large coherent off-plane structure) with a RANSAC-style minimal-sample-consensus plane fit followed by a PCA refit on its inlier set, so the off-plane/on-plane classification used for the OFF_QUOTA reserve and the repair's soft-off-plane bucket targets the plane that actually contains the most points, instead of whatever direction happens to minimise total squared perpendicular distance.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020/c027)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold / RANSAC inlier threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = GX * GY   # bounded only by the grid size itself: let every empty cell be repaired
SOFT_OFF_FRAC = 0.5   # fraction of the robust threshold used as a "soft off-plane" cutoff in repair
BETA = 0.3            # strength of corner-proximity bonus when choosing an in-cell repair candidate
MAX_CORNER_DIST = (320.0 ** 2 + 240.0 ** 2) ** 0.5   # farthest any pixel can be from its nearest corner (image centre)

RANSAC_ITERS = 50     # minimal-sample-consensus iterations for dominant-plane search
RANSAC_SEED = 1234    # fixed seed keeps the whole picker deterministic


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


def _corner_proximity(ui, vi):
    # distance to nearest of the 4 image corners, normalised so corners -> 1, image centre -> 0
    d0 = (ui ** 2 + vi ** 2) ** 0.5
    d1 = ((640.0 - ui) ** 2 + vi ** 2) ** 0.5
    d2 = (ui ** 2 + (480.0 - vi) ** 2) ** 0.5
    d3 = ((640.0 - ui) ** 2 + (480.0 - vi) ** 2) ** 0.5
    dmin = min(d0, d1, d2, d3)
    return 1.0 - min(1.0, dmin / MAX_CORNER_DIST)


def _ransac_plane_dist(P):
    """Robust dominant-plane perpendicular distance for every row of P (nk,3).

    Minimal-sample-consensus over random triples, scored by MIN_THRESH-inlier
    count, then refined by a PCA refit restricted to the winning inlier set.
    Returns None if no usable plane could be found (degenerate / too few points).
    """
    n_pts = P.shape[0]
    if n_pts < 6:
        return None

    rng = np.random.default_rng(RANSAC_SEED)
    n_iters = min(RANSAC_ITERS, max(10, n_pts - 1))

    best_count = -1
    best_normal = None
    best_centroid = None

    for _ in range(n_iters):
        idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = P[idx[0]], P[idx[1]], P[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(normal)
        if nn < 1e-9:
            continue
        normal = normal / nn
        d = -float(normal @ p0)
        dist = np.abs(P @ normal + d)
        count = int(np.sum(dist < MIN_THRESH))
        if count > best_count:
            best_count = count
            best_normal = normal
            best_centroid = p0
            best_dist = dist

    if best_normal is None:
        return None

    # refine using the winning inlier set (PCA on just those points), which is
    # less sensitive to any single coherent off-plane cluster than global PCA
    inlier_mask = best_dist < MIN_THRESH
    if np.sum(inlier_mask) >= 3:
        inliers = P[inlier_mask]
        centroid = inliers.mean(axis=0)
        Pc = inliers - centroid
        cov = Pc.T @ Pc
        evals, evecs = np.linalg.eigh(cov)
        refined_normal = evecs[:, 0]
        rn = np.linalg.norm(refined_normal)
        if rn > 1e-9:
            refined_normal = refined_normal / rn
            dist_full = np.abs((P - centroid) @ refined_normal)
            return dist_full

    # fallback: use the raw RANSAC winning plane if refit was degenerate
    return np.abs(P @ best_normal - float(best_normal @ best_centroid))


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

    # dist_full[pos] (pos indexes ok/P) = robust off-plane distance of that candidate; -1 if not computed
    dist_full = np.full(len(ok), -1.0, dtype=float)
    soft_thresh = None

    # ---- RANSAC minimal-sample-consensus dominant-plane fit (replaces global PCA) ----
    ransac_dist = _ransac_plane_dist(P)
    if ransac_dist is not None:
        dist_full = ransac_dist
        med = float(np.median(dist_full))
        mad = float(np.median(np.abs(dist_full - med))) * 1.4826 + 1e-9
        thresh = max(MIN_THRESH, med + MAD_K * mad)
        soft_thresh = SOFT_OFF_FRAC * thresh
        off_mask = dist_full > thresh
        if np.any(off_mask):
            off_local = np.flatnonzero(off_mask)   # already score-descending
            quota = min(OFF_QUOTA, m, len(off_local))
            for c in off_local[:quota]:
                orig = int(ok[c])
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

    # ---- bounded (but full-grid-sized) image-coverage repair: fill every genuinely
    #      empty coarse grid cell, preferring each empty cell's still-off-plane
    #      candidate so the off-plane and wide-coverage signals stack instead of
    #      competing, and never swapping out an original off-plane quota pick.
    #      Within each cell/bucket, pick the candidate whose score is boosted most
    #      by closeness to the nearest image corner. (unchanged from the parent) ----
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
        best_any_in_cell = {}   # cell -> (orig_index, desirability)
        best_off_in_cell = {}   # cell -> (orig_index, desirability)
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            c = cell_of(u[orig], v[orig])
            desir = s[orig] * (1.0 + BETA * _corner_proximity(u[orig], v[orig]))

            cur_any = best_any_in_cell.get(c)
            if cur_any is None or desir > cur_any[1]:
                best_any_in_cell[c] = (orig, desir)

            if soft_thresh is not None and dist_full[pos] > soft_thresh:
                cur_off = best_off_in_cell.get(c)
                if cur_off is None or desir > cur_off[1]:
                    best_off_in_cell[c] = (orig, desir)

        empty_cells = [c for c in range(GX * GY) if c not in sel_cells and c in best_any_in_cell]
        # corners-first ordering: farthest from image center processed first so the
        # repair preferentially restores corner/edge coverage before the center
        empty_cells.sort(
            key=lambda c: -((cell_center(c)[0] - cx_img) ** 2 + (cell_center(c)[1] - cy_img) ** 2)
        )

        protected = set(off_quota_picks)
        swaps = 0
        sel_list = list(sel_arr)
        for c in empty_cells:
            if swaps >= MAX_SWAPS:
                break
            candidate = best_off_in_cell[c][0] if c in best_off_in_cell else best_any_in_cell[c][0]
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
