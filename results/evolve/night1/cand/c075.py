# idea: replace the parent's fixed 0.5 m-voxel cube round-robin fill with a greedy farthest-point sampling (FPS) in camera-frame 3D (ties broken by score order, since input is score-descending), so the spatial-spread fill adapts to each scene's actual depth scale instead of forcing one voxel size on every near/far planar scene.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020/c027)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = 6         # tightly bounded repair budget
SOFT_OFF_FRAC = 0.5   # fraction of the robust threshold used as a "soft off-plane" cutoff in repair
BETA = 0.3            # strength of corner-proximity bonus when choosing an in-cell repair candidate
MAX_CORNER_DIST = (320.0 ** 2 + 240.0 ** 2) ** 0.5   # farthest any pixel can be from its nearest corner (image centre)


def _fps_fill(P, idx_ok, taken_set, k):
    """Greedily grow a 3D-spread set of original indices by farthest-point sampling.

    P and idx_ok are aligned, score-descending arrays (P camera-frame 3D coords,
    idx_ok original candidate indices). Already-selected original indices are
    given in taken_set: they seed the running nearest-distance field so new
    picks spread away from them, and are excluded from the candidate pool."""
    if k <= 0:
        return []

    n_all = len(idx_ok)
    taken_mask = np.fromiter((int(idx_ok[i]) in taken_set for i in range(n_all)), dtype=bool, count=n_all)
    avail_pos = np.flatnonzero(~taken_mask)
    if len(avail_pos) == 0:
        return []

    Pa = P[avail_pos]
    idx_a = idx_ok[avail_pos]
    n = len(avail_pos)

    seed_pos = np.flatnonzero(taken_mask)
    sel = []
    remaining_mask = np.ones(n, dtype=bool)

    if len(seed_pos) > 0:
        seeds = P[seed_pos]
        d2 = np.full(n, np.inf)
        for tp in seeds:
            diff = Pa - tp
            dd = np.einsum('ij,ij->i', diff, diff)
            d2 = np.minimum(d2, dd)
    else:
        # no existing selection to spread from: seed with the single strongest
        # available candidate (idx_a[0], since the input is score-descending)
        first = 0
        sel.append(int(idx_a[first]))
        remaining_mask[first] = False
        diff = Pa - Pa[first]
        d2 = np.einsum('ij,ij->i', diff, diff)

    while len(sel) < k and np.any(remaining_mask):
        d2_masked = np.where(remaining_mask, d2, -np.inf)
        nxt = int(np.argmax(d2_masked))
        sel.append(int(idx_a[nxt]))
        remaining_mask[nxt] = False
        diff = Pa - Pa[nxt]
        dd = np.einsum('ij,ij->i', diff, diff)
        d2 = np.minimum(d2, dd)

    return sel


def _corner_proximity(ui, vi):
    # distance to nearest of the 4 image corners, normalised so corners -> 1, image centre -> 0
    d0 = (ui ** 2 + vi ** 2) ** 0.5
    d1 = ((640.0 - ui) ** 2 + vi ** 2) ** 0.5
    d2 = (ui ** 2 + (480.0 - vi) ** 2) ** 0.5
    d3 = ((640.0 - ui) ** 2 + (480.0 - vi) ** 2) ** 0.5
    dmin = min(d0, d1, d2, d3)
    return 1.0 - min(1.0, dmin / MAX_CORNER_DIST)


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
                for c in off_local[:quota]:
                    orig = int(ok[c])
                    if orig not in taken:
                        sel.append(orig)
                        taken.add(orig)
                        off_quota_picks.add(orig)

    # ---- fill the remainder with greedy 3D farthest-point sampling, spreading
    #      away from any off-plane quota picks already chosen. Unlike a fixed
    #      voxel-size cube grid, this adapts automatically to each scene's depth
    #      scale (a close-up tabletop and a long corridor need very different
    #      cube sizes; FPS needs none). ----
    remaining = m - len(sel)
    if remaining > 0:
        rr = _fps_fill(P, ok, taken, remaining)
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

    # ---- bounded image-coverage repair: fill genuinely empty coarse grid cells.
    #      Within each empty cell, prefer a still-off-plane candidate (boolean split),
    #      but among candidates in the same bucket pick the one whose score is
    #      boosted most by closeness to the nearest image corner (wide baseline
    #      helps PnP). Original off-plane quota picks stay protected from ever
    #      being swapped out as repair "victims". ----
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
