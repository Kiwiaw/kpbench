# idea: in c088's corner-restricted, off-availability-first coverage repair, replace the single naive PCA plane-normal estimate (fit on all points including off-plane ones, which biases the covariance and hence the normal) with a one-step IRLS-refined robust PCA: fit once, Tukey-biweight-downweight points whose residual exceeds the robust MAD threshold, then refit the centroid/covariance/normal on the downweighted cloud, so the dominant-plane direction is estimated mainly from genuinely planar points and the resulting off-plane classification (used both for the quota and for repair desirability) is more accurate on near-planar scenes with real 3D structure.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX
from collections import defaultdict

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020/c027/c042)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = 6         # tightly bounded repair budget
SOFT_OFF_FRAC = 0.5   # fraction of the robust threshold used as a "soft off-plane" cutoff in repair
BETA_CORNER = 0.5     # strength of corner-proximity bonus, applied only within the 4 true corner cells
MAX_CORNER_DIST = (320.0 ** 2 + 240.0 ** 2) ** 0.5   # farthest any pixel can be from its nearest corner (image centre)


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

    # ---- robust dominant-plane fit: naive PCA, then one IRLS (Tukey biweight)
    #      reweighting pass so the final normal/centroid are driven mainly by the
    #      genuinely planar majority rather than by whatever off-plane structure
    #      happens to be in the candidate set. ----
    if len(P) >= 6:
        centroid0 = P.mean(axis=0)
        Pc0 = P - centroid0
        cov0 = Pc0.T @ Pc0
        evals0, evecs0 = np.linalg.eigh(cov0)   # ascending eigenvalues
        normal0 = evecs0[:, 0]
        nn0 = np.linalg.norm(normal0)
        if nn0 > 1e-9:
            normal0 = normal0 / nn0
            dist0 = np.abs(Pc0 @ normal0)
            med0 = float(np.median(dist0))
            mad0 = float(np.median(np.abs(dist0 - med0))) * 1.4826 + 1e-9
            c_tukey = max(MIN_THRESH, med0 + MAD_K * mad0)

            r = dist0 / (c_tukey + 1e-9)
            w = np.where(r < 1.0, (1.0 - r ** 2) ** 2, 0.0)
            wsum = float(w.sum())

            normal = normal0
            centroid = centroid0
            if wsum > 1e-6:
                centroid_w = (P * w[:, None]).sum(axis=0) / wsum
                Pcw = P - centroid_w
                covw = (Pcw * w[:, None]).T @ Pcw
                evals_w, evecs_w = np.linalg.eigh(covw)
                normal_w = evecs_w[:, 0]
                nnw = np.linalg.norm(normal_w)
                if nnw > 1e-9:
                    normal = normal_w / nnw
                    centroid = centroid_w

            Pc = P - centroid
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

    # ---- bounded image-coverage repair: fill genuinely empty coarse grid cells.
    #      Within each empty cell, prefer a still-off-plane candidate (boolean split,
    #      as c040 found this beats a continuous off-plane blend). Among candidates in
    #      the same bucket, the 4 literal corner cells of the coarse grid additionally
    #      boost desirability by corner-proximity; all other empty cells simply take
    #      the strongest (first-encountered) candidate (c051's corner-restricted rule).
    #      Original off-plane quota picks stay protected from ever being swapped out
    #      as repair "victims". The swap victim is the weakest unprotected point
    #      inside a currently CROWDED coarse-grid cell (>=2 selected points there),
    #      falling back to the globally weakest unprotected point only when no cell
    #      is crowded (c051's rule, so a scarce swap removes genuine redundancy).
    #      Empty cells are processed off-availability-first (c054/c088): cells that
    #      still have an available off-plane filler candidate are repaired before
    #      any cell that only has an any-candidate filler, with corner-distance used
    #      purely as the tie-breaker within each group. ----
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

        CORNER_CELLS = {0, GX - 1, (GY - 1) * GX, GX * GY - 1}

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
            if c in CORNER_CELLS:
                desir = s[orig] * (1.0 + BETA_CORNER * _corner_proximity(u[orig], v[orig]))
            else:
                desir = s[orig]

            cur_any = best_any_in_cell.get(c)
            if cur_any is None or desir > cur_any[1]:
                best_any_in_cell[c] = (orig, desir)

            if soft_thresh is not None and dist_full[pos] > soft_thresh:
                cur_off = best_off_in_cell.get(c)
                if cur_off is None or desir > cur_off[1]:
                    best_off_in_cell[c] = (orig, desir)

        empty_cells = [c for c in range(GX * GY) if c not in sel_cells and c in best_any_in_cell]
        # off-availability first, corner-distance (farthest from image center) as
        # the tie-breaker within each group
        empty_cells.sort(
            key=lambda c: (
                0 if c in best_off_in_cell else 1,
                -((cell_center(c)[0] - cx_img) ** 2 + (cell_center(c)[1] - cy_img) ** 2),
            )
        )

        protected = set(off_quota_picks)
        swaps = 0
        sel_list = list(sel_arr)
        for c in empty_cells:
            if swaps >= MAX_SWAPS:
                break
            candidate = best_off_in_cell[c][0] if c in best_off_in_cell else best_any_in_cell[c][0]

            # ---- choose victim: weakest unprotected point inside a crowded cell ----
            cell_members = defaultdict(list)
            for p, idx_sel in enumerate(sel_list):
                if idx_sel in protected:
                    continue
                cidx = cell_of(u[idx_sel], v[idx_sel])
                cell_members[cidx].append(p)

            crowded_positions = [p for plist in cell_members.values() if len(plist) >= 2 for p in plist]

            victim_pos = None
            if crowded_positions:
                scores_crowd = np.array([s[sel_list[p]] for p in crowded_positions])
                victim_pos = int(crowded_positions[int(np.argmin(scores_crowd))])
            else:
                scores = np.array([s[i] for i in sel_list])
                order_by_weak = np.argsort(scores)  # ascending score
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
