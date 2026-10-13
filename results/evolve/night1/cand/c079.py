# idea: in the parent's off-plane quota selection, replace plain "take the OFF_QUOTA strongest off-plane candidates" with a greedy minimum-pixel-separation selection among off-plane candidates (falling back to the strongest leftovers if spacing can't fill the quota), so the off-plane reserve itself is spread across the image instead of potentially clustering on one off-plane patch, since wide image coverage is known to help PnP.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020/c027)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = 6         # tightly bounded repair budget
SOFT_OFF_FRAC = 0.5   # fraction of the robust threshold used as a "soft off-plane" cutoff in repair
BETA = 0.3            # strength of corner-proximity bonus when choosing an in-cell repair candidate
MAX_CORNER_DIST = (320.0 ** 2 + 240.0 ** 2) ** 0.5   # farthest any pixel can be from its nearest corner (image centre)
MIN_SEP = 50.0        # minimum pixel separation enforced greedily among off-plane quota picks


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

                # ---- greedy minimum-pixel-separation selection among off-plane
                #      candidates: instead of blindly taking the OFF_QUOTA
                #      strongest off-plane points (which may all sit on the same
                #      off-plane object/patch), walk them in score order and only
                #      accept a candidate if it is at least MIN_SEP pixels from
                #      every already-accepted quota pick. If spacing leaves the
                #      quota under-filled, top it up with the strongest leftover
                #      off-plane candidates regardless of spacing, so the quota
                #      size itself is never reduced. ----
                quota = min(OFF_QUOTA, m, len(off_local))
                selected_local = []
                selected_uv = []
                if quota > 0:
                    for idx_local in off_local:
                        if len(selected_local) >= quota:
                            break
                        orig_c = int(ok[idx_local])
                        uu, vv = float(u[orig_c]), float(v[orig_c])
                        far_enough = True
                        for suu, svv in selected_uv:
                            if (uu - suu) ** 2 + (vv - svv) ** 2 < MIN_SEP ** 2:
                                far_enough = False
                                break
                        if far_enough:
                            selected_local.append(idx_local)
                            selected_uv.append((uu, vv))

                    if len(selected_local) < quota:
                        chosen_set = set(selected_local)
                        for idx_local in off_local:
                            if len(selected_local) >= quota:
                                break
                            if idx_local not in chosen_set:
                                selected_local.append(idx_local)
                                chosen_set.add(idx_local)

                    for c in selected_local:
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
    #      as c040 found this beats a fully continuous off-plane blend), but among
    #      candidates in the same bucket (off-plane-preferred, or the any-candidate
    #      fallback) pick the one whose score is boosted most by closeness to the
    #      nearest image corner, so the fill is actively pulled toward true corners
    #      (wide baseline helps PnP) rather than just toward whichever strongest
    #      point happens to land in that coarse cell. Original off-plane quota picks
    #      stay protected from ever being swapped out as repair "victims". ----
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
