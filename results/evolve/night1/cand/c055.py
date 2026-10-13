# idea: in the parent's crowded-cell-victim, CAND_POOL-pooled-fill coverage repair (which the c020->c027->c042 sibling lineage shows underperforms by ~1pt fail vs. the simpler "weakest-unprotected victim, off-plane-preferred fill" combo), drop the crowd-cell victim search and the pooled candidate scan, and instead reinforce the two already-separately-proven signals so they only combine where they should: apply the corner-proximity score bonus exclusively to off-plane candidates inside the 4 literal corner cells (falling back to the plain strongest candidate, with NO bonus, when a corner cell has no off-plane option), and pick the swap victim as the globally weakest still-unprotected point.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020/c027)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = 6         # tightly bounded repair budget
SOFT_OFF_FRAC = 0.5   # fraction of the robust threshold used as a "soft off-plane" cutoff in repair
BETA_CORNER = 0.5     # strength of corner-proximity bonus, applied only to off-plane candidates in the 4 true corners
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
    dist_map = np.zeros(n)          # robust off-plane distance per original index (0 if unknown/on-plane)

    sel = []
    taken = set()
    off_quota_picks = set()   # protected: never used as a repair "victim"
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
            dist_map[ok] = dist
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
    #      For each empty cell, prefer a still-off-plane candidate; only within the
    #      4 literal corner cells does an off-plane candidate also get a corner-
    #      proximity score bonus (reinforcing the two proven signals exactly where
    #      both apply). If a corner cell has no off-plane candidate at all, fall
    #      back to the plain strongest candidate there with NO bonus, to avoid
    #      letting a weak on-plane point win purely on corner distance. Non-corner
    #      empty cells simply prefer off-plane else plain strongest, as in c027.
    #      The swap victim is always the globally weakest still-unprotected point;
    #      original off-plane quota picks and any newly added repair points are
    #      protected from ever being evicted again. ----
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
        best_any_in_cell = {}   # cell -> (orig_index, plain score) -- no bonus, pure fallback
        best_off_in_cell = {}   # cell -> (orig_index, desirability) -- off-plane, bonus only at corners
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            c = cell_of(u[orig], v[orig])

            cur_any = best_any_in_cell.get(c)
            if cur_any is None or s[orig] > cur_any[1]:
                best_any_in_cell[c] = (orig, s[orig])

            if soft_thresh is not None and dist_map[orig] > soft_thresh:
                if c in CORNER_CELLS:
                    desir = s[orig] * (1.0 + BETA_CORNER * _corner_proximity(u[orig], v[orig]))
                else:
                    desir = s[orig]
                cur_off = best_off_in_cell.get(c)
                if cur_off is None or desir > cur_off[1]:
                    best_off_in_cell[c] = (orig, desir)

        empty_cells = [c for c in range(GX * GY)
                       if c not in sel_cells and (c in best_off_in_cell or c in best_any_in_cell)]
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
            if c in best_off_in_cell:
                candidate = best_off_in_cell[c][0]
            else:
                candidate = best_any_in_cell[c][0]

            # victim: globally weakest currently-selected, unprotected point
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
