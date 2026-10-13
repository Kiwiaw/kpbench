# idea: in the off-plane quota selection, round-robin the quota picks across the same coarse image grid used by the coverage repair (instead of taking the top-OFF_QUOTA strongest off-plane points as one score-sorted block), so the off-plane reinforcement itself spreads across the image and doesn't clump in one region, leaving the bounded repair pass free to fix genuine coverage gaps instead of correcting clumped off-plane picks.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = 6         # tightly bounded repair budget


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

    # cube key for every point with valid depth, used later to detect crowded 3D cubes
    cube_key = np.floor(P / CUBE).astype(np.int64)
    cube_of = {}
    for pos in range(len(ok)):
        cube_of[int(ok[pos])] = (int(cube_key[pos, 0]), int(cube_key[pos, 1]), int(cube_key[pos, 2]))

    # coarse image grid, shared by the off-plane quota spreader and the coverage repair
    cw = 640.0 / GX
    ch = 480.0 / GY

    def cell_of(ui, vi):
        gx = int(min(GX - 1, max(0, np.floor(ui / cw))))
        gy = int(min(GY - 1, max(0, np.floor(vi / ch))))
        return gy * GX + gx

    sel = []
    taken = set()
    protected = set()   # points that must never be evicted by the repair pass

    # ---- deterministic dominant-plane fit via PCA (smallest-eigenvalue normal) ----
    off_mask_full = np.zeros(len(ok), dtype=bool)
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
            off_mask_full = dist > thresh
            if np.any(off_mask_full):
                off_local = np.flatnonzero(off_mask_full)   # already score-descending

                # spread the off-plane quota picks across the coarse image grid
                # (one per cell per round, in score-descending order within each
                # cell) instead of taking the top-OFF_QUOTA strongest off-plane
                # points as a single score-sorted block, which can clump them all
                # into one region of the image if the off-plane structure is local.
                cell_groups = {}
                cell_order = []
                for c in off_local:
                    orig = int(ok[c])
                    cc = cell_of(u[orig], v[orig])
                    lst = cell_groups.get(cc)
                    if lst is None:
                        cell_groups[cc] = [orig]
                        cell_order.append(cc)
                    else:
                        lst.append(orig)

                quota = min(OFF_QUOTA, m, len(off_local))
                picked = 0
                r = 0
                while picked < quota:
                    added = False
                    for cc in cell_order:
                        g = cell_groups[cc]
                        if r < len(g):
                            orig = g[r]
                            if orig not in taken:
                                sel.append(orig)
                                taken.add(orig)
                                protected.add(orig)
                                picked += 1
                                added = True
                                if picked >= quota:
                                    break
                    if not added:
                        break
                    r += 1

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

    # ---- bounded image-coverage repair: fill genuinely empty coarse grid cells ----
    if m >= 2 and len(sel) >= 2:
        off_set = set(int(ok[p]) for p in np.flatnonzero(off_mask_full)) if off_mask_full.any() else set()

        sel_list = list(sel)
        sel_cells = [cell_of(u[i], v[i]) for i in sel_list]
        sel_cell_set = set(sel_cells)

        ok_set_taken = set(sel_list)
        # per cell: strongest unselected candidate, and strongest unselected OFF-plane candidate
        best_in_cell = {}
        best_off_in_cell = {}
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            c = cell_of(u[orig], v[orig])
            if c not in best_in_cell:
                best_in_cell[c] = orig
            if orig in off_set and c not in best_off_in_cell:
                best_off_in_cell[c] = orig

        empty_cells = [c for c in range(GX * GY)
                       if c not in sel_cell_set and c in best_in_cell]

        swaps = 0
        for c in empty_cells:
            if swaps >= MAX_SWAPS:
                break
            # reinforce: prefer an off-plane candidate to fill the gap if one exists here
            candidate = best_off_in_cell.get(c, best_in_cell[c])

            # victim: weakest unprotected point whose 3D cube already holds >1 selected point
            cube_counts = {}
            for i in sel_list:
                kk = cube_of.get(i)
                if kk is not None:
                    cube_counts[kk] = cube_counts.get(kk, 0) + 1

            victim_pos = None
            best_score = None
            for p, i in enumerate(sel_list):
                if i in protected:
                    continue
                kk = cube_of.get(i)
                if kk is not None and cube_counts.get(kk, 0) > 1:
                    if best_score is None or s[i] < best_score:
                        best_score = s[i]
                        victim_pos = p

            if victim_pos is None:
                # fall back to the globally weakest unprotected selected point
                best_score = None
                for p, i in enumerate(sel_list):
                    if i in protected:
                        continue
                    if best_score is None or s[i] < best_score:
                        best_score = s[i]
                        victim_pos = p

            if victim_pos is None:
                break

            victim = sel_list[victim_pos]
            sel_list[victim_pos] = candidate
            sel_cells[victim_pos] = c
            sel_cell_set = set(sel_cells)
            protected.add(candidate)
            ok_set_taken.discard(victim)
            ok_set_taken.add(candidate)
            swaps += 1

        sel = sel_list

    return np.array(sel[:m], dtype=int)
