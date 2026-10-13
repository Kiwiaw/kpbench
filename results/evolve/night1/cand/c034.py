# idea: in c023-style crowded-cell-victim coverage repair, raise the swap budget from the fixed 6 to GX*GY=12 (every coarse grid cell) since the crowded-cell victim rule already protects sparse regions from being thinned, so a larger budget should let the repair finish filling every empty cell instead of leaving half of them untouched by an arbitrary cap.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = GX * GY   # repair budget raised to cover every coarse grid cell


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

    sel = []
    taken = set()

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
            med = float(np.median(dist))
            mad = float(np.median(np.abs(dist - med))) * 1.4826 + 1e-9
            thresh = max(MIN_THRESH, med + MAD_K * mad)
            off_mask = dist > thresh
            if np.any(off_mask):
                off_local = np.flatnonzero(off_mask)   # already score-descending
                quota = min(OFF_QUOTA, m, len(off_local))
                for c in off_local[:quota]:
                    orig = int(ok[c])
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

    sel = sel[:m]

    # ---- bounded image-coverage repair: fill genuinely empty coarse grid cells ----
    if m >= 2 and len(sel) >= 2:
        cw = 640.0 / GX
        ch = 480.0 / GY

        def cell_of(ui, vi):
            gx = int(min(GX - 1, max(0, np.floor(ui / cw))))
            gy = int(min(GY - 1, max(0, np.floor(vi / ch))))
            return gy * GX + gx

        sel_list = list(sel)
        sel_cells = [cell_of(u[i], v[i]) for i in sel_list]
        cell_counts = {}
        for c in sel_cells:
            cell_counts[c] = cell_counts.get(c, 0) + 1

        # strongest unselected candidate (with valid depth) per cell, score order preserved
        ok_set_taken = set(sel_list)
        best_in_cell = {}
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            c = cell_of(u[orig], v[orig])
            if c not in best_in_cell:
                best_in_cell[c] = orig

        empty_cells = [c for c in range(GX * GY)
                        if cell_counts.get(c, 0) == 0 and c in best_in_cell]

        protected = set()
        swaps = 0
        for c in empty_cells:
            if swaps >= MAX_SWAPS:
                break
            candidate = best_in_cell[c]

            # victim: weakest unprotected point inside the most crowded cell (count > 1),
            # so we never rob a cell that already has just one covering point
            crowd_cells = [cc for cc, cnt in cell_counts.items() if cnt > 1]
            victim_pos = None
            if crowd_cells:
                target_cell = max(crowd_cells, key=lambda cc: cell_counts[cc])
                cand_positions = [p for p in range(len(sel_list))
                                   if sel_cells[p] == target_cell and sel_list[p] not in protected]
                if cand_positions:
                    cand_positions.sort(key=lambda p: s[sel_list[p]])
                    victim_pos = cand_positions[0]
            if victim_pos is None:
                scores = np.array([s[i] for i in sel_list])
                order_by_weak = np.argsort(scores)  # ascending score
                for p in order_by_weak:
                    if sel_list[p] not in protected:
                        victim_pos = int(p)
                        break
            if victim_pos is None:
                break

            victim = sel_list[victim_pos]
            old_cell = sel_cells[victim_pos]
            cell_counts[old_cell] -= 1
            sel_list[victim_pos] = candidate
            sel_cells[victim_pos] = c
            cell_counts[c] = cell_counts.get(c, 0) + 1
            protected.add(candidate)
            ok_set_taken.discard(victim)
            ok_set_taken.add(candidate)
            swaps += 1

        sel = sel_list

    return np.array(sel[:m], dtype=int)
