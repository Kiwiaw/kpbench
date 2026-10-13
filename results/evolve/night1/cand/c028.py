# idea: align the off-plane quota and the bounded coverage repair on the exact same coarse image grid (not separate 3D-cube vs image-grid partitions): spread the PCA off-plane quota itself across that grid via round-robin, and permanently swap-protect its picks during repair, so the two proven signals actively build the same spatial coverage instead of merely coexisting and occasionally fighting over the same slots.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid shared by off-plane quota AND coverage repair
MAX_SWAPS = 6         # tightly bounded repair budget
CAND_POOL = 3         # per empty cell, pool of strongest unselected candidates to consider


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
    dist_map = np.zeros(n)          # robust off-plane distance per original index

    cw = 640.0 / GX
    ch = 480.0 / GY

    def cell_of(ui, vi):
        gx = int(min(GX - 1, max(0, np.floor(ui / cw))))
        gy = int(min(GY - 1, max(0, np.floor(vi / ch))))
        return gy * GX + gx

    sel = []
    taken = set()
    protected = set()   # picks that the repair step must never swap out

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
            off_mask = dist > thresh
            if np.any(off_mask):
                off_local = np.flatnonzero(off_mask)   # score-descending among ok
                off_orig = ok[off_local]
                quota = min(OFF_QUOTA, m, len(off_orig))
                if quota > 0:
                    # spread the off-plane quota across the SAME coarse image grid that the
                    # coverage repair later uses, via round-robin (strongest per cell first)
                    groups = {}
                    order = []
                    for idx in off_orig:
                        idx = int(idx)
                        c = cell_of(u[idx], v[idx])
                        lst = groups.get(c)
                        if lst is None:
                            groups[c] = [idx]
                            order.append(c)
                        else:
                            lst.append(idx)
                    picked = []
                    r = 0
                    while len(picked) < quota:
                        added = False
                        for c in order:
                            g = groups[c]
                            if r < len(g):
                                picked.append(g[r])
                                added = True
                                if len(picked) >= quota:
                                    break
                        if not added:
                            break
                        r += 1
                    sel += picked
                    taken.update(picked)
                    protected.update(picked)

    # ---- fill the remainder with seed_cube's exact 0.5 m cube round-robin ----
    remaining = m - len(sel)
    if remaining > 0:
        key = np.floor(P / CUBE).astype(np.int64)
        groups = {}
        order = []
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in taken:
                continue
            kk = (int(key[pos, 0]), int(key[pos, 1]), int(key[pos, 2]))
            lst = groups.get(kk)
            if lst is None:
                groups[kk] = [orig]
                order.append(kk)
            else:
                lst.append(orig)
        rr = []
        r = 0
        while len(rr) < remaining:
            added = False
            for kk in order:
                g = groups[kk]
                if r < len(g):
                    rr.append(g[r])
                    added = True
                    if len(rr) >= remaining:
                        break
            if not added:
                break
            r += 1
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

    # ---- bounded image-coverage repair, on the same grid as the off-plane quota ----
    if m >= 2 and len(sel) >= 2:
        sel_list = list(sel)
        sel_cells = [cell_of(u[i], v[i]) for i in sel_list]
        cell_counts = {}
        for c in sel_cells:
            cell_counts[c] = cell_counts.get(c, 0) + 1

        ok_taken = set(sel_list)
        pools = {}
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_taken:
                continue
            c = cell_of(u[orig], v[orig])
            lst = pools.setdefault(c, [])
            if len(lst) < CAND_POOL:
                lst.append(orig)

        best_in_cell = {}
        for c, pool in pools.items():
            best_idx = pool[0]
            best_d = dist_map[pool[0]]
            for i in pool[1:]:
                if dist_map[i] > best_d:
                    best_idx = i
                    best_d = dist_map[i]
            best_in_cell[c] = best_idx

        empty_cells = [c for c in range(GX * GY)
                       if cell_counts.get(c, 0) == 0 and c in best_in_cell]

        swaps = 0
        for c in empty_cells:
            if swaps >= MAX_SWAPS:
                break
            candidate = best_in_cell[c]

            # victim: weakest unprotected point inside the most crowded cell
            # (never touch off-plane-quota picks, which are permanently protected)
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
                order_by_weak = np.argsort([s[i] for i in sel_list])
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
            ok_taken.discard(victim)
            ok_taken.add(candidate)
            swaps += 1

        sel = sel_list

    return np.array(sel[:m], dtype=int)
