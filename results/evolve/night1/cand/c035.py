# idea: in c020's off-plane quota + bounded coverage repair, protect ALL selected points that are geometrically off-plane (not just the small quota subset) from being evicted during coverage repair, and prefer off-plane candidates when filling empty cells -- so every off-plane pick, whether it came from the quota or appeared naturally in the cube round-robin, reinforces wide image coverage instead of only the quota's first 10.
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

    sel = []
    taken = set()

    # offplane_by_orig[orig] = True if that candidate (any candidate with valid depth)
    # lies off the dominant plane; used both to seed the quota AND to protect any
    # off-plane point that later survives into the selection via the cube round-robin.
    offplane_by_orig = {}

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
            for pos in range(len(ok)):
                offplane_by_orig[int(ok[pos])] = bool(off_mask[pos])
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

    # ---- bounded image-coverage repair: fill genuinely empty coarse grid cells,
    #      preferring an off-plane candidate for the fill, and NEVER evicting a
    #      selected point that is itself off-plane (whether it arrived via the
    #      quota or simply survived the cube round-robin) -- so the off-plane
    #      signal and the coverage signal reinforce across the whole selection,
    #      not just the quota's first 10 picks ----
    if m >= 2 and len(sel) >= 2:
        cw = 640.0 / GX
        ch = 480.0 / GY

        def cell_of(ui, vi):
            gx = int(min(GX - 1, max(0, np.floor(ui / cw))))
            gy = int(min(GY - 1, max(0, np.floor(vi / ch))))
            return gy * GX + gx

        sel_list = list(sel)
        sel_cells = [cell_of(u[i], v[i]) for i in sel_list]
        sel_off = [offplane_by_orig.get(i, False) for i in sel_list]
        cell_counts = {}
        for c in sel_cells:
            cell_counts[c] = cell_counts.get(c, 0) + 1

        ok_set_taken = set(sel_list)

        # strongest unselected candidate per cell (score order preserved by iterating ok),
        # tracked separately for off-plane and any candidate so the repair can prefer
        # geometry-diverse points without ever leaving a cell unfilled
        best_off_in_cell = {}
        best_any_in_cell = {}
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            c = cell_of(u[orig], v[orig])
            if c not in best_any_in_cell:
                best_any_in_cell[c] = orig
            if offplane_by_orig.get(orig, False) and c not in best_off_in_cell:
                best_off_in_cell[c] = orig

        empty_cells = [c for c in range(GX * GY)
                       if cell_counts.get(c, 0) == 0 and c in best_any_in_cell]
        # corner-first ordering: process cells farthest from the image centre first,
        # so the tightly bounded swap budget preferentially restores corner coverage
        cx_img, cy_img = 320.0, 240.0

        def cell_center(c):
            gx = c % GX
            gy = c // GX
            return (gx + 0.5) * cw, (gy + 0.5) * ch

        empty_cells.sort(
            key=lambda c: -((cell_center(c)[0] - cx_img) ** 2 + (cell_center(c)[1] - cy_img) ** 2)
        )

        swaps = 0
        for c in empty_cells:
            if swaps >= MAX_SWAPS:
                break
            candidate = best_off_in_cell.get(c, best_any_in_cell[c])

            # victim search, in priority order:
            #  1) weakest ON-PLANE point inside the most crowded cell (count > 1)
            #  2) weakest ON-PLANE point anywhere in the selection (ignore crowding)
            #  3) weakest point overall (last resort, may evict an off-plane point)
            victim_pos = None

            crowd_cells = sorted(
                (cc for cc, cnt in cell_counts.items() if cnt > 1),
                key=lambda cc: -cell_counts[cc],
            )
            for target_cell in crowd_cells:
                cand_positions = [p for p in range(len(sel_list))
                                   if sel_cells[p] == target_cell and not sel_off[p]]
                if cand_positions:
                    cand_positions.sort(key=lambda p: s[sel_list[p]])
                    victim_pos = cand_positions[0]
                    break

            if victim_pos is None:
                on_plane_positions = [p for p in range(len(sel_list)) if not sel_off[p]]
                if on_plane_positions:
                    on_plane_positions.sort(key=lambda p: s[sel_list[p]])
                    victim_pos = on_plane_positions[0]

            if victim_pos is None:
                scores = np.array([s[i] for i in sel_list])
                victim_pos = int(np.argmin(scores))

            victim = sel_list[victim_pos]
            old_cell = sel_cells[victim_pos]
            cell_counts[old_cell] -= 1
            sel_list[victim_pos] = candidate
            sel_cells[victim_pos] = c
            sel_off[victim_pos] = offplane_by_orig.get(candidate, False)
            cell_counts[c] = cell_counts.get(c, 0) + 1
            ok_set_taken.discard(victim)
            ok_set_taken.add(candidate)
            swaps += 1

        sel = sel_list

    return np.array(sel[:m], dtype=int)
