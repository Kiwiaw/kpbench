# idea: extend c027's off-plane reinforcement beyond the quota+repair stages into the core cube round-robin fill itself, ordering each cube's candidate list off-plane-first (then by score) so every round-robin round also favors off-plane points, giving three mutually-reinforcing off-plane-aware stages instead of two.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair
MAX_SWAPS = 6         # tightly bounded repair budget


def _off_plane_mask(P):
    if len(P) < 6:
        return np.zeros(len(P), dtype=bool)
    centroid = P.mean(axis=0)
    Pc = P - centroid
    cov = Pc.T @ Pc
    evals, evecs = np.linalg.eigh(cov)   # ascending eigenvalues
    normal = evecs[:, 0]
    nn = np.linalg.norm(normal)
    if nn < 1e-9:
        return np.zeros(len(P), dtype=bool)
    normal = normal / nn
    dist = np.abs(Pc @ normal)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med))) * 1.4826 + 1e-9
    thresh = max(MIN_THRESH, med + MAD_K * mad)
    return dist > thresh


def _cube_round_robin_offplane_first(ok, P, off_mask, taken, k):
    # one point per CUBE-sized cube per round; within each cube, off-plane
    # candidates are listed before on-plane ones (each sub-group keeping its
    # original score-descending order), so the round-robin spread itself
    # reinforces the off-plane signal, not just the quota and the repair.
    if k <= 0:
        return []
    key = np.floor(P / CUBE).astype(np.int64)
    groups = {}
    order = []

    def ensure(kk):
        lst = groups.get(kk)
        if lst is None:
            lst = []
            groups[kk] = lst
            order.append(kk)
        return lst

    for pos in range(len(ok)):
        orig = int(ok[pos])
        if orig in taken or not off_mask[pos]:
            continue
        kk = (int(key[pos, 0]), int(key[pos, 1]), int(key[pos, 2]))
        ensure(kk).append(orig)

    for pos in range(len(ok)):
        orig = int(ok[pos])
        if orig in taken or off_mask[pos]:
            continue
        kk = (int(key[pos, 0]), int(key[pos, 1]), int(key[pos, 2]))
        ensure(kk).append(orig)

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
    off_mask = _off_plane_mask(P)

    sel = []
    taken = set()
    protected = set()

    # ---- small off-plane quota (strongest off-plane points first) ----
    if np.any(off_mask):
        off_local = np.flatnonzero(off_mask)   # score-descending among off-plane
        quota = min(OFF_QUOTA, m, len(off_local))
        for c in off_local[:quota]:
            orig = int(ok[c])
            if orig not in taken:
                sel.append(orig)
                taken.add(orig)
                protected.add(orig)

    # ---- fill remainder with off-plane-first cube round-robin ----
    remaining = m - len(sel)
    if remaining > 0:
        rr = _cube_round_robin_offplane_first(ok, P, off_mask, taken, remaining)
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

    # ---- bounded, off-plane-reinforced, corner-first coverage repair ----
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

        sel_list = list(sel)
        sel_cells = [cell_of(u[i], v[i]) for i in sel_list]
        cell_counts = {}
        for c in sel_cells:
            cell_counts[c] = cell_counts.get(c, 0) + 1

        ok_set_taken = set(sel_list)

        off_global = np.zeros(n, dtype=bool)
        off_global[ok[np.flatnonzero(off_mask)]] = True

        best_off_in_cell = {}
        best_any_in_cell = {}
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            c = cell_of(u[orig], v[orig])
            if c not in best_any_in_cell:
                best_any_in_cell[c] = orig
            if off_global[orig] and c not in best_off_in_cell:
                best_off_in_cell[c] = orig

        def fill_candidate(c):
            if c in best_off_in_cell:
                return best_off_in_cell[c]
            return best_any_in_cell.get(c)

        empty_cells = [c for c in range(GX * GY)
                        if cell_counts.get(c, 0) == 0 and fill_candidate(c) is not None]
        # corners-first ordering: largest PnP rotation lever arm gets the
        # tightly bounded swap budget first
        empty_cells.sort(
            key=lambda c: -((cell_center(c)[0] - cx_img) ** 2 + (cell_center(c)[1] - cy_img) ** 2)
        )

        swaps = 0
        for c in empty_cells:
            if swaps >= MAX_SWAPS:
                break
            candidate = fill_candidate(c)
            if candidate is None or candidate in ok_set_taken:
                continue

            # victim: weakest unprotected point inside the most crowded cell,
            # so emptying a cell never starves an already-sparse region
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
