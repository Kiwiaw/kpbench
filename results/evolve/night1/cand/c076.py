# idea: fuse the two separately-proven signals from different lineages that have never been combined - keep this parent's per-frame adaptive cube size (so the round-robin spread auto-tunes to each frame's actual 3D extent instead of a fixed 0.5 m) but drive the fill with the simpler proven combo from the other lineage (small off-plane quota picked first, then a bounded image-coverage repair that prefers off-plane fill candidates and evicts the globally weakest unprotected point), so geometry-adaptive spread and the proven off-plane/coverage repair reinforce each other instead of competing.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

TARGET_CELLS_PER_PICK = 3.0   # aim for ~3x as many spatial bins as remaining picks
MIN_AXIS_EXTENT = 0.3         # metres; floors degenerate (near-planar / tiny) extents
MIN_CUBE = 0.25
MAX_CUBE = 1.0

OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020/c027)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = 6         # tightly bounded repair budget


def _adaptive_cube(P, target_k):
    if len(P) >= 2:
        bbox = np.maximum(P.max(axis=0) - P.min(axis=0), MIN_AXIS_EXTENT)
    else:
        bbox = np.full(3, MIN_AXIS_EXTENT)
    vol = float(np.prod(bbox))
    target_cells = max(1.0, TARGET_CELLS_PER_PICK * max(1, target_k))
    cube = (vol / target_cells) ** (1.0 / 3.0)
    return min(max(cube, MIN_CUBE), MAX_CUBE)


def _cube_round_robin(idx_order, P, taken, k, cube):
    # one point per adaptively-sized cube per round; idx_order/P rows are score-descending
    if k <= 0:
        return []
    key = np.floor(P / cube).astype(np.int64)
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
    dist_map = np.zeros(n)          # robust off-plane distance per original index

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
            soft_thresh = thresh
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

    # ---- fill the remainder with an adaptively-sized cube round-robin ----
    remaining = m - len(sel)
    if remaining > 0:
        cube = _adaptive_cube(P, remaining)
        rr = _cube_round_robin(ok, P, taken, remaining, cube)
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
    #      preferring an off-plane candidate (else plain strongest) per empty cell,
    #      evicting the globally weakest still-unprotected selected point. ----
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
        best_any_in_cell = {}   # cell -> (orig_index, plain score)
        best_off_in_cell = {}   # cell -> (orig_index, score), off-plane only
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            c = cell_of(u[orig], v[orig])

            cur_any = best_any_in_cell.get(c)
            if cur_any is None or s[orig] > cur_any[1]:
                best_any_in_cell[c] = (orig, s[orig])

            if soft_thresh is not None and dist_map[orig] > soft_thresh:
                cur_off = best_off_in_cell.get(c)
                if cur_off is None or s[orig] > cur_off[1]:
                    best_off_in_cell[c] = (orig, s[orig])

        empty_cells = [c for c in range(GX * GY)
                       if c not in sel_cells and (c in best_off_in_cell or c in best_any_in_cell)]
        # farthest-from-centre cells processed first so the bounded swap budget
        # preferentially restores peripheral / corner coverage
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
