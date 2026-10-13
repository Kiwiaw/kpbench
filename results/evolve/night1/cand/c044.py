# idea: on top of the parent's per-frame adaptive cube size (which alone underperforms the fixed-cube baseline), port the best-known recipe -- PCA off-plane quota + bounded corner-weighted coverage repair, with repair cells choosing the MOST off-plane candidate (not merely the first/strongest encountered) -- and additionally make swap-victim selection prefer removing on-plane (or depth-less) selected points before off-plane ones, so every repair swap reinforces the off-plane signal from both the fill side and the victim side while keeping the adaptive scale for varying near-planar scene extents.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

TARGET_CELLS_PER_PICK = 3.0   # aim for ~3x as many spatial bins as picks requested
MIN_AXIS_EXTENT = 0.3         # metres; floors degenerate (near-planar / tiny) extents
MIN_CUBE = 0.25
MAX_CUBE = 1.0

OFF_QUOTA = 10        # small, conservative off-plane reserve (proven in c011/c020/.../c042)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = 6         # tightly bounded repair budget


def _cube_round_robin(idx_order, P, taken, k, cube):
    # one point per cube-sized voxel per round; idx_order/P rows are in score-descending order
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

    # ---- parent's adaptive cube size, driven by this frame's actual point-cloud extent ----
    if len(P) >= 2:
        bbox = np.maximum(P.max(axis=0) - P.min(axis=0), MIN_AXIS_EXTENT)
    else:
        bbox = np.full(3, MIN_AXIS_EXTENT)
    vol = float(np.prod(bbox))
    target_cells = max(1.0, TARGET_CELLS_PER_PICK * m)
    cube = (vol / target_cells) ** (1.0 / 3.0)
    cube = min(max(cube, MIN_CUBE), MAX_CUBE)

    sel = []
    taken = set()
    off_quota_picks = set()   # protected: never used as a repair "victim"

    off_mask_ok = np.zeros(len(ok), dtype=bool)
    dist_ok = np.zeros(len(ok), dtype=float)

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
            dist_ok = dist
            med = float(np.median(dist))
            mad = float(np.median(np.abs(dist - med))) * 1.4826 + 1e-9
            thresh = max(MIN_THRESH, med + MAD_K * mad)
            off_mask_ok = dist > thresh
            if np.any(off_mask_ok):
                off_local = np.flatnonzero(off_mask_ok)   # already score-descending
                quota = min(OFF_QUOTA, m, len(off_local))
                for c in off_local[:quota]:
                    orig = int(ok[c])
                    if orig not in taken:
                        sel.append(orig)
                        taken.add(orig)
                        off_quota_picks.add(orig)

    # map original index -> off-plane boolean (missing => on-plane / unknown-depth, i.e. removable)
    off_by_orig = {}
    for pos in range(len(ok)):
        off_by_orig[int(ok[pos])] = bool(off_mask_ok[pos])

    # ---- fill the remainder with adaptive-cube round-robin ----
    remaining = m - len(sel)
    if remaining > 0:
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
    #      preferring an off-plane candidate for the fill (boolean rule, proven),
    #      spending the scarce swap budget first on cells that actually HAVE an
    #      off-plane candidate available, corner-distance as tie-breaker; within a
    #      cell, choose the MOST off-plane candidate (not merely the first/
    #      strongest-score one encountered) so the fill maximally reinforces the
    #      geometric signal; victims are chosen preferring on-plane (or depth-less)
    #      selected points first, so swaps reinforce off-plane from both sides ----
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
        best_off_in_cell = {}   # cell -> (orig, dist) picking max dist
        best_any_in_cell = {}   # cell -> orig, first encountered (= strongest score)
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            c = cell_of(u[orig], v[orig])
            if c not in best_any_in_cell:
                best_any_in_cell[c] = orig
            if off_mask_ok[pos]:
                cur = best_off_in_cell.get(c)
                if cur is None or dist_ok[pos] > cur[1]:
                    best_off_in_cell[c] = (orig, float(dist_ok[pos]))

        empty_cells = [c for c in range(GX * GY) if c not in sel_cells and c in best_any_in_cell]
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
            if c in best_off_in_cell:
                candidate = best_off_in_cell[c][0]
            else:
                candidate = best_any_in_cell[c]

            # victim: prefer weakest-score unprotected point that is on-plane / depth-less
            scores = np.array([s[i] for i in sel_list])
            order_by_weak = np.argsort(scores)  # ascending score
            victim_pos = None
            for p in order_by_weak:
                idx_ = sel_list[int(p)]
                if idx_ in protected:
                    continue
                if not off_by_orig.get(idx_, False):
                    victim_pos = int(p)
                    break
            if victim_pos is None:
                for p in order_by_weak:
                    idx_ = sel_list[int(p)]
                    if idx_ not in protected:
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
