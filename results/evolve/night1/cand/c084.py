# idea: in the parent's upfront guaranteed-coverage seeding, choose each grid cell's seed (both off-plane-preferred and any-candidate buckets) by corner-proximity-weighted score instead of first-encountered-strongest, so every one of the (always fully covered) cells also pulls toward true image corners without needing any swap/eviction.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020/c027)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for guaranteed-coverage seeding (640x480 -> 160x160 cells)
BETA = 0.3            # strength of corner-proximity bonus when choosing a cell's seed candidate
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

    # dist[pos] (pos indexes ok/P) = robust off-plane distance of that candidate
    dist = np.full(len(ok), -1.0, dtype=float)
    soft_thresh = None
    off_local = np.array([], dtype=int)   # positions (into ok/P) that are robustly off-plane, score-descending

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
            d = np.abs(Pc @ normal)
            dist = d
            med = float(np.median(d))
            mad = float(np.median(np.abs(d - med))) * 1.4826 + 1e-9
            thresh = max(MIN_THRESH, med + MAD_K * mad)
            soft_thresh = thresh   # use the same robust threshold for both seeding-preference and quota
            off_local = np.flatnonzero(d > thresh)   # already score-descending

    # ---- guaranteed-coverage seeding: pick one candidate per coarse grid cell,
    #      off-plane-preferred, else plain strongest, BEFORE anything else spends
    #      budget. Unlike the parent's first-encountered-strongest rule, within
    #      each cell/bucket we now compare ALL candidates in that cell by a
    #      corner-proximity-weighted desirability score, so the single seed that
    #      guarantees that cell's coverage is also the one that pulls hardest
    #      toward true image corners (known to widen the PnP baseline), rather
    #      than whichever strongest point happens to appear first in score order.
    #      This still guarantees full coverage of all (up to GX*GY) cells that
    #      have any candidate at all, with no swap/eviction ever needed. ----
    cw = 640.0 / GX
    ch = 480.0 / GY

    def cell_of(ui, vi):
        gx = int(min(GX - 1, max(0, np.floor(ui / cw))))
        gy = int(min(GY - 1, max(0, np.floor(vi / ch))))
        return gy * GX + gx

    best_any_in_cell = {}   # cell -> (orig_index, desirability)
    best_off_in_cell = {}   # cell -> (orig_index, desirability)
    for pos in range(len(ok)):
        orig = int(ok[pos])
        c = cell_of(u[orig], v[orig])
        desir = s[orig] * (1.0 + BETA * _corner_proximity(u[orig], v[orig]))

        cur_any = best_any_in_cell.get(c)
        if cur_any is None or desir > cur_any[1]:
            best_any_in_cell[c] = (orig, desir)

        if soft_thresh is not None and dist[pos] > soft_thresh:
            cur_off = best_off_in_cell.get(c)
            if cur_off is None or desir > cur_off[1]:
                best_off_in_cell[c] = (orig, desir)

    sel = []
    taken = set()
    for c in range(GX * GY):
        if len(sel) >= m:
            break
        chosen_tuple = best_off_in_cell.get(c, best_any_in_cell.get(c))
        chosen = chosen_tuple[0] if chosen_tuple is not None else None
        if chosen is not None and chosen not in taken:
            sel.append(chosen)
            taken.add(chosen)

    off_set = set(int(ok[p]) for p in off_local)

    # ---- off-plane quota: top up to OFF_QUOTA total off-plane points among those
    #      selected so far (seeds already contribute), spending only the extra
    #      budget needed, strongest-first. ----
    off_count = sum(1 for o in sel if o in off_set)
    if off_count < OFF_QUOTA:
        for p in off_local:
            if off_count >= OFF_QUOTA or len(sel) >= m:
                break
            orig = int(ok[p])
            if orig in taken:
                continue
            sel.append(orig)
            taken.add(orig)
            off_count += 1

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

    return np.array(sel[:m], dtype=int)
