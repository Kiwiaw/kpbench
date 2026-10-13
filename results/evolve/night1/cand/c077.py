# idea: replace the parent's bounded (MAX_SWAPS=6), victim-swap-based coverage repair with an unconditional "guarantee-first" coverage pass -- after the off-plane quota, immediately reserve one point (off-plane-preferred, else strongest) for every still-empty coarse grid cell before the cube round-robin fill, so every cell with any candidate gets covered deterministically and completely instead of only as many as a tight swap budget and victim-selection heuristic happen to restore.
import numpy as np
from collections import defaultdict
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020/c027/c042)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage guarantee (640x480 -> 160x160 cells)


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
    nk = len(ok)

    sel = []
    taken = set()

    off_mask = np.zeros(nk, dtype=bool)

    # ---- deterministic dominant-plane fit via PCA (smallest-eigenvalue normal) ----
    if nk >= 6:
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

    # ---- coarse-grid cell lookup ----
    cw = 640.0 / GX
    ch = 480.0 / GY

    def cell_of(ui, vi):
        gx = int(min(GX - 1, max(0, np.floor(ui / cw))))
        gy = int(min(GY - 1, max(0, np.floor(vi / ch))))
        return gy * GX + gx

    # per-cell lists of positions into ok (already score-descending since ok preserves order)
    cell_lists = defaultdict(list)
    for pos in range(nk):
        c = cell_of(u[ok[pos]], v[ok[pos]])
        cell_lists[c].append(pos)

    covered_cells = {cell_of(u[i], v[i]) for i in sel}

    # ---- unconditional coverage guarantee: every coarse cell that has any
    #      candidate at all gets exactly one reserved point before the cube
    #      round-robin fill runs, no swap budget and no victim-selection
    #      heuristic involved. Prefer a still-off-plane candidate in that
    #      cell (reinforcing the off-plane-helps-PnP signal even in filled
    #      cells); otherwise take the strongest untaken candidate in it. ----
    if m - len(sel) > 0:
        for c in range(GX * GY):
            if len(sel) >= m:
                break
            if c in covered_cells:
                continue
            plist = cell_lists.get(c)
            if not plist:
                continue
            cand_any = None
            cand_off = None
            for pos in plist:
                orig = int(ok[pos])
                if orig in taken:
                    continue
                if cand_any is None:
                    cand_any = orig
                if off_mask[pos]:
                    cand_off = orig
                    break
            chosen = cand_off if cand_off is not None else cand_any
            if chosen is not None:
                sel.append(chosen)
                taken.add(chosen)
                covered_cells.add(c)

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
