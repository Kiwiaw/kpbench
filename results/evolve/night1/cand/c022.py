# idea: within c020's small PCA off-plane quota, replace the plain strongest-first slice with a cube-bucketed round-robin (reusing the same 0.5 m 3D grouping key as the main selector) so the <=10 off-plane picks are spread across distinct spatial buckets instead of clustering in one corner, giving more pose-stabilising viewing diversity without touching the proven cube round-robin core or the bounded coverage repair.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for coverage repair (640x480 -> 160x160 cells)
MAX_SWAPS = 6         # tightly bounded repair budget


def _bucket_round_robin(orig_indices, keys, taken, k):
    # one point per spatial bucket (3D cube key) per round, preserving score order
    # within each bucket (orig_indices/keys are already in score-descending order).
    if k <= 0:
        return []
    groups = {}
    order = []
    for pos in range(len(orig_indices)):
        orig = int(orig_indices[pos])
        if orig in taken:
            continue
        kk = (int(keys[pos, 0]), int(keys[pos, 1]), int(keys[pos, 2]))
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
    key = np.floor(P / CUBE).astype(np.int64)

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
                if quota > 0:
                    off_orig = ok[off_local]
                    off_key = key[off_local]
                    # spread the small off-plane quota across distinct 3D cube buckets
                    picked = _bucket_round_robin(off_orig, off_key, taken, quota)
                    sel += picked
                    taken.update(picked)

    # ---- fill the remainder with seed_cube's exact 0.5 m cube round-robin ----
    remaining = m - len(sel)
    if remaining > 0:
        rr = _bucket_round_robin(ok, key, taken, remaining)
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

        sel_arr = np.array(sel, dtype=int)
        sel_cells = {cell_of(u[i], v[i]) for i in sel_arr}

        # strongest unselected candidate (with valid depth) per cell, score order preserved
        ok_set_taken = set(sel_arr.tolist())
        best_in_cell = {}
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            c = cell_of(u[orig], v[orig])
            if c not in best_in_cell:
                best_in_cell[c] = orig

        empty_cells = [c for c in range(GX * GY) if c not in sel_cells and c in best_in_cell]

        protected = set()
        swaps = 0
        sel_list = list(sel_arr)
        for c in empty_cells:
            if swaps >= MAX_SWAPS:
                break
            candidate = best_in_cell[c]
            # find weakest (lowest score) currently selected, unprotected point
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
