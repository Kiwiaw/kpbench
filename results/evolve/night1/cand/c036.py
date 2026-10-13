# idea: replace the parent's pure DPP-on-score-decay core with c027/c031's proven recipe (PCA off-plane quota, reinforced by an off-plane-preferring, corner-first bounded coverage repair) but keep the DPP greedy (instead of the discrete cube round-robin) as the diversity engine for the non-quota budget, running it "conditioned" on the quota points already chosen so DPP spreads the remaining picks around them rather than independently -- testing whether DPP's continuous 3D-distance diversity, combined with the two signals that reliably helped cube-based pickers, beats both the plain DPP parent (22.90%) and the cube-based incumbents.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

SIGMA, ALPHA = 0.5, 1.0
OFF_QUOTA = 10
MIN_THRESH = 0.04
MAD_K = 2.5
GX, GY = 4, 3
MAX_SWAPS = 6


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

    P = to3d(u[ok], v[ok], z[ok])   # (nk,3), rows in score-descending order
    nk = len(ok)

    sel = []
    taken = set()
    quota_local = []   # positions (into ok/P) reserved as off-plane quota, forced into DPP first

    # ---- PCA dominant-plane off-plane detection ----
    off_mask_local = np.zeros(nk, dtype=bool)
    if nk >= 6:
        centroid = P.mean(axis=0)
        Pc = P - centroid
        cov = Pc.T @ Pc
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]
        nn = np.linalg.norm(normal)
        if nn > 1e-9:
            normal = normal / nn
            dist = np.abs(Pc @ normal)
            med = float(np.median(dist))
            mad = float(np.median(np.abs(dist - med))) * 1.4826 + 1e-9
            thresh = max(MIN_THRESH, med + MAD_K * mad)
            off_mask_local = dist > thresh
            if np.any(off_mask_local):
                off_local = np.flatnonzero(off_mask_local)  # score-descending already
                quota = min(OFF_QUOTA, m, len(off_local))
                quota_local = list(off_local[:quota])

    # ---- DPP greedy, forced to select quota_local first (conditional diversity),
    #      then continued normally for the remaining budget ----
    q = np.exp(-ALPHA * np.arange(nk) / 100.0)
    d2 = q ** 2
    c = np.zeros((0, nk))
    dpp_order = []

    def dpp_select(j):
        nonlocal c, d2
        dj = np.sqrt(max(d2[j], 1e-12))
        Lj = q[j] * q * np.exp(-np.sum((P - P[j]) ** 2, axis=1) / (2 * SIGMA ** 2))
        prev = (c[:, j] @ c) if len(c) else 0.0
        e = (Lj - prev) / dj
        c = np.vstack([c, e])
        d2 = d2 - e ** 2
        d2[j] = -np.inf

    for j in quota_local:
        if d2[j] == -np.inf:
            continue
        dpp_select(int(j))
        dpp_order.append(int(j))

    budget = m - len(dpp_order)
    for _ in range(max(budget, 0)):
        j = int(np.argmax(d2))
        if d2[j] <= -1e11 or not np.isfinite(d2[j]) or d2[j] <= 1e-12:
            break
        dpp_select(j)
        dpp_order.append(j)

    for j in dpp_order:
        orig = int(ok[j])
        if orig not in taken:
            sel.append(orig)
            taken.add(orig)

    # ---- fallback for any shortfall (e.g. sky points) ----
    if len(sel) < m:
        for i in range(n):
            if len(sel) >= m:
                break
            if i not in taken:
                sel.append(i)
                taken.add(i)

    sel = sel[:m]

    # protected set = the off-plane quota picks (by original index), kept safe from eviction
    protected = {int(ok[j]) for j in quota_local if int(ok[j]) in taken}

    # ---- bounded, off-plane-reinforced, corner-first coverage repair ----
    if m >= 2 and len(sel) >= 2:
        cw = 640.0 / GX
        ch = 480.0 / GY
        cx_img, cy_img = 320.0, 240.0

        def cell_of(ui, vi):
            gx = int(min(GX - 1, max(0, np.floor(ui / cw))))
            gy = int(min(GY - 1, max(0, np.floor(vi / ch))))
            return gy * GX + gx

        def cell_center(cid):
            gx = cid % GX
            gy = cid // GX
            return (gx + 0.5) * cw, (gy + 0.5) * ch

        sel_arr = np.array(sel, dtype=int)
        sel_cells = {cell_of(u[i], v[i]) for i in sel_arr}

        ok_set_taken = set(sel_arr.tolist())
        # per cell: prefer an off-plane unselected candidate (strongest), else strongest overall
        best_in_cell = {}
        best_offplane_in_cell = {}
        for pos in range(nk):
            orig = int(ok[pos])
            if orig in ok_set_taken:
                continue
            cid = cell_of(u[orig], v[orig])
            if cid not in best_in_cell:
                best_in_cell[cid] = orig
            if off_mask_local[pos] and cid not in best_offplane_in_cell:
                best_offplane_in_cell[cid] = orig

        empty_cells = [cid for cid in range(GX * GY) if cid not in sel_cells and cid in best_in_cell]
        empty_cells.sort(
            key=lambda cid: -((cell_center(cid)[0] - cx_img) ** 2 + (cell_center(cid)[1] - cy_img) ** 2)
        )

        swaps = 0
        sel_list = list(sel_arr)
        for cid in empty_cells:
            if swaps >= MAX_SWAPS:
                break
            candidate = best_offplane_in_cell.get(cid, best_in_cell[cid])
            scores = np.array([s[i] for i in sel_list])
            order_by_weak = np.argsort(scores)
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
