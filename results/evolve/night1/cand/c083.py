# idea: replace the parent's coarse 4x3 grid-cell coverage repair with a continuous minimum-pixel-distance (blue-noise style) crowding repair, since two adjacent grid cells can each report "occupied" while the actual selected points sit right next to each other across the shared cell border, so explicit pairwise-distance thinning-and-refill should spread the 100 points more evenly across the image than grid occupancy alone.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5            # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10         # small, conservative off-plane reserve
MIN_THRESH = 0.04      # metres, floor on robust off-plane distance threshold
MAD_K = 2.5            # robust-scale multiplier on median-absolute-deviation
SOFT_OFF_FRAC = 0.5    # fraction of the robust threshold used as a "soft off-plane" cutoff in repair
BETA_CORNER = 0.3      # strength of corner-proximity bonus, used only to break ties among valid repair candidates
MAX_CORNER_DIST = (320.0 ** 2 + 240.0 ** 2) ** 0.5
MIN_DIST_FRAC = 0.5    # target minimum pixel spacing, as a fraction of the "uniform" spacing sqrt(area/K)
MAX_SWAPS = 20         # bounded crowding-repair budget


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


def _corner_proximity_arr(ui, vi):
    # vectorised distance to the nearest of the 4 image corners, normalised so corners -> 1, centre -> 0
    d0 = np.hypot(ui, vi)
    d1 = np.hypot(640.0 - ui, vi)
    d2 = np.hypot(ui, 480.0 - vi)
    d3 = np.hypot(640.0 - ui, 480.0 - vi)
    dmin = np.minimum(np.minimum(d0, d1), np.minimum(d2, d3))
    return 1.0 - np.minimum(1.0, dmin / MAX_CORNER_DIST)


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
    off_quota_picks = set()   # protected: never used as a repair "victim"

    dist_full = np.full(len(ok), -1.0, dtype=float)
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
            dist_full = dist
            med = float(np.median(dist))
            mad = float(np.median(np.abs(dist - med))) * 1.4826 + 1e-9
            thresh = max(MIN_THRESH, med + MAD_K * mad)
            soft_thresh = SOFT_OFF_FRAC * thresh
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

    # off-plane flag per original index (soft threshold), used only to prefer
    # off-plane candidates when a crowded point needs replacing
    off_flag_full = np.zeros(n, dtype=bool)
    if soft_thresh is not None:
        off_flag_full[ok] = dist_full > soft_thresh

    # ---- continuous minimum-pixel-distance crowding repair.
    #      Grid-cell occupancy (as used previously) can be fooled by two points
    #      that sit on opposite sides of a shared cell border: each cell looks
    #      "occupied" even though the two points are almost on top of each other
    #      in the image and contribute little extra baseline/coverage. Here we
    #      instead measure actual pixel-space crowding directly: repeatedly find
    #      the weakest, unprotected, currently-selected point that has another
    #      selected point closer than a target spacing MIN_DIST (sized for a
    #      roughly uniform spread of m points over the 640x480 image), and swap
    #      it for the best available unselected candidate that is itself at
    #      least MIN_DIST from every *other* selected point (off-plane and
    #      corner-proximity used only to break ties among valid candidates; if
    #      no candidate satisfies the spacing constraint, fall back to the one
    #      with the largest achievable minimum distance). Bounded by MAX_SWAPS. ----
    area = 640.0 * 480.0
    MIN_DIST = MIN_DIST_FRAC * np.sqrt(area / max(1, m))

    if m >= 2 and len(sel) < n:
        protected = set(off_quota_picks)
        sel_list = list(sel)
        taken_set = set(sel_list)

        for _ in range(MAX_SWAPS):
            if len(taken_set) >= n:
                break
            mm = len(sel_list)
            if mm < 2:
                break
            pos_sel = np.stack([u[sel_list], v[sel_list]], axis=1)
            diff = pos_sel[:, None, :] - pos_sel[None, :, :]
            dmat = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dmat, np.inf)
            mind_sel = dmat.min(axis=1)

            crowd_mask = mind_sel < MIN_DIST
            victim_pos = None
            victim_score = None
            for p in range(mm):
                if not crowd_mask[p]:
                    continue
                orig = sel_list[p]
                if orig in protected:
                    continue
                sc = s[orig]
                if victim_pos is None or sc < victim_score:
                    victim_pos = p
                    victim_score = sc
            if victim_pos is None:
                break

            victim = sel_list[victim_pos]
            other_pos = np.delete(pos_sel, victim_pos, axis=0)

            unsel = np.array([i for i in range(n) if i not in taken_set], dtype=int)
            if len(unsel) == 0:
                break
            cu = u[unsel]
            cv = v[unsel]

            if other_pos.shape[0] > 0:
                diff2 = np.stack([cu, cv], axis=1)[:, None, :] - other_pos[None, :, :]
                dcand = np.sqrt(np.sum(diff2 * diff2, axis=2))
                mind_cand = dcand.min(axis=1)
            else:
                mind_cand = np.full(len(unsel), np.inf)

            valid = mind_cand >= MIN_DIST
            if np.any(valid):
                valid_pos = np.flatnonzero(valid)
                off_valid = valid_pos[off_flag_full[unsel[valid_pos]]]
                pool = off_valid if len(off_valid) > 0 else valid_pos
                desir = s[unsel[pool]] * (1.0 + BETA_CORNER * _corner_proximity_arr(cu[pool], cv[pool]))
                best = pool[int(np.argmax(desir))]
                cand_idx = int(unsel[best])
            else:
                best = int(np.argmax(mind_cand))
                cand_idx = int(unsel[best])

            sel_list[victim_pos] = cand_idx
            taken_set.discard(victim)
            taken_set.add(cand_idx)
            protected.add(cand_idx)

        sel = sel_list

    return np.array(sel[:m], dtype=int)
