# idea: replace the parent's coarse fixed-grid coverage repair (shown repeatedly to be insensitive to or hurt by grid-resolution changes: c056's 5x4 grid and c061's adaptive grid both underperformed) with a continuous image-space blue-noise relaxation -- each bounded swap evicts the weaker-scored point of the current tightest (closest-pair) crowd and replaces it with the unselected candidate that maximizes spacing from the rest of the selection, with a small bonus for off-plane candidates, instead of quantizing coverage into a fixed pixel grid whose cell boundaries can hide or misjudge true crowding.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
MAX_SWAPS = 6         # tightly bounded repair budget (same budget as prior siblings)
OFF_BONUS = 1.25      # multiplicative preference for off-plane fillers when ranking spacing


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
    off_quota_picks = set()   # protected: never used as a repair victim

    offplane_full = np.zeros(n, dtype=bool)  # lookup, by original index, over ALL candidates with valid depth

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
            offplane_full[ok[off_mask]] = True
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

    # ---- bounded continuous blue-noise coverage repair ----
    # Instead of a fixed pixel grid (shown fragile to resolution choice in prior
    # siblings), directly operate on image-space spacing: repeatedly find the
    # current closest pair among selected points (the tightest crowd), evict its
    # weaker-scored unprotected member, and replace it with the unselected,
    # valid-depth candidate that maximizes the minimum distance to the rest of
    # the selection (i.e. fills the single biggest remaining gap), with a small
    # multiplicative bonus for off-plane candidates so the two proven signals
    # (wide coverage, off-plane geometry) reinforce instead of competing.
    if m >= 2 and len(sel) >= 2:
        sel_list = list(sel)
        protected = set(off_quota_picks)

        # candidate pool for filling: valid-depth points not currently selected
        ok_list = ok.tolist()
        taken_set = set(sel_list)
        cand_pool = [int(o) for o in ok_list if o not in taken_set]
        cand_pool_arr = np.array(cand_pool, dtype=int)

        for _ in range(MAX_SWAPS):
            if len(cand_pool_arr) == 0 or len(sel_list) < 2:
                break

            sel_arr = np.array(sel_list, dtype=int)
            su = u[sel_arr]
            sv = v[sel_arr]
            dx = su[:, None] - su[None, :]
            dy = sv[:, None] - sv[None, :]
            D = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(D, np.inf)
            minval = float(D.min())
            if not np.isfinite(minval):
                break
            i_idx, j_idx = np.unravel_index(int(np.argmin(D)), D.shape)

            # choose the weaker-score, unprotected member of the closest pair as victim
            cands_pair = [int(i_idx), int(j_idx)]
            cands_pair.sort(key=lambda p: s[sel_list[p]])
            victim_pos = None
            for p in cands_pair:
                if sel_list[p] not in protected:
                    victim_pos = p
                    break
            if victim_pos is None:
                break

            # compute spacing of every pool candidate to the selection MINUS the victim
            keep_mask = np.ones(len(sel_list), dtype=bool)
            keep_mask[victim_pos] = False
            ku = su[keep_mask]
            kv = sv[keep_mask]

            cu = u[cand_pool_arr]
            cv = v[cand_pool_arr]
            cdx = cu[:, None] - ku[None, :]
            cdy = cv[:, None] - kv[None, :]
            cD = np.sqrt(cdx * cdx + cdy * cdy)
            if cD.shape[1] == 0:
                spacing = np.full(len(cand_pool_arr), np.inf)
            else:
                spacing = cD.min(axis=1)

            off_bonus_mask = offplane_full[cand_pool_arr]
            ranked = spacing * np.where(off_bonus_mask, OFF_BONUS, 1.0)
            best_pool_pos = int(np.argmax(ranked))
            best_spacing = float(spacing[best_pool_pos])

            if best_spacing <= minval:
                # no genuine spread improvement available; stop spending budget
                break

            candidate = int(cand_pool_arr[best_pool_pos])
            victim = sel_list[victim_pos]

            sel_list[victim_pos] = candidate
            protected.add(candidate)

            # update candidate pool: remove chosen candidate, add back the victim
            cand_pool_arr = np.delete(cand_pool_arr, best_pool_pos)
            cand_pool_arr = np.append(cand_pool_arr, victim)

        sel = sel_list

    return np.array(sel[:m], dtype=int)
