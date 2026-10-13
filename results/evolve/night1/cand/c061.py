# idea: replace the fixed coarse pixel-grid coverage repair (shown to be sensitive to bin resolution: finer grids hurt) with a continuous greedy repulsion rule directly in bearing-vector (viewing-angle) space, since PnP rotation conditioning depends on angular ray spread, not pixel-grid occupancy; each swap removes the weaker member of the most redundant (highest-cosine-similarity) selected pair and inserts the available candidate ray farthest (in angle) from the whole current selection, with a boolean off-plane bonus, accepting the swap only if it genuinely increases the minimum pairwise angular separation.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020/c027)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
MAX_SWAPS = 6         # tightly bounded repair budget
BETA_OFF = 0.5        # boolean off-plane bonus multiplier applied to candidate gap score
FX = FY = 320.0
CX, CY = 320.0, 240.0


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


def _bearing(u, v):
    # unit viewing-direction (bearing) vector for each pixel; independent of depth
    x = (u - CX) / FX
    y = (v - CY) / FY
    d = np.stack([x, y, np.ones_like(x)], axis=1)
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm = np.where(norm < 1e-12, 1e-12, norm)
    return d / norm


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

    dist_arr = np.full(n, -1.0, dtype=float)   # robust off-plane distance per original index
    thresh = None

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
            dist_arr[ok] = dist
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

    # ---- continuous bearing-space farthest-point coverage repair ----
    # Rationale: a fixed coarse pixel grid (used by earlier repairs) is sensitive to
    # bin resolution -- a finer grid measurably hurt in an earlier experiment -- and
    # it ignores the nonlinear relationship between pixel position and viewing angle.
    # PnP conditioning (especially for rotation) really depends on the angular spread
    # of the bearing rays actually used, not on pixel-grid occupancy. So, instead of
    # detecting "empty grid cells" and filling them, we run a tightly bounded greedy
    # repulsion loop directly in bearing-vector space: at each step find the most
    # redundant (highest-cosine-similarity) pair inside the current selection, evict
    # its weaker UNPROTECTED member, and bring in whichever available candidate ray is
    # farthest (in angle) from the entire current selection -- with a boolean off-plane
    # bonus (proven better than a continuous blend) -- accepting the swap only if it
    # genuinely raises the worst-case (most-redundant) pairwise angular separation.
    # Off-plane quota picks stay protected from ever being evicted as victims.
    if m >= 2 and len(sel) >= 2 and len(sel) < n:
        B_all = _bearing(u, v)   # (n, 3) unit bearing vectors, depth-independent
        sel_list = list(sel)
        protected = set(off_quota_picks)
        inf_thresh = thresh if thresh is not None else np.inf

        for _ in range(MAX_SWAPS):
            ns = len(sel_list)
            sel_arr = np.array(sel_list, dtype=int)
            B_sel = B_all[sel_arr]                      # (ns, 3)
            sim_sel = B_sel @ B_sel.T                    # (ns, ns) cosine similarities
            np.fill_diagonal(sim_sel, -2.0)

            cur_worst_sim = float(sim_sel.max())

            # victim: weaker UNPROTECTED member of the most redundant eligible pair
            flat_order = np.argsort(sim_sel, axis=None)[::-1]
            victim_pos = None
            for fidx in flat_order:
                i, j = divmod(int(fidx), ns)
                if sim_sel[i, j] < -1.5:
                    break
                cand_i, cand_j = sel_list[i], sel_list[j]
                eligible = []
                if cand_i not in protected:
                    eligible.append(i)
                if cand_j not in protected:
                    eligible.append(j)
                if not eligible:
                    continue
                if len(eligible) == 2:
                    victim_pos = i if s[cand_i] <= s[cand_j] else j
                else:
                    victim_pos = eligible[0]
                break
            if victim_pos is None:
                break

            taken_set = set(sel_list)
            pool = np.array([idx for idx in range(n) if idx not in taken_set], dtype=int)
            if len(pool) == 0:
                break

            B_pool = B_all[pool]
            sims_pool = B_pool @ B_sel.T                 # (npool, ns)
            nn_sim = sims_pool.max(axis=1)                # nearest-neighbour similarity to current selection
            gap = 1.0 - nn_sim
            off_bonus = np.where(dist_arr[pool] > inf_thresh, 1.0 + BETA_OFF, 1.0)
            desir = gap * off_bonus
            best_pos = int(np.argmax(desir))
            candidate = int(pool[best_pos])
            candidate_nn_sim = float(nn_sim[best_pos])

            # only accept the swap if it genuinely improves worst-case angular separation
            if candidate_nn_sim >= cur_worst_sim:
                break

            sel_list[victim_pos] = candidate
            protected.add(candidate)

        sel = sel_list

    return np.array(sel[:m], dtype=int)
