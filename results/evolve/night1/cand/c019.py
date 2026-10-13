# idea: keep seed_cube's proven 3D-cube round-robin exactly as the base selector, then apply one small (<=8-pick), targeted swap that replaces the weakest-ranked, most on-plane selections with the best available off-plane candidates (scored by a distance-to-PCA-plane / strength composite), so only the few "redundant-on-plane" tail picks are nudged off-plane instead of reserving a large fixed quota (which previously hurt) or reordering the whole ranking (which also hurt).
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5      # proven voxel size from seed_cube (metres)
MAX_SWAP = 8    # tiny, late-stage quota -- most of the proven ranking is left untouched
BETA = 0.3      # small strength tie-break weight when choosing off-plane replacements


def _cube_round_robin(ok, P):
    """Exact seed_cube style bucketing + round-robin, operating on positions into `ok`."""
    key = np.floor(P / CUBE).astype(np.int64)
    buckets = {}
    order = []
    for pos in range(len(ok)):
        k = (int(key[pos, 0]), int(key[pos, 1]), int(key[pos, 2]))
        lst = buckets.get(k)
        if lst is None:
            buckets[k] = [pos]
            order.append(k)
        else:
            lst.append(pos)

    rank_pos = []
    r = 0
    while True:
        added = False
        for k in order:
            g = buckets[k]
            if r < len(g):
                rank_pos.append(g[r])
                added = True
        if not added:
            break
        r += 1
    return rank_pos


def pick(u, v, s, z, K):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if len(ok) == 0:
        return np.arange(m, dtype=int)

    P = to3d(u[ok], v[ok], z[ok])  # (nk, 3) camera-frame metres, score-descending order preserved

    # ---- base selection: proven 3D-cube round-robin (seed_cube), unchanged ----
    rank_pos = _cube_round_robin(ok, P)
    full_rank = list(ok[np.array(rank_pos, dtype=int)])
    ok_set = set(int(i) for i in ok)
    leftover = [i for i in range(n) if i not in ok_set]
    full_rank += leftover
    sel_list = full_rank[:m]

    nk = len(ok)
    if nk < 3 or MAX_SWAP <= 0:
        return np.array(sel_list[:m], dtype=int)

    # ---- dominant plane via fast deterministic PCA/SVD (c011 style) ----
    centroid = P.mean(axis=0)
    Pc = P - centroid
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    normal = Vt[-1]
    dist_all = np.abs(Pc @ normal)  # perpendicular distance to dominant plane, per ok-position

    dist_map = {int(ok[i]): float(dist_all[i]) for i in range(nk)}

    # ---- composite "worth swapping in" metric: off-plane distance + mild strength tie-break ----
    max_d = float(dist_all.max()) if dist_all.max() > 0 else 1.0
    dist_norm = dist_all / max_d
    rank_norm = np.arange(nk, dtype=float) / max(nk - 1, 1)  # 0 = strongest score
    metric = dist_norm - BETA * rank_norm

    sel_set = set(sel_list)

    # candidates to remove: the weakest-ranked selected points that sit close to the plane
    sel_dists = [dist_map[idx] for idx in sel_list if idx in dist_map]
    if not sel_dists:
        return np.array(sel_list[:m], dtype=int)
    thresh = float(np.median(sel_dists))

    removal_candidates = []
    for idx in reversed(sel_list):
        if idx in dist_map and dist_map[idx] <= thresh:
            removal_candidates.append(idx)
        if len(removal_candidates) >= MAX_SWAP:
            break

    if not removal_candidates:
        return np.array(sel_list[:m], dtype=int)

    order_by_metric = np.argsort(-metric)
    add_candidates = []
    for i in order_by_metric:
        idx = int(ok[i])
        if idx in sel_set:
            continue
        if dist_all[i] <= thresh:
            continue
        add_candidates.append(idx)
        if len(add_candidates) >= len(removal_candidates):
            break

    for rem, add in zip(removal_candidates, add_candidates):
        pos = sel_list.index(rem)
        sel_list[pos] = add
        sel_set.discard(rem)
        sel_set.add(add)

    return np.array(sel_list[:m], dtype=int)
