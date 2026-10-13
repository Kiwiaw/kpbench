# idea: fix a hidden truncation bug in seed_cube's/c011's cube round-robin: when the number of occupied 0.5 m cubes exceeds K the old code only ever visited the first K cubes in score-first-occurrence order, silently ignoring whole regions of 3D/image space whose best point happened to rank lower -- replace that visitation order with a deterministic Morton (Z-order) spatial sort and, whenever cubes outnumber the budget, take an evenly-spaced stride sample along that curve so the fixed pick budget is spread across the *whole* occupied volume instead of being hijacked by score-dense regions, directly targeting the documented "wide image coverage / 3D spread helps pose" factor.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5          # seed_cube's proven voxel size (metres), left untouched
OFF_QUOTA = 12       # small, conservative reserve for off-plane points (as in c011)
MIN_THRESH = 0.04    # metres, floor on the robust off-plane distance threshold
MAD_K = 2.5          # robust-scale multiplier on median-absolute-deviation

MORTON_BITS = 16
MORTON_OFFSET = 1 << 15  # shift so signed 16-bit-range cube coords become non-negative


def _spread_bits(v):
    # interleave the low MORTON_BITS bits of v with two zero gaps (for 3D Morton code)
    r = 0
    for i in range(MORTON_BITS):
        if (v >> i) & 1:
            r |= 1 << (3 * i)
    return r


def _morton3(x, y, z):
    xi = int(x) + MORTON_OFFSET
    yi = int(y) + MORTON_OFFSET
    zi = int(z) + MORTON_OFFSET
    xi &= (1 << MORTON_BITS) - 1
    yi &= (1 << MORTON_BITS) - 1
    zi &= (1 << MORTON_BITS) - 1
    return _spread_bits(xi) | (_spread_bits(yi) << 1) | (_spread_bits(zi) << 2)


def _cube_round_robin(idx_order, P, taken, k):
    # group remaining candidates (score-descending) into 0.5 m cubes
    if k <= 0:
        return []
    key = np.floor(P / CUBE).astype(np.int64)
    groups = {}
    for pos in range(len(idx_order)):
        orig = int(idx_order[pos])
        if orig in taken:
            continue
        kk = (int(key[pos, 0]), int(key[pos, 1]), int(key[pos, 2]))
        lst = groups.get(kk)
        if lst is None:
            groups[kk] = [orig]
        else:
            lst.append(orig)

    if not groups:
        return []

    cube_keys = list(groups.keys())
    C = len(cube_keys)
    # deterministic spatial (Morton / Z-order) sort -- replaces the old score-first-occurrence
    # order, which silently starved whole regions of space whenever C > k.
    order = sorted(cube_keys, key=lambda kk: _morton3(kk[0], kk[1], kk[2]))

    if C <= k:
        # budget is generous enough to touch every cube at least once; classic round-robin,
        # now just using the spatial order for determinism/tie-breaking.
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

    # C > k: budget cannot touch every cube. Instead of silently truncating the spatially
    # sorted list (which would still bias toward one side of space), take an evenly spaced
    # stride sample of cube indices along the Morton curve so the k picks are spread across
    # the *whole* occupied volume, then take the strongest point in each selected cube.
    picks_idx = np.unique(np.round(np.linspace(0, C - 1, k)).astype(int))
    if len(picks_idx) < k:
        chosen_set = set(int(i) for i in picks_idx)
        remaining = [i for i in range(C) if i not in chosen_set]
        needed = k - len(picks_idx)
        picks_idx = np.concatenate([picks_idx, np.array(remaining[:needed], dtype=int)])
    picks_idx = np.sort(picks_idx)[:k]

    sel = [groups[order[int(i)]][0] for i in picks_idx]
    return sel[:k]


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
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])   # (nk, 3), rows in score-descending order (ok preserves input order)

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
                off_local = np.flatnonzero(off_mask)   # already score-descending (same order as ok/P)
                quota = min(OFF_QUOTA, m, len(off_local))
                for c in off_local[:quota]:
                    orig = int(ok[c])
                    if orig not in taken:
                        sel.append(orig)
                        taken.add(orig)

    # ---- fill the remainder with the spatially-sorted (Morton) cube round-robin ----
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
