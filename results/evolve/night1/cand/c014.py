# idea: reserve up to 24 picks for an explicit 6x4 image-space grid (one strongest point per cell) to guarantee wide pixel coverage, then fill the remaining budget with seed_cube's proven unmodified 0.5 m 3D-cube round-robin, since the brief states wide image coverage helps pose but no prior child tested it directly instead of tweaking 3D/plane logic.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5
IMG_COLS = 6
IMG_ROWS = 4
IMG_QUOTA = IMG_COLS * IMG_ROWS  # 24, one guaranteed pick per image-grid cell


def _cube_round_robin(orig_idx, P, taken, k):
    # one point per CUBE-sized 3D cube per round, in score order (seed_cube's proven logic, untouched)
    if k <= 0:
        return []
    key = np.floor(P / CUBE).astype(np.int64)
    groups = {}
    order = []
    for pos in range(len(orig_idx)):
        o = int(orig_idx[pos])
        if o in taken:
            continue
        kk = (int(key[pos, 0]), int(key[pos, 1]), int(key[pos, 2]))
        lst = groups.get(kk)
        if lst is None:
            groups[kk] = [o]
            order.append(kk)
        else:
            lst.append(o)
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


def _image_grid_quota(orig_idx, u_ok, v_ok, taken, k):
    # one strongest point per image-space grid cell per round -> guarantees wide pixel coverage
    if k <= 0:
        return []
    cell_w = 640.0 / IMG_COLS
    cell_h = 480.0 / IMG_ROWS
    cx = np.clip((u_ok // cell_w).astype(np.int64), 0, IMG_COLS - 1)
    cy = np.clip((v_ok // cell_h).astype(np.int64), 0, IMG_ROWS - 1)
    groups = {}
    order = []
    for pos in range(len(orig_idx)):
        o = int(orig_idx[pos])
        if o in taken:
            continue
        kk = (int(cx[pos]), int(cy[pos]))
        lst = groups.get(kk)
        if lst is None:
            groups[kk] = [o]
            order.append(kk)
        else:
            lst.append(o)
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

    # ---- phase 1: explicit image-space grid quota for wide pixel coverage ----
    img_quota = min(IMG_QUOTA, m)
    img_sel = _image_grid_quota(ok, u[ok], v[ok], taken, img_quota)
    sel += img_sel
    taken.update(img_sel)

    # ---- phase 2: fill the remainder with seed_cube's exact 0.5 m 3D-cube round-robin ----
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
