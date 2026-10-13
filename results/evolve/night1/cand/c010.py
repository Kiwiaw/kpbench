# idea: keep seed_cube's proven 3D-cube round-robin selection exactly as the base, then apply a tightly bounded (<=12) swap repair that fills any completely empty cell of a coarse 2D image grid by stealing the weakest point from an over-populated cell, directly targeting the known "wide image coverage helps pose" factor without disturbing frames where coverage is already fine.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5
GRID_COLS = 8
GRID_ROWS = 6
MAX_SWAPS = 12
IMG_W = 640.0
IMG_H = 480.0


def pick(u, v, s, z, K):
    n = len(u)
    m = min(K, n)
    if m <= 0:
        return np.array([], dtype=int)

    ok = np.flatnonzero(z < DEPTH_MAX)
    if not len(ok):
        return np.arange(m)

    P = to3d(u[ok], v[ok], z[ok])
    key = np.floor(P / CUBE).astype(int)
    g = {}
    for jj, j in enumerate(ok):
        g.setdefault(tuple(key[jj]), []).append(int(j))

    groups = list(g.values())
    sel = []
    r = 0
    while len(sel) < m and any(r < len(x) for x in groups):
        sel += [x[r] for x in groups if r < len(x)]
        r += 1
    sel = sel[:m]

    if len(sel) < m:
        used = set(sel)
        sel += [i for i in range(n) if i not in used][:m - len(sel)]
    sel = sel[:m]

    # --- bounded 2D-coverage repair pass (does nothing if coverage already ok) ---
    if m >= 8 and len(ok) > m:
        cell_w = IMG_W / GRID_COLS
        cell_h = IMG_H / GRID_ROWS

        def cell_of(idx):
            cx = int(min(GRID_COLS - 1, max(0, u[idx] // cell_w)))
            cy = int(min(GRID_ROWS - 1, max(0, v[idx] // cell_h)))
            return cy * GRID_COLS + cx

        sel_set = set(sel)
        cell_members = {}
        for idx in sel:
            c = cell_of(idx)
            cell_members.setdefault(c, []).append(idx)

        n_cells = GRID_COLS * GRID_ROWS
        empty_cells = [c for c in range(n_cells) if c not in cell_members]

        if empty_cells and cell_members:
            empty_set = set(empty_cells)
            pool_cell = {}
            for cand in ok:
                cand = int(cand)
                if cand in sel_set:
                    continue
                c = cell_of(cand)
                if c in empty_set and c not in pool_cell:
                    pool_cell[c] = cand

            swaps = 0
            for c in empty_cells:
                if swaps >= MAX_SWAPS:
                    break
                cand = pool_cell.get(c)
                if cand is None:
                    continue
                donor_cell = max(
                    cell_members,
                    key=lambda cc: (len(cell_members[cc]), max(cell_members[cc])),
                )
                if len(cell_members[donor_cell]) <= 1:
                    break
                worst = max(cell_members[donor_cell])
                sel_set.discard(worst)
                sel_set.add(cand)
                cell_members[donor_cell].remove(worst)
                cell_members.setdefault(c, []).append(cand)
                swaps += 1

            sel = list(sel_set)

    sel = sel[:m]
    if len(sel) < m:
        used = set(sel)
        sel += [i for i in range(n) if i not in used][:m - len(sel)]

    return np.array(sel[:m], dtype=int)
