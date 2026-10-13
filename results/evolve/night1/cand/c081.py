# idea: replace the parent's post-hoc, bounded (MAX_SWAPS=6) grid-cell coverage-repair swap with an upfront two-level round-robin allocation that interleaves per-image-cell cube-diversity queues across all GX x GY cells from the very first pick, guaranteeing uniform spatial coverage and 3D cube diversity simultaneously instead of only patching up to 6 empty cells after a global cube-only fill.
import numpy as np
from kpbench.data import to3d, DEPTH_MAX

CUBE = 0.5           # seed_cube's proven voxel size (metres)
OFF_QUOTA = 10        # small, conservative off-plane reserve (as in c011/c020/c027/c042)
MIN_THRESH = 0.04     # metres, floor on robust off-plane distance threshold
MAD_K = 2.5           # robust-scale multiplier on median-absolute-deviation
GX, GY = 4, 3         # coarse image grid for upfront stratified allocation (640x480 -> 160x160 cells)


def _cell_cube_queue(idxs, P_sub):
    # build the full score-descending, cube-round-robin ordering of idxs restricted to one image cell
    if len(idxs) == 0:
        return []
    key = np.floor(P_sub / CUBE).astype(np.int64)
    groups = {}
    order = []
    for i in range(len(idxs)):
        kk = (int(key[i, 0]), int(key[i, 1]), int(key[i, 2]))
        lst = groups.get(kk)
        if lst is None:
            groups[kk] = [idxs[i]]
            order.append(kk)
        else:
            lst.append(idxs[i])

    q = []
    r = 0
    while True:
        added = False
        for kk in order:
            g = groups[kk]
            if r < len(g):
                q.append(g[r])
                added = True
        if not added:
            break
        r += 1
    return q


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
                for c in off_local[:quota]:
                    orig = int(ok[c])
                    if orig not in taken:
                        sel.append(orig)
                        taken.add(orig)

    # ---- upfront stratified fill: partition remaining candidates into GX x GY image
    #      cells, build a per-cell cube-round-robin queue (one point per CUBE-sized
    #      voxel per round, strongest first), then interleave ACROSS cells in a
    #      single outer round-robin. This bakes both wide image coverage and 3D cube
    #      diversity into the very first picks, instead of only patching a handful of
    #      empty cells after a global, image-coverage-blind cube fill. ----
    remaining = m - len(sel)
    if remaining > 0 and len(ok) > 0:
        cw = 640.0 / GX
        ch = 480.0 / GY

        def cell_of(ui, vi):
            gx = int(min(GX - 1, max(0, np.floor(ui / cw))))
            gy = int(min(GY - 1, max(0, np.floor(vi / ch))))
            return gy * GX + gx

        cell_idxs = {c: [] for c in range(GX * GY)}
        cell_pos = {c: [] for c in range(GX * GY)}
        for pos in range(len(ok)):
            orig = int(ok[pos])
            if orig in taken:
                continue
            c = cell_of(u[orig], v[orig])
            cell_idxs[c].append(orig)
            cell_pos[c].append(pos)

        cell_queues = {}
        for c in range(GX * GY):
            idxs = cell_idxs[c]
            if not idxs:
                continue
            P_sub = P[cell_pos[c]]
            cell_queues[c] = _cell_cube_queue(idxs, P_sub)

        cell_order = sorted(cell_queues.keys())
        pointers = {c: 0 for c in cell_order}
        while remaining > 0:
            progressed = False
            for c in cell_order:
                if remaining <= 0:
                    break
                q = cell_queues[c]
                p = pointers[c]
                if p < len(q):
                    orig = q[p]
                    pointers[c] = p + 1
                    if orig not in taken:
                        sel.append(orig)
                        taken.add(orig)
                        remaining -= 1
                        progressed = True
                    else:
                        progressed = True
            if not progressed:
                break

    # ---- fallback: any leftover untaken valid-depth candidates by score order ----
    if len(sel) < m:
        for pos in range(len(ok)):
            if len(sel) >= m:
                break
            orig = int(ok[pos])
            if orig not in taken:
                sel.append(orig)
                taken.add(orig)

    # ---- final fallback: any leftover indices (e.g. sky points) by original order ----
    if len(sel) < m:
        for i in range(n):
            if len(sel) >= m:
                break
            if i not in taken:
                sel.append(i)
                taken.add(i)

    return np.array(sel[:m], dtype=int)
