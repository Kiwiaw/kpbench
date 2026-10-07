"""Pickers: choose K of the candidates of one frame. Input arrays are in score order (index 0 = strongest).

All pickers except 'score' only use candidates with depth (z < DEPTH_MAX); that is the 'd' in 'sd'.
Names:
  score        strongest K, any depth
  sd           strongest K with depth
  cube<s>      round robin over s-metre cubes in the camera frame (e.g. cube0.5), strongest first per cube
  grid         ceil(sqrt K)^2 image cells over the strongest M = clip(4K, 80, 400) candidates; best per cell, fill by rank
  grid10x8     10 x 8 image cells, round robin
  fps          2D farthest point sampling over the same pool as grid, score-aware: next = argmax dist * (1 - rank/M)
  fps0         plain 2D farthest point sampling over the pool (no score term)
  cube<s>p<M>  cube round robin over the strongest M candidates only (cube0.5p200)
  hyb<f>[c<s>] strongest f*K with depth first, the rest by cube round robin (s m, default 0.5) over the grid pool (hyb0.5)
  offp<n>[c][p<M>] K-n strongest with depth + n points OFF the dominant plane (RANSAC plane on the strongest 300 with depth,
               30 cm tolerance), spread over image cells (c = 0.5 m cubes instead) among the strongest M (default 2000), best per cell
  orep[c][p<M>] ORACLE: candidates re-ranked by how many of the frames f-20, f-10, f+10, f+20 re-find them (true pose and
               depth, 3 px, 5 % depth, among that frame's strongest 300 with depth), ties by rank; c = cube 0.5 m round robin on
               the re-ranked list; pool M (default 2000). An upper bound for any learned "predict which points persist" model.
  dist<t>[c][p<M>] intra-frame distinctiveness: candidates whose descriptor has a cosine similarity above t to another candidate of
               the pool (default 400) are moved behind the distinct ones (ties by rank); c = cube 0.5 m round robin after that.
  far2d, far3d plain farthest point sampling over the strongest 2000 candidates with depth: pixel distance (2d) or
               camera-frame Euclidean distance in metres (3d); suffix w = score-aware gain, p<M> = pool size (far3dp400)
A picker returns candidate indices (at most K; fewer only when the frame has too few candidates).
To add a picker: write a function (u, v, s, z, K) -> indices and register it in PICKERS or in parse().
"""
import os
import re
import numpy as np
from .data import to3d, DEPTH_MAX, W, H


def round_robin(groups, K):
    """groups: lists of indices, each in score order; take the best of each group in turn (groups ordered by their best)"""
    order = sorted(groups, key=lambda g: g[0]); sel = []; r = 0
    while len(sel) < K and any(len(g) > r for g in order):
        for g in order:
            if len(g) > r:
                sel.append(g[r])
            if len(sel) == K:
                break
        r += 1
    return np.array(sel, int)


def fill(sel, n, K, pool=None):
    """top up a selection with the best remaining candidates (from pool, else all) in score order"""
    if len(sel) >= K:
        return np.asarray(sel[:K], int)
    have = set(int(i) for i in sel)
    rest = [i for i in (range(n) if pool is None else pool) if int(i) not in have][:K - len(sel)]
    return np.concatenate([np.asarray(sel, int), np.asarray(rest, int)]) if rest else np.asarray(sel, int)


def pick_score(u, v, s, z, K):
    return np.arange(min(K, len(u)))


def pick_sd(u, v, s, z, K):
    return np.flatnonzero(z < DEPTH_MAX)[:K]


def pick_cube(size):
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        key = np.floor(to3d(u[ok], v[ok], z[ok]) / size).astype(int); g = {}
        for jj, j in enumerate(ok):
            g.setdefault(tuple(key[jj]), []).append(int(j))
        return fill(round_robin(list(g.values()), K), len(u), K)
    return f


def pool_of(z, K):
    M = int(np.clip(4 * K, 80, 400))
    return np.flatnonzero(z < DEPTH_MAX)[:M]


def grid_cells(P, K):
    """indices into P: best-ranked point per cell of a g x g grid, g = ceil(sqrt K); then fill by rank"""
    g = int(np.ceil(np.sqrt(K)))
    cell = (np.clip(P[:, 0] * g // W, 0, g - 1) * g + np.clip(P[:, 1] * g // H, 0, g - 1)).astype(int)
    _, first = np.unique(cell, return_index=True)
    sel = sorted(first)[:K]
    if len(sel) < K:
        sel += [i for i in range(len(P)) if i not in set(sel)][:K - len(sel)]
    return np.array(sel, int)


def pick_grid(u, v, s, z, K):
    pool = pool_of(z, K)
    if not len(pool):
        return np.arange(min(K, len(u)))
    return pool[grid_cells(np.stack([u[pool], v[pool]], 1), K)]


def pick_grid10x8(u, v, s, z, K):
    ok = np.flatnonzero(z < DEPTH_MAX); g = {}
    for j in ok:
        g.setdefault((int(u[j] * 10 // W), int(v[j] * 8 // H)), []).append(int(j))
    return fill(round_robin(list(g.values()), K) if g else np.array([], int), len(u), K)


def fps_2d(P, K, score_aware):
    """greedy farthest point sampling in the image; seed = rank 0. score_aware: gain = dist * (1 - rank / M)"""
    M = len(P)
    if M == 0:
        return np.array([], int)
    w = (1.0 - np.arange(M) / M) if score_aware else np.ones(M)
    sel = [0]; d = np.linalg.norm(P - P[0], axis=1)
    while len(sel) < min(K, M):
        d[sel] = -1
        i = int(np.argmax(d * w)); sel.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.array(sel, int)


def pick_fps(score_aware):
    def f(u, v, s, z, K):
        pool = pool_of(z, K)
        if not len(pool):
            return np.arange(min(K, len(u)))
        return pool[fps_2d(np.stack([u[pool], v[pool]], 1), K, score_aware)]
    return f


def fps_nd(P, K, w=None):
    """greedy farthest point sampling in any dimension (Euclidean distance); seed = rank 0; optional per-point weight on the gain"""
    M = len(P)
    if M == 0:
        return np.array([], int)
    w = np.ones(M) if w is None else w
    sel = [0]; d = np.linalg.norm(P - P[0], axis=1)
    while len(sel) < min(K, M):
        d[sel] = -1
        i = int(np.argmax(d * w)); sel.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.array(sel, int)


def pick_far(space, pool_size, score_aware):
    """far2d / far3d: plain farthest point sampling over the strongest `pool_size` candidates with depth
    (default 2000 = the whole candidate list). 2d = pixel distance, 3d = Euclidean distance in the camera frame (metres).
    Suffix w = score-aware gain dist * (1 - rank / pool)."""
    def f(u, v, s, z, K):
        pool = np.flatnonzero(z < DEPTH_MAX)[:pool_size]
        if not len(pool):
            return np.arange(min(K, len(u)))
        P = to3d(u[pool], v[pool], z[pool]) if space == '3d' else np.stack([u[pool], v[pool]], 1)
        w = (1.0 - np.arange(len(pool)) / len(pool)) if score_aware else None
        return pool[fps_nd(P, K, w)]
    return f


def pick_seeded_far(n_seed, M):
    """sfar<n>[p<M>]: seed-stable farthest apart. The n strongest candidates with depth are kept as seeds (the stable, reliably
    re-detected points), then farthest-point sampling in the camera frame (3D metres) over the strongest M candidates adds the
    remaining K - n points, each the farthest from everything already chosen. Spread is bought only from the strong pool."""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        n = min(n_seed, K, len(ok)); pool = ok[:M]
        if len(pool) <= n:
            return fill(pool, len(u), K)
        P = to3d(u[pool], v[pool], z[pool]); sel = list(range(n))
        d = np.min(np.linalg.norm(P[:, None] - P[None, :n], axis=2), axis=1)
        while len(sel) < min(K, len(pool)):
            d[sel] = -1; i = int(np.argmax(d)); sel.append(i)
            d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
        return fill(pool[np.array(sel, int)], len(u), K)
    return f


GEO_STEP = 8   # depth map downsampled by this factor for the surface graph (80 x 60 nodes)


def geodesic_graph(depth, step=GEO_STEP):
    """surface graph of a frame: nodes = pixels of the downsampled depth map with depth, edges = 8-neighbours,
    weight = 3D distance between the two surface points (a jump across an occlusion edge costs its full 3D length)"""
    from scipy.sparse import coo_matrix
    D = depth[::step, ::step]; h, w = D.shape
    vv, uu = np.mgrid[0:h, 0:w]; X = to3d((uu * step).ravel().astype(float), (vv * step).ravel().astype(float), D.ravel())
    valid = (D.ravel() < DEPTH_MAX); idx = np.arange(h * w).reshape(h, w); rows, cols, wts = [], [], []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = idx[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)].ravel(); b = idx[max(0, dy):h + min(0, dy) or h, max(0, dx):w + min(0, dx) or w].ravel()
        a, b = a[:len(b)], b[:len(a)]
        m = valid[a] & valid[b]; rows.append(a[m]); cols.append(b[m]); wts.append(np.linalg.norm(X[a[m]] - X[b[m]], axis=1))
    r = np.concatenate(rows); c = np.concatenate(cols); wt = np.concatenate(wts) + 1e-6
    G = coo_matrix((np.concatenate([wt, wt]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(h * w, h * w)).tocsr()
    return G, h, w


def pick_geodesic(M):
    """geo[p<M>]: farthest apart along the surface. Distances are geodesic on the frame's depth map (shortest path over the
    surface graph of geodesic_graph), not straight-line, so two points on different walls or across a gap count as far apart
    only if the surface between them is long. Greedy farthest-point sampling over the strongest M candidates with depth,
    seed = the strongest. Needs the frame's depth map (needs_ctx)."""
    def f(u, v, s, z, K, ctx=None):
        from scipy.sparse.csgraph import dijkstra
        pool = np.flatnonzero(z < DEPTH_MAX)[:M]
        if ctx is None or len(pool) < 2:
            return fill(pool, len(u), K)
        F, C, fi = ctx; G, h, w = geodesic_graph(F.depth(fi))
        node = (np.clip(v[pool] / GEO_STEP, 0, h - 1).astype(int)) * w + np.clip(u[pool] / GEO_STEP, 0, w - 1).astype(int)
        sel = [0]; d = dijkstra(G, directed=False, indices=[node[0]], min_only=True)[node]
        while len(sel) < min(K, len(pool)):
            d[sel] = -1; d[~np.isfinite(d)] = -1; i = int(np.argmax(d))
            if d[i] < 0:
                break
            sel.append(i); d = np.minimum(d, dijkstra(G, directed=False, indices=[node[i]], min_only=True)[node])
        return fill(pool[np.array(sel, int)], len(u), K)
    f.needs_ctx = True
    return f


def cube_groups(u, v, z, idx, size):
    """candidate indices idx (score order) -> list of per-cube index lists (camera-frame cubes of `size` metres)"""
    key = np.floor(to3d(u[idx], v[idx], z[idx]) / size).astype(int); g = {}
    for jj, j in enumerate(idx):
        g.setdefault(tuple(key[jj]), []).append(int(j))
    return list(g.values())


def pick_hyb(frac, size):
    """hyb<frac>: insurance against starvation. The strongest frac*K points with depth are taken unconditionally, the rest
    are filled by cube round robin (size m) over the strongest M = clip(4K, 80, 400) candidates with depth, skipping cubes
    the first part already occupies first. E.g. hyb0.5 at K=100: 50 strongest + 50 spread."""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        h = int(round(frac * K)); first = ok[:h]
        pool = ok[h:int(np.clip(4 * K, 80, 400))]
        if not len(pool):
            return fill(first, len(u), K)
        taken = set(tuple(k) for k in np.floor(to3d(u[first], v[first], z[first]) / size).astype(int))
        groups = cube_groups(u, v, z, pool, size)
        key_of = {g[0]: tuple(np.floor(to3d(u[[g[0]]], v[[g[0]]], z[[g[0]]]) / size).astype(int)[0]) for g in groups}
        fresh = [g for g in groups if key_of[g[0]] not in taken]; old = [g for g in groups if key_of[g[0]] in taken]
        rest = round_robin(fresh, K - h) if fresh else np.array([], int)
        if len(rest) < K - h and old:
            rest = np.concatenate([rest, round_robin(old, K - h - len(rest))])
        return fill(np.concatenate([first, rest]).astype(int), len(u), K)
    return f


def pick_cube_pool(size, M):
    """cube<size>p<M>: cube round robin restricted to the strongest M candidates with depth (the cube analogue of far3dp<M>)"""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)[:M]
        if not len(ok):
            return np.arange(min(K, len(u)))
        return fill(round_robin(cube_groups(u, v, z, ok, size), K), len(u), K)
    return f


def dominant_plane(P, tol=0.3, iters=300, seed=0):
    """RANSAC plane through 3 random points (same trials as labels.plane_frac); returns (point, unit normal, inlier count)"""
    rng = np.random.default_rng(seed); best = (None, None, 0)
    if len(P) < 10:
        return best
    for _ in range(iters):
        i = rng.choice(len(P), 3, replace=False); a, b, c = P[i]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn; cnt = int((np.abs((P - a) @ n) < tol).sum())
        if cnt > best[2]:
            best = (a, n, cnt)
    return best


def pick_offplane(n_off, M, cubes, tol=0.3):
    """offp<n>: the K-n strongest candidates with depth, plus n candidates that lie off the dominant plane of the frame
    (farther than tol from the RANSAC plane fitted to the strongest 300 with depth), spread uniformly: best point per image
    cell of a ceil(sqrt n)^2 grid (cubes=True: 0.5 m cube round robin), among the strongest M candidates with depth.
    Too few off-plane points -> fill by rank. The idea: keep RaCo's strong points, add a few that break the plane."""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        h = max(K - n_off, 0); first = ok[:h]
        a, nrm, cnt = dominant_plane(to3d(u[ok[:300]], v[ok[:300]], z[ok[:300]]), tol)
        if a is None:
            return fill(first, len(u), K)
        pool = ok[h:M]
        d = np.abs((to3d(u[pool], v[pool], z[pool]) - a) @ nrm)
        off = pool[d > tol]
        if len(off):
            if cubes:
                sel = round_robin(cube_groups(u, v, z, off, 0.5), K - h)
            else:
                sel = off[grid_cells(np.stack([u[off], v[off]], 1), K - h)]
        else:
            sel = np.array([], int)
        return fill(np.concatenate([first, sel]).astype(int), len(u), K)
    return f


def pick_dband(ratio, cubes, M):
    """dband<ratio>[c][p<M>]: depth-band round robin, for position conditioning on flat views.
    Candidates = the strongest M with depth. Depth is cut into bands growing by `ratio` (ratio 2: 0.5-1, 1-2, 2-4, 4-8 m ...),
    so near and far points are both guaranteed a share: with few inliers all at one depth the pose can slide along the
    viewing direction with the rotation still right (the dominant remaining failure on flat views). Within a band the order
    is score order, or cube 0.5 m round robin (c). Bands are served in turn, best band first; the rest filled by rank."""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)[:M]
        if not len(ok):
            return np.arange(min(K, len(u)))
        band = np.floor(np.log(np.maximum(z[ok], 0.25)) / np.log(ratio)).astype(int); g = {}
        for jj, j in enumerate(ok):
            g.setdefault(int(band[jj]), []).append(int(j))
        groups = list(g.values())
        if cubes:
            groups = [list(np.asarray(gr, int)[pick_cube(0.5)(u[gr], v[gr], s[gr], z[gr], len(gr))]) for gr in groups]
        return fill(round_robin(groups, K), len(u), K)
    return f


_LEARN = {}


def pick_learn(cubes, M, model='learn'):
    """learn[c][p<M>]: a learned re-ranker on top of the detector. A LightGBM model (kpbench/models/<model>.txt, trained on
    pose-derived labels: a candidate is positive when it became a PnP inlier for its query) scores every candidate among the
    strongest M with depth from frame-only features (rank, score, depth, position, local density, the frame's distinct-cube
    count, and the 256-d descriptor). The K highest scores are kept, or cube 0.5 m round robin over that order (c).
    Needs the descriptors (needs_desc)."""
    def load():
        if model not in _LEARN:
            import lightgbm as lgb
            _LEARN[model] = lgb.Booster(model_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', model + '.txt'))
        return _LEARN[model]

    def f(u, v, s, z, K, de=None):
        from scipy.spatial import cKDTree
        cand = np.flatnonzero(z < DEPTH_MAX)[:M]
        if len(cand) < 20 or de is None:
            return np.flatnonzero(z < DEPTH_MAX)[:K]
        allc = np.flatnonzero(z < DEPTH_MAX)[:500]
        tree = cKDTree(np.stack([u[allc], v[allc]], 1))
        dens = np.array([len(x) - 1 for x in tree.query_ball_point(np.stack([u[cand], v[cand]], 1), 12.0)])
        top = np.flatnonzero(z < DEPTH_MAX)[:100]
        ncube = len(set(map(tuple, np.floor(to3d(u[top], v[top], z[top]) / 0.5).astype(int))))
        X = np.stack([cand / 2000.0, s[cand] / max(float(s[0]), 1e-6), np.log(np.maximum(z[cand], 0.1)), u[cand] / W, v[cand] / H,
                      dens / 20.0, np.full(len(cand), ncube / 100.0)], 1).astype(np.float32)
        X = np.concatenate([X, de[cand].astype(np.float32)], 1)
        order = cand[np.argsort(-load().predict(X))]
        return _rerank_pick(order, u, v, s, z, K, cubes)
    f.needs_desc = True
    return f


def _rerank_pick(order, u, v, s, z, K, cubes):
    """take K from candidates in the given order (indices), or cube 0.5 m round robin over that order"""
    if not cubes:
        return np.asarray(order[:K], int)
    o = np.asarray(order, int)
    return o[pick_cube(0.5)(u[o], v[o], s[o], z[o], K)]


ORACLE_LAGS = (-20, -10, 10, 20)


def pick_oracle_rep(cubes, M):
    """orep: ORACLE picker (uses the true poses of the neighbouring frames). Needs ctx=(Flight, Cands, frame)."""
    from scipy.spatial import cKDTree
    from .data import FX, FY, CX, CY
    cache = {}
    def f(u, v, s, z, K, ctx=None):
        F, C, q = ctx
        ok = np.flatnonzero(z < DEPTH_MAX)[:M]
        if not len(ok):
            return np.arange(min(K, len(u)))
        Rq, tq = F.cam_pose(q); Xw = to3d(u[ok], v[ok], z[ok]) @ Rq.T + tq
        cnt = np.zeros(len(ok), int)
        for L in ORACLE_LAGS:
            g = q + L
            if g < 0 or g >= F.N:
                continue
            key = (F.slug, g)
            if key not in cache:
                u2, v2, s2, z2, _ = C.frame(g); j = np.flatnonzero(z2 < DEPTH_MAX)[:300]
                cache[key] = (cKDTree(np.stack([u2[j], v2[j]], 1)) if len(j) else None, z2[j])
            tree, z2 = cache[key]
            if tree is None:
                continue
            Rf, tf = F.cam_pose(g); Xc = (Xw - tf) @ Rf; zc = Xc[:, 2]; good = zc > 0.05
            pu = FX * Xc[:, 0] / np.where(good, zc, 1) + CX; pv = FY * Xc[:, 1] / np.where(good, zc, 1) + CY
            d, nn = tree.query(np.stack([pu, pv], 1), distance_upper_bound=3.0)
            hit = good & np.isfinite(d)
            hit[hit] &= np.abs(z2[nn[hit]] - zc[hit]) <= 0.05 * zc[hit]
            cnt += hit
        order = ok[np.lexsort((np.arange(len(ok)), -cnt))]
        return fill(_rerank_pick(order, u, v, s, z, K, cubes), len(u), K)
    f.needs_ctx = True
    return f


def pick_distinct(t, cubes, M):
    """dist<t>: intra-frame descriptor distinctiveness. Needs de= (descriptors, L2-normalised rows)."""
    def f(u, v, s, z, K, de=None):
        ok = np.flatnonzero(z < DEPTH_MAX)[:M]
        if not len(ok):
            return np.arange(min(K, len(u)))
        D = de[ok].astype(np.float32); D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
        S = D @ D.T; np.fill_diagonal(S, -1)
        amb = (S.max(1) > t).astype(int)
        order = ok[np.lexsort((np.arange(len(ok)), amb))]
        return fill(_rerank_pick(order, u, v, s, z, K, cubes), len(u), K)
    f.needs_desc = True
    return f


PICKERS = {'score': pick_score, 'sd': pick_sd, 'grid': pick_grid, 'grid10x8': pick_grid10x8,
           'fps': pick_fps(True), 'fps0': pick_fps(False)}


def pick_poisson_disk(r0, theta):
    """pd<r>[a<theta>]: 3D Poisson-disk selection in score order. A candidate is accepted when its camera-frame distance to every
    point accepted in the same pass is at least r = max(r0, theta * min(Z_i, Z_j)) (theta = 0: fixed metric radius, the clean
    version of the cube round robin; theta > 0: radius grows with depth, so close-up views are thinned at a finer scale).
    Passes repeat over the rejected candidates until K points are chosen (second pass = second point per disk, like the cube
    round robin), then the rest is filled by rank."""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        P = to3d(u[ok], v[ok], z[ok]); Z = z[ok]; sel = []; rest = list(range(len(ok)))
        while rest and len(sel) < K:
            acc = []; A = np.zeros((0, 3)); AZ = np.zeros(0)
            for j in rest:
                if len(acc):
                    rr = np.maximum(r0, theta * np.minimum(AZ, Z[j]))
                    if (np.linalg.norm(A - P[j], axis=1) < rr).any():
                        continue
                acc.append(j); A = np.vstack([A, P[j]]); AZ = np.append(AZ, Z[j])
                if len(sel) + len(acc) >= K:
                    break
            sel += acc; rest = [j for j in rest if j not in set(acc)]
        return fill(ok[np.array(sel[:K], int)], len(u), K)
    return f


def pick_dpp(sigma, alpha):
    """dpp<sigma>[a<alpha>]: greedy MAP of a determinantal point process over the candidates with depth, L = diag(q) K diag(q),
    K_ij = exp(-|X_i - X_j|^2 / (2 sigma^2)) in camera-frame metres, quality q_i = exp(-alpha * rank_i / 100).
    One objective for strength and spread: each pick maximises the gain in log det (fast greedy of Chen, Zhang, Zhou 2018)."""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        P = to3d(u[ok], v[ok], z[ok]); n = len(ok); q = np.exp(-alpha * np.arange(n) / 100.0)
        d2 = q ** 2; c = np.zeros((0, n)); sel = []
        for _ in range(min(K, n)):
            j = int(np.argmax(d2))
            if d2[j] <= 1e-12:
                break
            dj = np.sqrt(d2[j]); Lj = q[j] * q * np.exp(-np.sum((P - P[j]) ** 2, 1) / (2 * sigma ** 2))
            e = (Lj - (c[:, j] @ c if len(c) else 0.0)) / dj; c = np.vstack([c, e]); d2 = d2 - e ** 2; d2[j] = -np.inf; sel.append(j)
        return fill(ok[np.array(sel, int)], len(u), K)
    return f


def parse(name):
    """picker name -> function. cube<size>; far2d|far3d[w][p<pool>], e.g. far3d, far2dw, far3dp400"""
    if name in PICKERS:
        return PICKERS[name]
    m = re.fullmatch(r'cube([0-9.]+)p([0-9]+)', name)
    if m:
        return pick_cube_pool(float(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r'pd([0-9.]+)(?:a([0-9.]+))?', name)
    if m:
        return pick_poisson_disk(float(m.group(1)), float(m.group(2) or 0))
    m = re.fullmatch(r'dpp([0-9.]+)(?:a([0-9.]+))?', name)
    if m:
        return pick_dpp(float(m.group(1)), float(m.group(2) or 1.0))
    m = re.fullmatch(r'orep(c?)(?:p([0-9]+))?', name)
    if m:
        return pick_oracle_rep(m.group(1) == 'c', int(m.group(2) or 2000))
    m = re.fullmatch(r'dist([0-9.]+)(c?)(?:p([0-9]+))?', name)
    if m:
        return pick_distinct(float(m.group(1)), m.group(2) == 'c', int(m.group(3) or 400))
    m = re.fullmatch(r'offp([0-9]+)(c?)(?:p([0-9]+))?', name)
    if m:
        return pick_offplane(int(m.group(1)), int(m.group(3) or 2000), m.group(2) == 'c')
    m = re.fullmatch(r'sfar([0-9]+)(?:p([0-9]+))?', name)
    if m:
        return pick_seeded_far(int(m.group(1)), int(m.group(2) or 400))
    m = re.fullmatch(r'geo(?:p([0-9]+))?', name)
    if m:
        return pick_geodesic(int(m.group(1) or 400))
    m = re.fullmatch(r'learn(c?)(?:p([0-9]+))?', name)
    if m:
        return pick_learn(m.group(1) == 'c', int(m.group(2) or 400))
    m = re.fullmatch(r'dband([0-9.]+)(c?)(?:p([0-9]+))?', name)
    if m:
        return pick_dband(float(m.group(1)), m.group(2) == 'c', int(m.group(3) or 400))
    m = re.fullmatch(r'hyb([0-9.]+)(?:c([0-9.]+))?', name)
    if m:
        return pick_hyb(float(m.group(1)), float(m.group(2) or 0.5))
    if name.startswith('cube'):
        return pick_cube(float(name[4:]))
    m = re.fullmatch(r'far(2d|3d)(w?)(?:p(\d+))?', name)
    if m:
        return pick_far(m.group(1), int(m.group(3) or 2000), m.group(2) == 'w')
    raise KeyError('unknown picker %s' % name)
