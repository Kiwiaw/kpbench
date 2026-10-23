"""Pickers: choose K of the candidates of one frame. Input arrays are in score order (index 0 = strongest).

All pickers except 'score' only use candidates with depth (z < DEPTH_MAX); that is the 'd' in 'sd'.
Names:
  score        strongest K, any depth
  sd           strongest K with depth
  cube<s>      round robin over s-metre cubes in the camera frame (e.g. cube0.5), strongest first per cube
  cubem<c>     cube<s> with s = c x median depth of the frame's candidates with depth (cubem0.1, gap-agent ad_r0p1_a0)
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
  sfar<n>[p<M>] n strongest with depth as seeds + farthest apart (3D metres) over the strongest M (default 400)
  seed1<d|o>[p<M>] ONE stable seed (most re-found in f-20, f-10, f+10, f+20: d = mutual descriptor match, o = ORACLE
               reprojection), then farthest apart over the strongest M (default 400)
  dpp2d<sigma>[a<alpha>] dpp with the kernel on pixel coordinates (sigma in px) instead of camera-frame metres: depth only
               decides which candidates are eligible (as for sd), not the spread (T11 (b), docs/t11_dpp2d.md)
  up[b<m>][r<r>]+dpp<sigma>[a<alpha>][p<N>] upstream filter before dpp (T8, docs/t8_upstream_design.md): border margin m px,
               greedy NMS radius r px in score order, first N survivors; padded back to K from the dropped points
  ssc[<o>][p<M>] suppression via square covering (ANMS, Bailo 2018) over the strongest M with depth, ~o*K kept, K strongest
  dppr<sigma>a<alpha>b<beta>t<tau> dpp whose quality also rewards past-frame track count r^beta (T4, docs/t4_dppr_design.md)
  dppm<c>[a<alpha>] dpp with a per-frame scale sigma = c x median depth of the frame's candidates with depth (scene-scale dpp)
  dppmr<c>a<alpha>b<beta>t<tau> dppm's per-frame sigma = c x median depth + dppr's past-frame track bonus r^beta
  dpps<n>[q|l|e] dpp0.5 with a per-depth-slice cap ceil(K/n) (T12, docs/t12_dpps.md): q quantile, l log, e equal-width slices
  dppl<c>[a<alpha>] dpp with a per-point scale l_i = c x z_i (Gibbs kernel): spread measured relative to depth (~ angle c rad)
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


def pick_cube(size, rel=False):
    """cube<s>: round robin over s-metre cubes. rel (cubem<c>): cube edge = c x median depth of the frame's candidates with
    depth (the gap-agent harness picker ad_r<c>_a0, e.g. cubem0.1 = ad_r0p1_a0: the cube size follows the scene)"""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        e = size * float(np.median(z[ok])) if rel else size
        key = np.floor(to3d(u[ok], v[ok], z[ok]) / e).astype(int); g = {}
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


def pick_seed1_far(mode, M):
    """seed1<d|o>[p<M>]: ONE stable seed, then farthest apart. The seed is the candidate (strongest 300 with depth) that the
    frames f-20, f-10, f+10, f+20 re-find most often: 'd' = by mutual nearest-neighbour descriptor match (no pose used),
    'o' = ORACLE, by reprojection with the true poses (within 3 px and 5 % depth). Ties go to the stronger point. The other
    K - 1 points come from farthest-point sampling in the camera frame (3D metres) over the strongest M candidates, seeded at
    that point. Needs ctx=(Flight, Cands, frame) and de=descriptors."""
    from scipy.spatial import cKDTree
    from .data import FX, FY, CX, CY
    cache = {}
    def f(u, v, s, z, K, ctx=None, de=None):
        F, C, q = ctx
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        pool = ok[:M]; cand = ok[:min(300, M)]; cnt = np.zeros(len(cand), int)
        Rq, tq = F.cam_pose(q); Xw = to3d(u[cand], v[cand], z[cand]) @ Rq.T + tq
        for L in ORACLE_LAGS:
            g = q + L
            if g < 0 or g >= F.N:
                continue
            key = (F.slug, g)
            if key not in cache:
                u2, v2, s2, z2, de2 = C.frame(g); j = np.flatnonzero(z2 < DEPTH_MAX)[:300]
                D2 = de2[j].astype(np.float32); D2 /= np.linalg.norm(D2, axis=1, keepdims=True) + 1e-9
                cache[key] = (cKDTree(np.stack([u2[j], v2[j]], 1)) if len(j) else None, z2[j], D2)
            tree, z2, D2 = cache[key]
            if tree is None:
                continue
            if mode == 'o':
                Rf, tf = F.cam_pose(g); Xc = (Xw - tf) @ Rf; zc = Xc[:, 2]; good = zc > 0.05
                pu = FX * Xc[:, 0] / np.where(good, zc, 1) + CX; pv = FY * Xc[:, 1] / np.where(good, zc, 1) + CY
                d, nn = tree.query(np.stack([pu, pv], 1), distance_upper_bound=3.0)
                hit = good & np.isfinite(d)
                hit[hit] &= np.abs(z2[nn[hit]] - zc[hit]) <= 0.05 * zc[hit]
            else:
                D1 = de[cand].astype(np.float32); D1 /= np.linalg.norm(D1, axis=1, keepdims=True) + 1e-9
                S = D1 @ D2.T; a = S.argmax(1); b = S.argmax(0)
                hit = b[a] == np.arange(len(cand))
            cnt += hit
        seed = int(np.lexsort((np.arange(len(cand)), -cnt))[0])        # most re-found, then strongest
        P = to3d(u[pool], v[pool], z[pool]); sel = [seed]
        d = np.linalg.norm(P - P[seed], axis=1)
        while len(sel) < min(K, len(pool)):
            d[sel] = -1; i = int(np.argmax(d)); sel.append(i)
            d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
        return fill(pool[np.array(sel, int)], len(u), K)
    f.needs_ctx = True; f.needs_desc = True
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


def dpp_greedy(P, q, K, sigma, ell=None, slc=None, cap=None):
    """greedy MAP of L = diag(q) K diag(q), K_ij = exp(-|P_i - P_j|^2 / (2 sigma^2)) (fast greedy of Chen, Zhang, Zhou 2018).
    ell (per-point length scales, dppl): Gibbs kernel K_ij = (2 l_i l_j / (l_i^2 + l_j^2))^(D/2) exp(-|P_i - P_j|^2 / (l_i^2 + l_j^2)),
    positive semi-definite for any l > 0 (Gibbs 1997, Paciorek 2003); equal l_i = l_j = sigma gives the fixed kernel.
    slc, cap (dpps, docs/t12_dpps.md): slice id per point; each pick takes the argmax gain among points whose slice holds < cap
    picks; once no non-full slice has a point left (gain > 1e-12), the cap is dropped (phase 2, same objective)."""
    n = len(P); d2 = q ** 2; c = np.zeros((0, n)); sel = []
    if ell is not None:
        l2 = ell ** 2; hd = P.shape[1] / 2.0
    if slc is not None:
        cnt = np.zeros(int(slc.max()) + 1 if n else 1, int)
    for _ in range(min(K, n)):
        if slc is not None:
            el = (cnt[slc] < cap) & (d2 > 1e-12)
            if not el.any():
                slc = None                                   # phase 2: uncapped
        j = int(np.argmax(d2 if slc is None else np.where(el, d2, -np.inf)))
        if d2[j] <= 1e-12:
            break
        if ell is None:
            dj = np.sqrt(d2[j]); Lj = q[j] * q * np.exp(-np.sum((P - P[j]) ** 2, 1) / (2 * sigma ** 2))
        else:
            s2 = l2 + l2[j]; dj = np.sqrt(d2[j])
            Lj = q[j] * q * (2 * ell * ell[j] / s2) ** hd * np.exp(-np.sum((P - P[j]) ** 2, 1) / s2)
        e = (Lj - (c[:, j] @ c if len(c) else 0.0)) / dj; c = np.vstack([c, e]); d2 = d2 - e ** 2; d2[j] = -np.inf; sel.append(j)
        if slc is not None:
            cnt[slc[j]] += 1
    return np.array(sel, int)


def pick_dpp_slices(n, mode, sigma=0.5, alpha=1.0, pool=400):
    """dpps<n>[q|l|e] (T12, docs/t12_dpps.md): dpp0.5 (alpha 1) with a per-depth-slice cap ceil(K/n) in the greedy loop.
    Slice edges from the depths of the strongest `pool` candidates with depth: q equal-count quantiles, l log-spaced and
    e equal-width between their 5th and 95th percentile (outer slices open-ended). dpps1 = dpp0.5 exactly."""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        zz = z[ok]; P = to3d(u[ok], v[ok], zz); q = np.exp(-alpha * np.arange(len(ok)) / 100.0); zp = zz[:pool]
        if mode == 'q':
            edges = np.quantile(zp, np.arange(1, n) / n)
        else:
            lo, hi = np.percentile(zp, [5, 95])
            edges = (np.geomspace(max(lo, 1e-6), max(hi, 1e-6), n + 1) if mode == 'l' else np.linspace(lo, hi, n + 1))[1:-1]
        slc = np.searchsorted(edges, zz, side='right')
        return fill(ok[dpp_greedy(P, q, K, sigma, slc=slc, cap=int(np.ceil(K / n)))], len(u), K)
    return f


def pick_dpp(sigma, alpha, space='3d'):
    """dpp<sigma>[a<alpha>]: greedy MAP of a determinantal point process over the candidates with depth, L = diag(q) K diag(q),
    K_ij = exp(-|X_i - X_j|^2 / (2 sigma^2)) in camera-frame metres, quality q_i = exp(-alpha * rank_i / 100).
    One objective for strength and spread: each pick maximises the gain in log det (fast greedy of Chen, Zhang, Zhou 2018).
    space='2d' (dpp2d..): X = pixel coordinates, sigma in px."""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        P = to3d(u[ok], v[ok], z[ok]) if space == '3d' else np.stack([u[ok], v[ok]], 1); q = np.exp(-alpha * np.arange(len(ok)) / 100.0)
        return fill(ok[dpp_greedy(P, q, K, sigma)], len(u), K)
    return f


def pick_dpp_depth(c, alpha, mode):
    """Depth-adaptive dpp, same quality q_i = exp(-alpha * rank_i / 100) and candidates (with depth) as dpp<sigma>a<alpha>.
    mode 'm' (dppm<c>[a<alpha>]): one scale per frame, sigma = c * median depth of the frame's candidates with depth
      (dppm c with c * median = 0.5 m is dpp0.5 on that frame).
    mode 'l' (dppl<c>[a<alpha>]): per-point scale l_i = c * z_i in a Gibbs kernel (dpp_greedy ell), so two points are 'close'
      when their distance is small relative to their depth (~ angular separation c rad); PSD for any depths.
    f.sig collects the effective per-frame sigma (dppl: c * median depth of the picks) for diagnostics."""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        zz = z[ok]; P = to3d(u[ok], v[ok], zz); q = np.exp(-alpha * np.arange(len(ok)) / 100.0)
        if mode == 'm':
            sig = c * float(np.median(zz)); sel = dpp_greedy(P, q, K, sig)
        else:
            sel = dpp_greedy(P, q, K, None, ell=c * zz); sig = c * float(np.median(zz[sel])) if len(sel) else None
        f.sig.append(sig)
        return fill(ok[sel], len(u), K)
    f.sig = []
    return f


DPPR_PAST = (1, 2, 3)      # past frames used by the dppr track count
DPPR_POOL = 1000           # candidates matched per frame


def track_count(C, q, de, ok, tau, cache):
    """r_i = (m_i + 1) / (M + 1): m_i = past frames q-1..q-3 (M of them inside the flight) where candidate ok[i] has a mutual
    nearest-neighbour descriptor match with cosine >= tau among that frame's strongest DPPR_POOL candidates. No pose, no future."""
    pool = ok[:DPPR_POOL]; m = np.zeros(len(ok)); M = 0
    D1 = de[pool].astype(np.float32); D1 /= np.linalg.norm(D1, axis=1, keepdims=True) + 1e-9
    for L in DPPR_PAST:
        g = q - L
        if g < 0:
            continue
        M += 1
        if g not in cache:
            D2 = C.frame(g)[4][:DPPR_POOL].astype(np.float32); D2 /= np.linalg.norm(D2, axis=1, keepdims=True) + 1e-9
            cache[g] = D2
        D2 = cache[g]
        if not len(D2) or not len(pool):
            continue
        S = D1 @ D2.T; a = S.argmax(1); b = S.argmax(0)
        m[:len(pool)] += (b[a] == np.arange(len(pool))) & (S[np.arange(len(pool)), a] >= tau)
    while len(cache) > 8:
        cache.pop(min(cache))
    return (m + 1) / (M + 1)


def pick_dppr(sigma, alpha, beta, tau, depth_scaled=False):
    """dppr<sigma>a<alpha>b<beta>t<tau> (T4 step 3, SUB3, docs/t4_dppr_design.md): dpp with quality
    q_i = exp(-alpha * rank_i / 100) * r_i^beta, r_i = track_count over the PAST frames q-1..q-3 (descriptor matches, no pose).
    beta = 0 is dpp<sigma>a<alpha> exactly. ABLATION env KPBENCH_DPPR_QMODE: quality (beta 0) | rep (alpha 0) | both.
    Needs ctx=(Flight, Cands, frame) for the past frames only (needs_past, not an oracle) and de=descriptors.
    depth_scaled (dppmr<c>a<alpha>b<beta>t<tau>): sigma is c and the kernel scale is c * median depth of the frame's
    candidates with depth, as in dppm; beta = 0 is dppm<c>a<alpha> exactly. f.sig collects the per-frame scale."""
    mode = os.environ.get('KPBENCH_DPPR_QMODE', 'both')
    a_, b_ = (alpha, 0.0) if mode == 'quality' else (0.0, beta) if mode == 'rep' else (alpha, beta)
    caches = {}
    def f(u, v, s, z, K, ctx=None, de=None):
        F, C, q = ctx
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        w = np.exp(-a_ * np.arange(len(ok)) / 100.0)
        if b_ > 0:
            r = track_count(C, q, de, ok, tau, caches.setdefault(F.slug, {})); w = w * r ** b_
            f.rstd.append(float(r.std()))
        sig = sigma * float(np.median(z[ok])) if depth_scaled else sigma
        if depth_scaled:
            f.sig.append(sig)
        return fill(ok[dpp_greedy(to3d(u[ok], v[ok], z[ok]), w, K, sig)], len(u), K)
    f.needs_past = True; f.needs_desc = True; f.rstd = []; f.sig = []
    return f


def nms_order(u, v, r):
    """greedy NMS in score order: keep a point if no kept point lies within r px (grid hash, cell r)"""
    if r <= 0:
        return np.ones(len(u), bool)
    keep = np.zeros(len(u), bool); cells = {}
    for i in range(len(u)):
        cx, cy = int(u[i] // r), int(v[i] // r); ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((cx + dx, cy + dy), ()):
                    if (u[i] - u[j]) ** 2 + (v[i] - v[j]) ** 2 < r * r:
                        ok = False; break
                if not ok: break
            if not ok: break
        if ok:
            keep[i] = True; cells.setdefault((cx, cy), []).append(i)
    return keep


def pick_nms(r):
    """nms<r>: score-ranked top K after greedy NMS of radius r px over the candidates with depth (the plain NMS baseline of
    the spread class; nms0 = sd); fill by score from the rest with depth"""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        return fill(ok[nms_order(u[ok], v[ok], r)][:K], len(u), K, ok)
    return f


def pick_ssc(o, M):
    """ssc<o>[p<M>]: suppression via square covering (Bailo et al. 2018, the ANMS baseline of the spread class): among the
    strongest M candidates with depth, binary search on the square side w px so that greedy covering in score order (a kept
    point suppresses every weaker point inside its w x w square) keeps about o*K points; the K strongest of those are returned,
    fill by rank. o = 1: plain SSC; o > 1 trades spread for strength."""
    def keep_count(u, v, w):
        keep = np.zeros(len(u), bool); cells = {}
        for i in range(len(u)):
            cx, cy = int(u[i] // w), int(v[i] // w); ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in cells.get((cx + dx, cy + dy), ()):
                        if abs(u[i] - u[j]) < w / 2 and abs(v[i] - v[j]) < w / 2:
                            ok = False; break
                    if not ok: break
                if not ok: break
            if ok:
                keep[i] = True; cells.setdefault((cx, cy), []).append(i)
        return keep

    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)[:M]
        if not len(ok):
            return np.arange(min(K, len(u)))
        T = min(len(ok), int(round(o * K))); lo, hi = 1.0, float(max(W, H)); best = np.ones(len(ok), bool)
        for _ in range(14):                                   # largest w that still keeps >= T points
            w = 0.5 * (lo + hi); kp = keep_count(u[ok], v[ok], w)
            if kp.sum() >= T:
                lo, best = w, kp
            else:
                hi = w
            if hi - lo < 0.5:
                break
        return fill(ok[np.flatnonzero(best)[:K]], len(u), K)
    return f


def pick_upstream(m, r, N, inner):
    """up...: border margin m, NMS radius r, first N survivors, then inner; pad to K from the rest (depth first, score order).
    f.stats counts calls and padded frames."""
    def f(u, v, s, z, K):
        f.stats['calls'] += 1
        keep = (u >= m) & (u <= W - m) & (v >= m) & (v <= H - m)
        sub = np.flatnonzero(keep)
        sub = sub[nms_order(u[sub], v[sub], r)][:N]
        sel = sub[np.asarray(inner(u[sub], v[sub], s[sub], z[sub], K), int)] if len(sub) else np.array([], int)
        if len(sel) < min(K, len(u)):
            f.stats['padded'] += 1
            used = np.zeros(len(u), bool); used[sel] = True
            rest = [i for i in np.argsort(z >= DEPTH_MAX, kind='stable') if not used[i]]
            sel = np.concatenate([sel, np.asarray(rest[:K - len(sel)], int)])
        return sel
    f.stats = dict(calls=0, padded=0)
    return f


def pick_window(N, inner):
    """win<N>[+dpp..]: keep the strongest candidate of each N x N px window (R2D2-style local max, favours points that
    stand out in their neighbourhood), then inner on the survivors (none: score order); pad to K from the rest as up..."""
    def f(u, v, s, z, K):
        cell = (v // N).astype(np.int64) * (W // N + 2) + (u // N).astype(np.int64)
        sub = np.sort(np.unique(cell, return_index=True)[1])          # candidates are in score order: first per cell = max
        sel = sub[np.asarray(inner(u[sub], v[sub], s[sub], z[sub], K), int)] if inner and len(sub) else sub[:K]
        if len(sel) < min(K, len(u)):
            used = np.zeros(len(u), bool); used[sel] = True
            rest = [i for i in np.argsort(z >= DEPTH_MAX, kind='stable') if not used[i]]
            sel = np.concatenate([sel, np.asarray(rest[:K - len(sel)], int)])
        return sel
    return f


def parse(name):
    """picker name -> function. cube<size>; far2d|far3d[w][p<pool>], e.g. far3d, far2dw, far3dp400"""
    if name in PICKERS:
        return PICKERS[name]
    if re.fullmatch(r'evo_[A-Za-z0-9_]+', name):                       # code-search result: kpbench/evolved/<name>.py
        import importlib
        return importlib.import_module('kpbench.evolved.' + name).pick
    m = re.fullmatch(r'cube([0-9.]+)p([0-9]+)', name)
    if m:
        return pick_cube_pool(float(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r'pd([0-9.]+)(?:a([0-9.]+))?', name)
    if m:
        return pick_poisson_disk(float(m.group(1)), float(m.group(2) or 0))
    m = re.fullmatch(r'up(?:b([0-9]+))?(?:r([0-9]+))?\+(dpp[0-9.]+(?:a[0-9.]+)?)(?:p([0-9]+))?', name)
    if m:
        return pick_upstream(int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(4) or 2000), parse(m.group(3)))
    m = re.fullmatch(r'win([0-9]+)(?:\+(dpp[0-9.]+(?:a[0-9.]+)?))?', name)
    if m:
        return pick_window(int(m.group(1)), parse(m.group(2)) if m.group(2) else None)
    m = re.fullmatch(r'nms([0-9.]+)', name)
    if m:
        return pick_nms(float(m.group(1)))
    m = re.fullmatch(r'ssc([0-9.]+)?(?:p([0-9]+))?', name)
    if m:
        return pick_ssc(float(m.group(1) or 1.0), int(m.group(2) or 2000))
    m = re.fullmatch(r'dppr([0-9.]+)a([0-9.]+)b([0-9.]+)t([0-9.]+)', name)
    if m:
        return pick_dppr(*map(float, m.groups()))
    m = re.fullmatch(r'dppmr([0-9.]+)a([0-9.]+)b([0-9.]+)t([0-9.]+)', name)
    if m:
        return pick_dppr(*map(float, m.groups()), depth_scaled=True)
    m = re.fullmatch(r'dpps([0-9]+)([qle]?)', name)
    if m:
        return pick_dpp_slices(int(m.group(1)), m.group(2) or 'q')
    m = re.fullmatch(r'dpp([ml])([0-9.]+)(?:a([0-9.]+))?', name)
    if m:
        return pick_dpp_depth(float(m.group(2)), float(m.group(3) or 1.0), m.group(1))
    m = re.fullmatch(r'dpp2d([0-9.]+)(?:a([0-9.]+))?', name)
    if m:
        return pick_dpp(float(m.group(1)), float(m.group(2) or 1.0), '2d')
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
    m = re.fullmatch(r'seed1([do])(?:p([0-9]+))?', name)
    if m:
        return pick_seed1_far(m.group(1), int(m.group(2) or 400))
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
    m = re.fullmatch(r'cubem([0-9.]+)', name)
    if m:
        return pick_cube(float(m.group(1)), rel=True)
    if name.startswith('cube'):
        return pick_cube(float(name[4:]))
    m = re.fullmatch(r'far(2d|3d)(w?)(?:p(\d+))?', name)
    if m:
        return pick_far(m.group(1), int(m.group(3) or 2000), m.group(2) == 'w')
    raise KeyError('unknown picker %s' % name)
