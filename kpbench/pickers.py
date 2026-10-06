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
A picker returns candidate indices (at most K; fewer only when the frame has too few candidates).
To add a picker: write a function (u, v, s, z, K) -> indices and register it in PICKERS or in parse().
"""
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


PICKERS = {'score': pick_score, 'sd': pick_sd, 'grid': pick_grid, 'grid10x8': pick_grid10x8,
           'fps': pick_fps(True), 'fps0': pick_fps(False)}


def parse(name):
    if name in PICKERS:
        return PICKERS[name]
    if name.startswith('cube'):
        return pick_cube(float(name[4:]))
    raise KeyError('unknown picker %s' % name)
