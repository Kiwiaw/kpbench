"""Same-picks re-time of the dpp0.5 selection step (MAIN, objection N3 'selection cost'; dispatch 2026-11-22 16:25 / 2026-11-23).

  python tools/n3_retime.py check [--flights a,b,c] [--jobs 6]          # identical picks on every query frame, K = 400 prefix
  python tools/n3_retime.py time  [--flights a,b,c] [--every 10] [--reps 3] [--out results/main/n3_retime.md]

kpbench.pickers.dpp_greedy already is the fast greedy MAP of Chen, Zhang, Zhou (NeurIPS 2018): one Cholesky row per pick.
Two re-implementations of the SAME greedy are timed against it; neither changes the objective or the kernel:
  pre   the same loop with the Cholesky rows in one preallocated (K, n) array (the repo version re-stacks them every pick).
  lazy  pre + lazy candidate prefix (the lazy idea of Minoux 1978 / Hemmi et al. NeurIPS 2022, here in closed form):
        the gain of candidate i never exceeds q_i^2 and q_i = exp(-alpha rank_i / 100) falls with rank, so while the best
        gain inside the first m candidates is >= q_m^2 no later candidate can be the argmax; rows are kept for the first m
        only and m doubles when the bound is crossed (the missing row entries are then filled by the same recurrence).
        Toy: best gain inside the prefix 0.02, q_m^2 = exp(-2 * 256 / 100) = 0.006 -> candidates 256.. cannot win, skip them.
`check` asserts the ordered pick list of pre and lazy equals the repo picker's on every query frame at K = 400; the greedy
is nested (the first K picks of a longer run are the run for K), which is asserted on the lists for K = 50, 100, 200.
`time` times the picker call only (as tools/select_time.py): pickers and K interleaved per query, fastest of --reps.
No pose and no fail number is read: tag [P] (laptop CPU).
"""
import argparse, os, platform, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from kpbench.pickers import parse, fill, to3d, DEPTH_MAX                       # noqa: E402
from kpbench.data import slug                                                  # noqa: E402

KF = 10; SIGMA = 0.5; ALPHA = 1.0; KS = [50, 100, 200, 400]
FLIGHTS = 'hospital/P041,oldtown/P000,oldtown/P001,oldtown/P002,oldtown/P006,oldtown/P007'


def greedy_pre(P, q, K, sigma):
    """dpp_greedy with preallocated rows; same arithmetic per pick"""
    n = len(P); K = min(K, n); d2 = q ** 2; C = np.zeros((K, n)); sel = []
    for k in range(K):
        j = int(np.argmax(d2))
        if d2[j] <= 1e-12:
            break
        dj = np.sqrt(d2[j]); Lj = q[j] * q * np.exp(-np.sum((P - P[j]) ** 2, 1) / (2 * sigma ** 2))
        e = (Lj - (C[:k, j] @ C[:k] if k else 0.0)) / dj; C[k] = e; d2 = d2 - e ** 2; d2[j] = -np.inf; sel.append(j)
    return np.array(sel, int)


def greedy_lazy(P, q, K, sigma, m0=128, stat=None):
    """dpp_greedy on a lazily grown candidate prefix (see module docstring); q must be non-increasing"""
    n = len(P); K = min(K, n); q2 = q ** 2; d2 = q2.copy(); C = np.zeros((K, n)); sel = []; dj = []; m = min(n, m0)
    for k in range(K):
        while True:
            j = int(np.argmax(d2[:m]))
            if m == n or d2[j] >= q2[m]:                     # ties: argmax keeps the lower index, as the full run does
                break
            m2 = min(n, 2 * m); B = slice(m, m2)             # bring candidates m..m2 up to date with the k picks so far
            for t, jt in enumerate(sel):
                Lt = q[jt] * q[B] * np.exp(-np.sum((P[B] - P[jt]) ** 2, 1) / (2 * sigma ** 2))
                e = (Lt - (C[:t, jt] @ C[:t, B] if t else 0.0)) / dj[t]; C[t, B] = e; d2[B] = d2[B] - e ** 2
            m = m2
        if d2[j] <= 1e-12:
            break
        d = np.sqrt(d2[j]); Lj = q[j] * q[:m] * np.exp(-np.sum((P[:m] - P[j]) ** 2, 1) / (2 * sigma ** 2))
        e = (Lj - (C[:k, j] @ C[:k, :m] if k else 0.0)) / d; C[k, :m] = e; d2[:m] = d2[:m] - e ** 2; d2[j] = -np.inf
        sel.append(j); dj.append(d)
    if stat is not None:
        stat.append(m)
    return np.array(sel, int)


def picker(greedy, **kw):
    """the pick_dpp shell (depth filter, 3D points, rank quality, top-up) around a greedy"""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        P = to3d(u[ok], v[ok], z[ok]); q = np.exp(-ALPHA * np.arange(len(ok)) / 100.0)
        return fill(ok[greedy(P, q, K, SIGMA, **kw)], len(u), K, ok)
    return f


def frames(flight, det='raco'):
    C = np.load(os.path.join(ROOT, 'cache', '%s_%s.npz' % (slug(flight), det)))
    n, U, V, S, Z = (C[k] for k in 'nuvsz'); qs = [f for f in range(len(n)) if f % KF]
    return qs, lambda f: tuple(x[f, :int(n[f])].astype(np.float64) for x in (U, V, S, Z))


def check_flight(flight):
    """(flight, queries, frames with any differing list, min / median / max candidates, median final prefix per K)"""
    qs, fr = frames(flight); ref = parse('dpp%g' % SIGMA); pre = picker(greedy_pre); bad = 0; nc = []; M = {K: [] for K in KS}
    for f in qs:
        a = fr(f); nc.append(len(a[0])); r = ref(*a, KS[-1]); ok = np.array_equal(r, pre(*a, KS[-1]))
        for K in KS:
            ok &= np.array_equal(picker(greedy_lazy, stat=M[K])(*a, K), ref(*a, K) if K == 100 else r[:K])
        bad += not ok
    return flight, len(qs), bad, min(nc), float(np.median(nc)), max(nc), {K: float(np.median(M[K])) for K in KS}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['check', 'time']); ap.add_argument('--flights', default=FLIGHTS)
    ap.add_argument('--jobs', type=int, default=6); ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--reps', type=int, default=3); ap.add_argument('--out', default=None)
    a = ap.parse_args(argv); fl = a.flights.split(',')
    if a.mode == 'check':
        with ProcessPoolExecutor(a.jobs) as ex:
            R = list(ex.map(check_flight, fl))
        L = ['| flight | queries | frames with a differing pick list (pre or lazy vs repo, K 50-400) | candidates min / median / max | '
             'median final prefix m at K 50 / 100 / 200 / 400 |', '|---|---|---|---|---|']
        L += ['| %s | %d | %d | %d / %d / %d | %s |' % (f, nq, bad, lo, md, hi, ' / '.join('%d' % m[K] for K in KS))
              for f, nq, bad, lo, md, hi, m in R]
        L.append('| total | %d | %d | | |' % (sum(r[1] for r in R), sum(r[2] for r in R)))
        txt = '\n'.join(L) + '\n'
    else:
        pk = {'repo': parse('dpp%g' % SIGMA), 'pre': picker(greedy_pre), 'lazy': picker(greedy_lazy), 'cube0.5': parse('cube0.5')}
        cells = [(p, K) for K in KS for p in pk]; T = {c: [] for c in cells}; nq = 0
        for flight in fl:
            qs, fr = frames(flight)
            for p, K in cells:
                pk[p](*fr(qs[0]), K)                                            # warm-up, not timed
            for f in qs[::a.every]:
                x = fr(f); nq += 1
                for p, K in cells:
                    best = np.inf
                    for _ in range(a.reps):
                        t0 = time.perf_counter(); pk[p](*x, K); best = min(best, time.perf_counter() - t0)
                    T[(p, K)].append(1e3 * best)
        med = {c: float(np.median(T[c])) for c in cells}
        L = ['%d queries (every %dth query of %d flights), fastest of %d repeats, median ms per frame (90th percentile in brackets). '
             'Machine: %s, %d logical cores, numpy %s, Python %s.' % (nq, a.every, len(fl), a.reps, platform.processor() or platform.machine(),
                                                                      os.cpu_count(), np.__version__, platform.python_version()), '',
             '| K | dpp0.5 repo | dpp0.5 pre | dpp0.5 lazy | cube0.5 | repo / lazy | lazy / cube0.5 | repo / cube0.5 |', '|---|---|---|---|---|---|---|---|']
        for K in KS:
            L.append('| %d | ' % K + ' | '.join('%.2f (%.2f)' % (med[(p, K)], np.percentile(T[(p, K)], 90)) for p in pk)
                     + ' | %.1fx | %.1fx | %.1fx |' % (med[('repo', K)] / med[('lazy', K)], med[('lazy', K)] / med[('cube0.5', K)],
                                                       med[('repo', K)] / med[('cube0.5', K)]))
        txt = '\n'.join(L) + '\n'
    print(txt)
    if a.out:
        open(os.path.join(ROOT, a.out), 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    main()
