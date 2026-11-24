"""T55 (MAIN, N3 follow-up, dispatch 2026-11-24 13:15): identity and timing of the opt-in picker dpplazy0.5 vs the benchmark dpp0.5.

  python tools/t55_lazy.py check [--jobs 6] [--out results/main/t55_lazy_identity.md]   # every query frame, K 50/100/200/400
  python tools/t55_lazy.py time  [--every 3] [--reps 3] [--out results/main/t55_lazy_time.md]

check: ordered pick lists of parse('dpplazy0.5') and parse('dpp0.5') (unchanged benchmark path) compared with np.array_equal on
every query frame (f % 10 != 0) of the laptop flights, each K run on its own (no nested-prefix shortcut).
time: picker call only, as tools/n3_retime.py (pickers and K interleaved per query, fastest of --reps, one warm-up per flight);
every --every-th query (3 -> ~1330 frames, vs 201 on 23 Nov). The repo dpp0.5 is timed at K100 only (30 ms per call).
No pose and no fail number is read: tag [P] (laptop CPU).
"""
import argparse, os, platform, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from kpbench.pickers import parse                                              # noqa: E402
from kpbench.data import slug                                                  # noqa: E402

KF = 10; KS = [50, 100, 200, 400]
FLIGHTS = 'hospital/P041,oldtown/P000,oldtown/P001,oldtown/P002,oldtown/P006,oldtown/P007'


def frames(flight, det='raco'):
    """query frame ids and a frame -> (u, v, s, z) reader from the laptop cache"""
    C = np.load(os.path.join(ROOT, 'cache', '%s_%s.npz' % (slug(flight), det)))
    n, U, V, S, Z = (C[k] for k in 'nuvsz'); qs = [f for f in range(len(n)) if f % KF]
    return qs, lambda f: tuple(x[f, :int(n[f])].astype(np.float64) for x in (U, V, S, Z))


def check_flight(flight):
    """(flight, queries, differing frames per K, candidates min / median / max)"""
    qs, fr = frames(flight); ref, lz = parse('dpp0.5'), parse('dpplazy0.5'); bad = {K: 0 for K in KS}; nc = []
    for f in qs:
        a = fr(f); nc.append(len(a[0]))
        for K in KS:
            bad[K] += not np.array_equal(ref(*a, K), lz(*a, K))
    return flight, len(qs), bad, min(nc), float(np.median(nc)), max(nc)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['check', 'time']); ap.add_argument('--flights', default=FLIGHTS)
    ap.add_argument('--jobs', type=int, default=6); ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--reps', type=int, default=3); ap.add_argument('--out', default=None)
    a = ap.parse_args(argv); fl = a.flights.split(',')
    if a.mode == 'check':
        with ProcessPoolExecutor(a.jobs) as ex:
            R = list(ex.map(check_flight, fl))
        L = ['| flight | queries | frames differing K50 / K100 / K200 / K400 (dpplazy0.5 vs dpp0.5) | candidates min / median / max |',
             '|---|---|---|---|']
        L += ['| %s | %d | %s | %d / %d / %d |' % (f, nq, ' / '.join('%d' % b[K] for K in KS), lo, md, hi) for f, nq, b, lo, md, hi in R]
        L.append('| total | %d | %s | |' % (sum(r[1] for r in R), ' / '.join('%d' % sum(r[2][K] for r in R) for K in KS)))
    else:
        pk = {'dpplazy0.5': parse('dpplazy0.5'), 'cube0.5': parse('cube0.5'), 'dpp0.5': parse('dpp0.5')}
        cells = [(p, K) for K in KS for p in pk if p != 'dpp0.5' or K == 100]; T = {c: [] for c in cells}; nq = 0
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
        np.savez(os.path.join(ROOT, 'results', 'main', 't55_lazy_time_raw.npz'), **{'%s_K%d' % c: np.array(T[c]) for c in cells})
        q = {c: np.percentile(T[c], [10, 50, 90]) for c in cells}
        L = ['%d query frames (every %dth query of %d flights), fastest of %d repeats, ms per frame: median [10th, 90th percentile]. '
             'Machine: %s, %d logical cores, numpy %s, Python %s. Tag [P] (laptop).' % (nq, a.every, len(fl), a.reps,
             platform.processor() or platform.machine(), os.cpu_count(), np.__version__, platform.python_version()), '',
             '| K | dpp0.5 benchmark path | dpplazy0.5 | cube0.5 | benchmark / lazy | lazy / cube0.5 |', '|---|---|---|---|---|---|']
        f3 = lambda v: '%.2f [%.2f, %.2f]' % (v[1], v[0], v[2])
        for K in KS:
            b = q.get(('dpp0.5', K))
            L.append('| %d | %s | %s | %s | %s | %.1fx |' % (K, f3(b) if b is not None else 'not timed', f3(q[('dpplazy0.5', K)]),
                     f3(q[('cube0.5', K)]), '%.1fx' % (b[1] / q[('dpplazy0.5', K)][1]) if b is not None else '-',
                     q[('dpplazy0.5', K)][1] / q[('cube0.5', K)][1]))
    txt = '\n'.join(L) + '\n'; print(txt)
    if a.out:
        open(os.path.join(ROOT, a.out), 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    main()
