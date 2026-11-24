"""T33 follow-up (SUB3, dispatch 2026-11-24 13:15): replay identity and selection time of dppr0.317a1.6b2.01t0.9 on the laptop flights.

  python tools/t33_dppr_replay.py [--every 3] [--reps 3] [--out results/sub3/dppr_replay_2026-11-24.md]

dppr reads past frames (track_count over q-1..q-3 descriptors, cached per flight), so a fresh call on one frame is not obviously
the benchmark's output. Three identity checks per query frame (f % 10 != 0), all at K100:
  A  sequential pass in frame order with one picker instance (as kpbench.bench calls it) vs a NEW picker instance per frame
     (cold cache): equal lists = the cache holds no state beyond the past-frame descriptors.
  B  A's sequential lists vs a second sequential pass in reverse frame order (cache filled in another order).
  C  lazy prefix: dppr quality w_i = exp(-a rank_i / 100) * r_i^b is NOT sorted by rank, so candidates are re-ordered by w
     (stable, descending), kpbench.pickers.dpp_greedy_lazy runs on the sorted list and picks are mapped back; vs A.
No stored dppr picks exist on the laptop (the FRESH dppr run has has_picks False and other flights), so a check against stored
Euler picks is owed with the DEV dump. Time: picker call only (track_count included), fastest of --reps, every --every-th query,
dppr benchmark path, dppr lazy, cube0.5. Tag [P] (laptop CPU). No pose and no fail number is read.
"""
import argparse, os, platform, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from kpbench import pickers as pk_                                             # noqa: E402
from kpbench.pickers import parse, fill, to3d, track_count, dpp_greedy_lazy, DEPTH_MAX   # noqa: E402
from kpbench.cache import Cands                                                # noqa: E402
from kpbench.data import slug                                                  # noqa: E402

NAME = 'dppr0.317a1.6b2.01t0.9'; SIG, ALPHA, BETA, TAU = 0.317, 1.6, 2.01, 0.9; KF = 10; K = 100
FLIGHTS = 'hospital/P041,oldtown/P000,oldtown/P001,oldtown/P002,oldtown/P006,oldtown/P007'


class F_:
    """the flight fields dppr reads (slug only)"""
    def __init__(self, s):
        self.slug = s


def dppr_lazy():
    """dppr0.317a1.6b2.01t0.9 with the lazy prefix on quality-sorted candidates (same objective and kernel)"""
    caches = {}
    def f(u, v, s, z, K, ctx=None, de=None):
        F, C, q = ctx
        ok = np.flatnonzero(z < DEPTH_MAX)
        if not len(ok):
            return np.arange(min(K, len(u)))
        w = np.exp(-ALPHA * np.arange(len(ok)) / 100.0) * track_count(C, q, de, ok, TAU, caches.setdefault(F.slug, {})) ** BETA
        o = np.argsort(-w, kind='stable'); P = to3d(u[ok], v[ok], z[ok])
        return fill(ok[o[dpp_greedy_lazy(P[o], w[o], K, SIG)]], len(u), K, ok)
    return f


def run(pick, F, C, frames):
    """frame -> pick list, frames called in the given order with one picker instance"""
    out = {}
    for f in frames:
        u, v, s, z, de = C.frame(f); out[f] = np.asarray(pick(u, v, s, z, K, ctx=(F, C, f), de=de))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--flights', default=FLIGHTS); ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--reps', type=int, default=3); ap.add_argument('--out', default=None)
    a = ap.parse_args(argv); L = ['| flight | queries | A: differ, sequential vs fresh instance | B: differ, forward vs reverse order | '
                                  'C: differ, lazy-sorted vs benchmark |', '|---|---|---|---|---|']
    tot = np.zeros(4, int); T = {p: [] for p in ('dppr benchmark', 'dppr lazy', 'cube0.5')}; cube = parse('cube0.5')
    for fl in a.flights.split(','):
        s = slug(fl); C = Cands(s, 'raco'); F = F_(s); qs = [f for f in range(C.N) if f % KF]
        A = run(parse(NAME), F, C, qs)
        fresh = {f: run(parse(NAME), F, C, [f])[f] for f in qs}
        rev = run(parse(NAME), F, C, qs[::-1]); lz = run(dppr_lazy(), F, C, qs)
        d = [sum(not np.array_equal(A[f], X[f]) for f in qs) for X in (fresh, rev, lz)]
        tot += [len(qs)] + d; L.append('| %s | %d | %d | %d | %d |' % (fl, len(qs), *d))
        pb, pl = parse(NAME), dppr_lazy()                                       # timing: frame order, warm past-frame cache
        for f in qs:
            u, v, s_, z, de = C.frame(f); kw = dict(ctx=(F, C, f), de=de)
            timed = f in qs[::a.every]
            for name, p, extra in (('dppr benchmark', pb, kw), ('dppr lazy', pl, kw), ('cube0.5', cube, {})):
                best = np.inf
                for _ in range(a.reps if timed else 1):
                    t0 = time.perf_counter(); p(u, v, s_, z, K, **extra); best = min(best, time.perf_counter() - t0)
                if timed:
                    T[name].append(1e3 * best)
    L.append('| total | %d | %d | %d | %d |' % tuple(tot))
    q = {p: np.percentile(T[p], [10, 50, 90]) for p in T}
    L += ['', '%d timed query frames (every %dth), K100, fastest of %d repeats, ms per frame median [10th, 90th percentile]; '
          'Machine: %s, %d logical cores, numpy %s. Tag [P] (laptop).' % (len(T['cube0.5']), a.every, a.reps,
          platform.processor() or platform.machine(), os.cpu_count(), np.__version__), '',
          '| picker | ms per frame | / cube0.5 |', '|---|---|---|']
    L += ['| %s | %.2f [%.2f, %.2f] | %.1fx |' % (p, v[1], v[0], v[2], v[1] / q['cube0.5'][1]) for p, v in q.items()]
    txt = '\n'.join(L) + '\n'; print(txt)
    if a.out:
        open(os.path.join(ROOT, a.out), 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    main()
