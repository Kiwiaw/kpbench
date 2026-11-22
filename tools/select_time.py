"""Selection-step time per frame (MAIN, dispatch 2026-11-21 13:40; objection N2 'runtime not measured').

  python tools/select_time.py [--flight hospital/P041] [--det raco] [--pickers dpp0.5,cube0.5] [--K 50,100,200,400]
                              [--reps 3] [--out results/main/select_time_2026-11-22.md]

Times ONLY pk(u, v, s, z, K) (the picker call of evaluate.checked_pick) on the cached candidates of the flight's queries
(frames with f % 10 != 0), no detector, no matching, no pose. Pickers and K values are interleaved per query, so every
(picker, K) cell sees the same frames under the same machine load; each call is repeated --reps times and the fastest
repeat is kept (the usual way to remove scheduler noise). Reported: median and 90th percentile over the queries, in ms.
Toy: three queries taking 2, 3 and 10 ms -> median 3 ms. No fail number and no pose is read: tag [P] (one flight, laptop CPU).
"""
import argparse, os, platform, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from kpbench.pickers import parse, DEPTH_MAX                                   # noqa: E402
from kpbench.data import slug                                                  # noqa: E402

KF = 10


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--flight', default='hospital/P041'); ap.add_argument('--det', default='raco')
    ap.add_argument('--pickers', default='dpp0.5,cube0.5'); ap.add_argument('--K', default='50,100,200,400')
    ap.add_argument('--reps', type=int, default=3); ap.add_argument('--out', default=None)
    a = ap.parse_args(argv); names = a.pickers.split(','); Ks = [int(k) for k in a.K.split(',')]
    C = np.load(os.path.join(ROOT, 'cache', '%s_%s.npz' % (slug(a.flight), a.det)))
    n, U, V, S, Z = (C[k] for k in 'nuvsz'); qs = [f for f in range(len(n)) if f % KF]
    fr = lambda f: tuple(x[f, :int(n[f])].astype(np.float64) for x in (U, V, S, Z))
    pk = {p: parse(p) for p in names}; cells = [(p, K) for K in Ks for p in names]
    for p, K in cells:                                                          # warm-up (imports, caches), not timed
        pk[p](*fr(qs[0]), K)
    T = {c: np.empty(len(qs)) for c in cells}; ncand = []; ndepth = []
    for i, f in enumerate(qs):
        u, v, s, z = fr(f); ncand.append(len(u)); ndepth.append(int((z < DEPTH_MAX).sum()))
        for p, K in cells:
            best = np.inf
            for _ in range(a.reps):
                t0 = time.perf_counter(); idx = pk[p](u, v, s, z, K); best = min(best, time.perf_counter() - t0)
            assert len(idx) == min(K, len(u)), (p, K, f, len(idx))
            T[(p, K)][i] = 1e3 * best
    L = ['# Selection time per frame, %s (%s candidates, laptop CPU) [P]' % (a.flight, a.det), '',
         'tools/select_time.py: picker call only (no detector, matcher or pose), %d queries, fastest of %d repeats per query, pickers and K '
         'interleaved per query. Candidates per query: median %d (min %d, max %d), with depth median %d. Machine: %s, %d logical cores, '
         'numpy %s, Python %s.' % (len(qs), a.reps, np.median(ncand), min(ncand), max(ncand), np.median(ndepth),
                                   platform.processor() or platform.machine(), os.cpu_count(), np.__version__, platform.python_version()), '',
         '| K | ' + ' | '.join('%s median ms | %s p90 ms' % (p, p) for p in names) + ' | ratio of medians %s / %s |' % (names[0], names[-1]),
         '|---|' + '---|' * (2 * len(names) + 1)]
    for K in Ks:
        med = [float(np.median(T[(p, K)])) for p in names]
        L.append('| %d | ' % K + ' | '.join('%.2f | %.2f' % (m, np.percentile(T[(p, K)], 90)) for p, m in zip(names, med))
                 + ' | %.1fx |' % (med[0] / med[-1]))
    txt = '\n'.join(L) + '\n'; print(txt)
    if a.out:
        open(os.path.join(ROOT, a.out), 'w', encoding='utf-8').write(txt)
    return T


if __name__ == '__main__':
    main()
