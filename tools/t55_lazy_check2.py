"""SUB2 second-method check of T55 (dispatch 2026-11-24 13:15): timing, identity count, benchmark path unchanged.

  python tools/t55_lazy_check2.py [--every 3] [--out results/audit/t55_lazy_check2_2026-11-24.md]

Independent of tools/t55_lazy.py on purpose:
  timing    blocked loops (one picker and K over all timed frames, then the next) instead of interleaved calls; clock
            time.monotonic_ns instead of perf_counter; median of 5 calls per frame instead of the fastest of 3. Same frames
            (every --every-th query frame of the 6 laptop flights). PASS per K row if the lazy median is within 20 % of
            method 1's median (results/main/t55_lazy_time_raw.npz); cube0.5 compared the same way.
  identity  calls kpbench.pickers.dpp_greedy / dpp_greedy_lazy directly (not through parse), benchmark once at K400 and
            compared to lazy at K by its first K picks (nested greedy, asserted on 23 Nov), counted by SHA-1 digest per frame.
  path      pick_dpp, dpp_greedy, fill, to3d and the dpp<sigma> parse branch byte-identical to git HEAD~1 (before T55).
Tag [P] (laptop CPU). No pose and no fail number is read.
"""
import argparse, hashlib, inspect, os, subprocess, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from kpbench import pickers as P_                                              # noqa: E402
from kpbench.pickers import parse, to3d, dpp_greedy, dpp_greedy_lazy, DEPTH_MAX   # noqa: E402
from kpbench.data import slug                                                  # noqa: E402

KF = 10; KS = [50, 100, 200, 400]
FLIGHTS = 'hospital/P041,oldtown/P000,oldtown/P001,oldtown/P002,oldtown/P006,oldtown/P007'


def frames(flight):
    C = np.load(os.path.join(ROOT, 'cache', '%s_raco.npz' % slug(flight)))
    n, U, V, S, Z = (C[k] for k in 'nuvsz'); qs = [f for f in range(len(n)) if f % KF]
    return qs, lambda f: tuple(x[f, :int(n[f])].astype(np.float64) for x in (U, V, S, Z))


def h(a):
    return hashlib.sha1(np.ascontiguousarray(a, np.int64).tobytes()).hexdigest()


def path_check(rev='HEAD~1'):
    """names whose source differs from rev (empty = benchmark path unchanged)"""
    show = lambda f: subprocess.run(['git', 'show', '%s:kpbench/%s' % (rev, f)], cwd=ROOT, capture_output=True, text=True).stdout
    old = show('pickers.py'); src = {'to3d': show('data.py')}                  # to3d is imported from kpbench.data
    bad = [n for n in ('pick_dpp', 'dpp_greedy', 'fill', 'to3d') if inspect.getsource(getattr(P_, n)) not in src.get(n, old)]
    br = "    m = re.fullmatch(r'dpp([0-9.]+)(?:a([0-9.]+))?', name)\n    if m:\n        return pick_dpp(float(m.group(1)), float(m.group(2) or 1.0))\n"
    if br not in old or br not in inspect.getsource(P_.parse):
        bad.append('parse dpp branch')
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--flights', default=FLIGHTS); ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--out', default=None); a = ap.parse_args(argv); fl = a.flights.split(',')
    nq = 0; diff = {K: 0 for K in KS}
    for flight in fl:                                                           # identity, every query frame
        qs, fr = frames(flight)
        for f in qs:
            u, v, s, z = fr(f); ok = np.flatnonzero(z < DEPTH_MAX); nq += 1
            if not len(ok):
                continue
            P = to3d(u[ok], v[ok], z[ok]); q = np.exp(-np.arange(len(ok)) / 100.0); ref = dpp_greedy(P, q, 400, 0.5)
            for K in KS:
                diff[K] += h(ref[:K]) != h(dpp_greedy_lazy(P, q, K, 0.5))
    X = []
    for flight in fl:
        qs, fr = frames(flight); X += [fr(f) for f in qs[::a.every]]
    pk = {'dpplazy0.5': parse('dpplazy0.5'), 'cube0.5': parse('cube0.5')}; T = {}
    for p in pk:                                                                # timing, blocked
        for K in KS:
            pk[p](*X[0], K); t = []
            for x in X:
                r = []
                for _ in range(5):
                    t0 = time.monotonic_ns(); pk[p](*x, K); r.append(time.monotonic_ns() - t0)
                t.append(np.median(r) / 1e6)
            T[(p, K)] = float(np.median(t))
    R = np.load(os.path.join(ROOT, 'results', 'main', 't55_lazy_time_raw.npz'))
    bad = path_check()
    L = ['Identity (second method, direct greedy calls, SHA-1 per frame, benchmark K400 prefix): %d query frames; frames differing '
         'K50 / K100 / K200 / K400 = %s.' % (nq, ' / '.join('%d' % diff[K] for K in KS)),
         'Benchmark path vs git HEAD~1: %s.' % ('unchanged (pick_dpp, dpp_greedy, fill, to3d, parse dpp branch byte-identical)' if not bad
                                               else 'CHANGED: ' + ', '.join(bad)), '',
         'Timing (second method: blocked loops, monotonic_ns, median of 5) on the same %d frames, ms per frame median:' % len(X), '',
         '| K | dpplazy0.5 method 1 | method 2 | ratio | cube0.5 method 1 | method 2 | ratio | verdict (both within 20 %) |', '|---|---|---|---|---|---|---|---|']
    for K in KS:
        m1l, m1c = float(np.median(R['dpplazy0.5_K%d' % K])), float(np.median(R['cube0.5_K%d' % K]))
        rl, rc = T[('dpplazy0.5', K)] / m1l, T[('cube0.5', K)] / m1c
        L.append('| %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %s |' % (K, m1l, T[('dpplazy0.5', K)], rl, m1c, T[('cube0.5', K)], rc,
                 'PASS' if abs(rl - 1) <= 0.2 and abs(rc - 1) <= 0.2 else 'FAIL'))
    txt = '\n'.join(L) + '\n'; print(txt)
    if a.out:
        open(os.path.join(ROOT, a.out), 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    main()
