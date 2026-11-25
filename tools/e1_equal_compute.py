"""E1 equal-compute check on FRESH-1 flat frames from mirrored rows (T20 step E1, dispatch 2026-11-25 13:15 MAIN). [P] post hoc.

  python tools/e1_equal_compute.py [--boot 10000] [--seed 0] [--out results/main/e1_equal_compute.md]

Question: cube0.5 selection time is flat in K (about 1.8 ms at K50-400, results/main/t55_lazy_time.md), dpp0.5 at K100 costs
2.2-2.5x that, so at equal SELECTION time cube0.5 can run at any K <= 400. Does cube0.5 at K400 fail fewer flat frames than
dpp0.5 at K100 on the same frames? Per flight d_f = cube K400 fails - dpp K100 fails (counts); pooled points = 100 sum d_f / n_flat;
flight-cluster CI (flights resampled, ratio of sums); exact two-sided sign-flip p over non-zero d_f. Fail = no pose, > 25 cm or
> 2 deg (tools/class_bootstrap.per_flight_fail); flat = plane_frac >= 0.85. Same rows and fail rule as tools/k_interaction.py.
"""
import argparse, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from class_bootstrap import per_flight_fail
from k_interaction import RUNS, signflip_p
from kpbench.gate import load_run, labels, split_info
from kpbench.splits import FLAT

PAIRS = [('cube0.5 K400 - dpp0.5 K100', 'c400', 'i100'), ('cube0.5 K100 - dpp0.5 K100', 'c100', 'i100'),
         ('cube0.5 K400 - dpp0.5 K400', 'c400', 'i400')]


def pair_stats(T, n, rng, boot):
    d = np.array(T); pt = 100 * d.sum() / n.sum(); bs = []
    for _ in range(boot):
        i = rng.integers(0, len(d), len(d)); bs.append(100 * d[i].sum() / n[i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return int(d.sum()), pt, lo, hi, int((d < 0).sum()), int((d > 0).sum()), signflip_p(d.tolist())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=10000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'main', 'e1_equal_compute.md')); a = ap.parse_args()
    flights = split_info('fresh')[0]; R = {}
    for k, run in RUNS.items():
        rs = 'runs' if os.path.isdir(os.path.join(ROOT, 'results', 'runs', run)) else 'runs_gate'
        R[k] = load_run(run, flights, rs)[1]
    F = {}
    for f in flights:
        q = [r[1] for r in R['c400'][f][0]]
        assert all([r[1] for r in R[k][f][0]] == q for k in RUNS), f
        c4, i1, n = per_flight_fail(R['c400'][f][0], R['i100'][f][0], labels(f), FLAT)
        c1, _, n2 = per_flight_fail(R['c100'][f][0], R['i100'][f][0], labels(f), FLAT)
        _, i4, n3 = per_flight_fail(R['c400'][f][0], R['i400'][f][0], labels(f), FLAT); assert n == n2 == n3
        if n: F[f] = dict(n=n, c400=c4, i100=i1, c100=c1, i400=i4)
    n = np.array([v['n'] for v in F.values()]); N = int(n.sum())
    L = ['# E1 equal-compute check, cube0.5 vs dpp0.5, FRESH-1 flat (%s) [P]' % time.strftime('%Y-%m-%d %H:%M'), '',
         'Rows: %s. %d flights with flat frames, %d flat. Script: tools/e1_equal_compute.py (boot %d, seed %d). Sign: first minus second, negative = first fails less.' % (
             ', '.join(RUNS.values()), len(F), N, a.boot, a.seed), '',
         'Flat fails of %d: cube0.5 K100 %d, K400 %d; dpp0.5 K100 %d, K400 %d.' % (N, *(sum(v[k] for v in F.values()) for k in ('c100', 'c400', 'i100', 'i400'))), '',
         '| pair | frames | points | flight-cluster 95 % CI | flights neg / pos / zero | exact sign-flip p2 |', '|---|---|---|---|---|---|']
    rng = np.random.default_rng(a.seed)
    for name, x, y in PAIRS:
        s, pt, lo, hi, neg, pos, p = pair_stats([v[x] - v[y] for v in F.values()], n, rng, a.boot)
        L.append('| %s | %+d | %+.2f | [%+.2f, %+.2f] | %d / %d / %d | %.4g |' % (name, s, pt, lo, hi, neg, pos, len(F) - neg - pos, p))
    L += ['', '| flight | n flat | cube K100 / K400 | dpp K100 / K400 |', '|---|---|---|---|']
    L += ['| %s | %d | %d / %d | %d / %d |' % (f, v['n'], v['c100'], v['c400'], v['i100'], v['i400']) for f, v in F.items()]
    open(a.out, 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L)); print('->', a.out)


if __name__ == '__main__':
    main()
