"""Power of the pre-registered FRESH-2 endpoint dpp0.5 vs cube0.5 K100 flat (MAIN, dispatch 2026-11-19T13:15).

  python tools/power_fresh2.py [--reps 20000] [--seed 0]

Test simulated = results/main/fresh2_k100_endpoint_prereg.md: per flight d_f = dpp0.5 - cube0.5 flat fails, exact ONE-sided
sign-flip on d_f (zeros dropped), reject at p < 0.05. A future set is simulated by drawing whole flights with replacement
from flights already run (FRESH-1 13 flights, board 27 flights, or both) until the flat count reaches the target; power =
share of simulated sets that reject. Toy: flights d = (-5, -3, +1): 8 patterns, sum <= -7 in 2 -> p 0.25, no reject.
'two-stage' also redraws the queries inside each drawn flight (new frames of a similar flight), so a flight's d_f is not
copied exactly; 'gap x 0.5' halves each flight's true gap first (the observed gap is a noisy, possibly lucky estimate),
'gap x 0.0' removes it (rows must then reject about 5 % or less: check of the simulator). The sums come from the row files via tools/j7_lofo_check.flight_counts.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j7_lofo_check import SETS, flight_counts


def p1_exact(d):
    """exact one-sided sign-flip p: share of the 2^m sign patterns of the nonzero d with sum <= observed sum (count convolution)"""
    x = [abs(int(v)) for v in d if v]
    if not x:
        return 1.0
    S = sum(x); c = np.zeros(2 * S + 1); c[S] = 1.0
    for v in x:
        n = np.zeros_like(c); n[v:] += c[:-v]; n[:-v] += c[v:]; c = n
    return float(c[:S + int(np.sum(d)) + 1].sum() / c.sum())


def power(F, target, reps, rng, two_stage=False, shrink=0.0):
    """F = array rows (n flat, only-dpp fails, only-cube fails); -> (power, mean flights drawn, mean nonzero flights)"""
    hit = 0; nf = 0; nz = 0; mean_n = F[:, 0].mean()
    for _ in range(reps):
        idx = rng.integers(0, len(F), int(3 * target / mean_n) + 8)
        k = int(np.searchsorted(np.cumsum(F[idx, 0]), target)) + 1
        assert k <= len(idx)
        G = F[idx[:k]]
        if two_stage:
            pa, pb = G[:, 1] / G[:, 0], G[:, 2] / G[:, 0]; pb = pb - shrink * (pb - pa)   # shrink 0.5 = half the observed gap in every flight
            a = rng.binomial(G[:, 0], pa); b = rng.binomial(G[:, 0] - a, np.minimum(pb / np.maximum(1 - pa, 1e-9), 1))
            d = a - b
        else:
            d = G[:, 1] - G[:, 2]
        hit += p1_exact(d) < 0.05; nf += k; nz += int((d != 0).sum())
    return hit / reps, nf / reps, nz / reps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=20000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--targets', default='2075,2774,4150,4849,6000,8000,12000,16000,24000'); a = ap.parse_args()
    src = {}
    for name, (ra, rb, _) in SETS.items():
        C = flight_counts(ra, rb, pairs=True)
        src[name] = np.array([(n, oa, fb - fa + oa) for n, fa, fb, oa in C.values()], dtype=np.int64)
    src['pooled'] = np.vstack([src['fresh1'], src['board']])
    T = [int(t) for t in a.targets.split(',')]
    for name, F in src.items():
        d = F[:, 1] - F[:, 2]
        print('%s: %d flights, flat %d, d sum %+d (%+.2f pt), one-sided exact p %.4f, flights better/tie/worse %d/%d/%d'
              % (name, len(F), F[:, 0].sum(), d.sum(), 100 * d.sum() / F[:, 0].sum(), p1_exact(d), (d < 0).sum(), (d == 0).sum(), (d > 0).sum()))
    print('| source | resampling | ' + ' | '.join('%d flat' % t for t in T) + ' | flights at 2774 / 4849 |')
    print('|---|---|' + '---|' * (len(T) + 1))
    for name, F in src.items():
        for ts, sh in ((False, 0.0), (True, 0.0), (True, 0.5), (True, 1.0)):
            rng = np.random.default_rng(a.seed); R = [power(F, t, a.reps, rng, ts, sh) for t in T]
            print('| %s | %s | ' % (name, ('two-stage' + (', gap x %.1f' % (1 - sh) if sh else '')) if ts else 'flights') + ' | '.join('%.1f' % (100 * r[0]) for r in R)
                  + ' | %.0f / %.0f |' % (R[T.index(2774)][1], R[T.index(4849)][1]), flush=True)
    print('power in %%; reps %d per cell, seed %d, Monte Carlo SE <= %.2f pt' % (a.reps, a.seed, 50 / np.sqrt(a.reps)))


if __name__ == '__main__':
    main()
