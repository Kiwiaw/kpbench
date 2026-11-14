"""Picker x K interaction on FRESH-1 flat frames from mirrored rows (MAIN T11 K step, dispatch 2026-11-14).

  python tools/k_interaction.py [--boot 10000] [--seed 0] [--out results/main/k_interaction_fresh1.md]

Per flight f: d_f = (cand - inc flat fails at K400) - (cand - inc flat fails at K100), counts. Pooled interaction in points =
100 * sum d_f / n_flat. Flight-cluster CI: flights resampled with replacement, ratio of sums. p = exact two-sided sign-flip of the
per-flight d_f (zeros dropped). Fail = no pose, > 25 cm or > 2 deg (tools/class_bootstrap.per_flight_fail); flat = plane_frac >= 0.85.
"""
import argparse, itertools, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from class_bootstrap import per_flight_fail
from kpbench.gate import load_run, labels, split_info
from kpbench.splits import FLAT

RUNS = dict(c400='raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400', i400='raco_dpp0.5_K400_same_lg_g-cube0.5-fresh-K400',
            c100='raco_cube0.5_K100_same_lg_g-cube0.5-fresh', i100='raco_dpp0.5_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh')


def signflip_p(d):
    """exact two-sided sign-flip p of sum(d) over the non-zero entries"""
    d = [x for x in d if x]; s = abs(sum(d))
    return float(np.mean([abs(sum(x * e for x, e in zip(d, sg))) >= s - 1e-9 for sg in itertools.product((1, -1), repeat=len(d))])) if d else 1.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=10000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'main', 'k_interaction_fresh1.md')); a = ap.parse_args()
    flights = split_info('fresh')[0]; R = {}
    for k, run in RUNS.items():
        rs = 'runs' if os.path.isdir(os.path.join(ROOT, 'results', 'runs', run)) else 'runs_gate'
        R[k] = load_run(run, flights, rs)[1]
    T = []
    for f in flights:
        q = [r[1] for r in R['c400'][f][0]]
        assert all([r[1] for r in R[k][f][0]] == q for k in RUNS), f
        c4, i4, n = per_flight_fail(R['c400'][f][0], R['i400'][f][0], labels(f), FLAT)
        c1, i1, n2 = per_flight_fail(R['c100'][f][0], R['i100'][f][0], labels(f), FLAT); assert n == n2
        if n: T.append((f, n, c4, i4, c1, i1, (c4 - i4) - (c1 - i1)))
    n = np.array([t[1] for t in T]); d = np.array([t[6] for t in T]); N = n.sum()
    tot = [int(sum(t[j] for t in T)) for j in (2, 3, 4, 5)]
    pt = 100 * d.sum() / N; rng = np.random.default_rng(a.seed); bs = []
    for _ in range(a.boot):
        i = rng.integers(0, len(T), len(T)); bs.append(100 * d[i].sum() / n[i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = signflip_p(d.tolist())
    neg, pos = int((d < 0).sum()), int((d > 0).sum())
    L = ['# Picker x K interaction, cube0.5 vs dpp0.5, FRESH-1 flat (%s)' % time.strftime('%Y-%m-%d %H:%M'), '',
         'Rows: %s. %d flights with flat frames, %d flat. Script: tools/k_interaction.py (boot %d, seed %d).' % (', '.join(RUNS.values()), len(T), N, a.boot, a.seed), '',
         'Flat fails of %d: K400 cube0.5 %d / dpp0.5 %d; K100 cube0.5 %d / dpp0.5 %d.' % (N, *tot), '',
         '| quantity | value |', '|---|---|',
         '| cube0.5 - dpp0.5 at K400 | %+d frames = %+.2f pt |' % (tot[0] - tot[1], 100 * (tot[0] - tot[1]) / N),
         '| cube0.5 - dpp0.5 at K100 | %+d frames = %+.2f pt |' % (tot[2] - tot[3], 100 * (tot[2] - tot[3]) / N),
         '| interaction (K400 - K100) | %+d frames = %+.2f pt |' % (d.sum(), pt),
         '| flight-cluster 95 %% CI | [%+.2f, %+.2f] pt |' % (lo, hi),
         '| flights neg / pos / zero | %d / %d / %d |' % (neg, pos, len(T) - neg - pos),
         '| exact sign-flip p2 (counts) | %.4g |' % p, '',
         '| flight | n flat | K400 c / i | K100 c / i | d |', '|---|---|---|---|---|']
    L += ['| %s | %d | %d / %d | %d / %d | %+d |' % t for t in T]
    open(a.out, 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L)); print('->', a.out)


if __name__ == '__main__':
    main()
