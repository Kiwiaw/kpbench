"""dpp0.5 vs cube0.5 FRESH-1 K100 flat, flight-level replication test (MAIN, dispatch 2026-11-15; prereg in results/main/dpp_vs_cube_fresh1_k100.md).

  python tools/dpp_vs_cube_k100.py [--boot 10000] [--seed 0]

Per flight: dpp0.5 - cube0.5 flat fail counts. Flight-cluster bootstrap of the ratio of sums (pt), exact two-sided sign-flip p
(tools/c43_k_check.exact_p), flights better / tie / worse for dpp0.5. Rows and fail rule as tools/c43_k_check.py.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c43_k_check import ROOT, RUNS, fails, exact_p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=10000); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
    T = []
    for f in json.load(open(os.path.join(ROOT, 'splits_fresh.json')))['fresh']:
        lab = json.load(open(os.path.join(ROOT, 'results', 'labels', f.replace('/', '_') + '.json')))
        i, c = fails(RUNS['i100'], f), fails(RUNS['c100'], f); assert set(i) == set(c), f
        flat = [q for q in i if (lab.get(str(q), {}).get('plane_frac') or -1) >= 0.85]
        if flat:
            T.append((f, len(flat), sum(i[q] for q in flat), sum(c[q] for q in flat)))
    n = np.array([t[1] for t in T]); di = np.array([t[2] for t in T]); dc = np.array([t[3] for t in T]); d = di - dc
    pick = np.random.default_rng(a.seed).integers(0, len(T), (a.boot, len(T)))
    bs = 100 * d[pick].sum(1) / n[pick].sum(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    print('flat %d, dpp0.5 %d, cube0.5 %d, diff %+d = %+.2f pt, CI [%+.2f, %+.2f], better/tie/worse %d/%d/%d, p2 %.4f'
          % (n.sum(), di.sum(), dc.sum(), d.sum(), 100 * d.sum() / n.sum(), lo, hi, (d < 0).sum(), (d == 0).sum(), (d > 0).sum(), exact_p(d)))
    for t, x in zip(T, d):
        print('| %s | %d | %d | %d | %+d |' % (*t, x))


if __name__ == '__main__':
    main()
