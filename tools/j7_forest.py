"""J7 part 1 (SUB2, dispatch 2026-11-16 16:10): per-flight forest of dpp0.5 - cube0.5 flat fails, FRESH-1 K100 (13 flights) or
the 31 leaderboard flights (board).

  python tools/j7_forest.py [--set fresh1|board] [--boot 5000] [--seed 0] [--png results/main/j7_forest_fresh1_k100.png]

Per flight: diff in pt = (dpp0.5 fails - cube0.5 fails) / n flat, 95 % CI from a paired query bootstrap inside the flight
(e.g. 237 flat, 14 vs 32 fails -> -7.6 pt). The flight sums must give 62 vs 95 of 2075 = -1.59 pt (board: 192 vs 227 of 1813),
else the number is withdrawn. Also: leave-one-flight-out pooled diff (does one flight carry the gap?). Rows and fail rule as tools/c43_k_check.py.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c43_k_check import ROOT, RUNS, fails, exact_p


SETS = dict(fresh1=(RUNS['i100'], RUNS['c100'], (2075, 62, 95)),
            board=('raco_dpp0.5_K100_same_lg', 'raco_cube0.5_K100_same_lg', (1813, 192, 227)))


def set_flights(s):
    """flight ids of a set: FRESH-1 from splits_fresh.json, board = the row files of the leaderboard dpp0.5 run"""
    if s == 'fresh1':
        return json.load(open(os.path.join(ROOT, 'splits_fresh.json')))['fresh']
    d = os.path.join(ROOT, 'results', 'runs', SETS[s][0])
    return sorted('/'.join(x[:-8].rsplit('_', 1)) for x in os.listdir(d) if x.endswith('.json.gz'))


def flights_table(s='fresh1'):
    """-> [(flight, inc 0/1 array, cube 0/1 array)] over the flat queries of each flight of the set"""
    T = []; ri, rc, _ = SETS[s]
    for f in set_flights(s):
        lab = json.load(open(os.path.join(ROOT, 'results', 'labels', f.replace('/', '_') + '.json')))
        i, c = fails(ri, f), fails(rc, f); assert set(i) == set(c), f
        q = sorted(q for q in i if (lab.get(str(q), {}).get('plane_frac') or -1) >= 0.85)
        if q:
            T.append((f, np.array([i[x] for x in q]), np.array([c[x] for x in q])))
    return T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=5000); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--set', default='fresh1', choices=list(SETS))
    ap.add_argument('--png', default=''); a = ap.parse_args()
    rng = np.random.default_rng(a.seed); T = flights_table(a.set)
    a.png = a.png or os.path.join(ROOT, 'results', 'main', 'j7_forest_%s_k100.png' % a.set)
    n = np.array([len(t[1]) for t in T]); di = np.array([t[1].sum() for t in T]); dc = np.array([t[2].sum() for t in T]); d = di - dc
    ok = (n.sum(), di.sum(), dc.sum()) == SETS[a.set][2]
    print('sums: flat %d, dpp0.5 %d, cube0.5 %d, %+.2f pt, p2 %.4f -> %s' % (n.sum(), di.sum(), dc.sum(), 100 * d.sum() / n.sum(),
          exact_p(d), 'MATCH' if ok else 'MISMATCH vs n, dpp0.5, cube0.5 = %s' % (SETS[a.set][2],)))
    rows = []
    for (f, x, y), k in zip(T, n):
        dd = (x - y).astype(float); b = 100 * dd[rng.integers(0, k, (a.boot, k))].mean(1)
        rows.append((f, k, x.sum(), y.sum(), 100 * dd.mean(), *np.percentile(b, [2.5, 97.5])))
    print('| flight | n flat | dpp0.5 | cube0.5 | diff pt | 95 % CI (queries) | pooled without it |')
    print('|---|---|---|---|---|---|---|')
    for j, r in enumerate(rows):
        m = np.arange(len(T)) != j
        print('| %s | %d | %d | %d | %+.2f | [%+.2f, %+.2f] | %+.2f (p2 %.3f) |' % (*r, 100 * d[m].sum() / n[m].sum(), exact_p(d[m])))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(rows) + 1.2)); y = np.arange(len(rows))[::-1]
    ax.errorbar([r[4] for r in rows], y, xerr=[[r[4] - r[5] for r in rows], [r[6] - r[4] for r in rows]], fmt='o', color='#2a6fb0', ms=4)
    ax.axvline(0, color='#888', lw=0.8); ax.axvline(100 * d.sum() / n.sum(), color='#c0392b', lw=0.8, ls='--')
    ax.set_yticks(y); ax.set_yticklabels(['%s (n %d)' % (r[0], r[1]) for r in rows], fontsize=7)
    ax.set_xlabel('dpp0.5 - cube0.5 flat fail [pt] (left = DPP fails less); dashed = pooled %+.2f' % (100 * d.sum() / n.sum()), fontsize=7)
    fig.tight_layout(); fig.savefig(a.png, dpi=150); print('wrote', os.path.relpath(a.png, ROOT))


if __name__ == '__main__':
    main()
