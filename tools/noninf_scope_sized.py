"""Size-matched rerun of the T48 / T49 non-inferiority pass chances (SUB2, dispatch 2026-11-21 13:40).

  python tools/noninf_scope_sized.py [--outer 4000] [--inner 1000] [--seed 7] [--out results/audit/noninf_scope_sized_2026-11-22.md]

T49 (tools/noninf_scope_check.py, method A) draws FRESH-1 flights with replacement until the flat count reaches the target and
keeps the last flight whole, so a simulated set holds about 105 flat queries more than the registered size (2877 / 3605 / 4953
vs 2774 / 3500 / 4849): more data -> tighter bound -> pass chance slightly too high. Here the same nested flight resampling
runs with three stopping rules, same seed and draw counts:
  over  = T49 (last flight whole)                      -> must reproduce 61.4 / 70.8 / 82.3 %;
  exact = last flight cut to a contiguous run of its flat queries (random start) so the set has exactly the target size.
          The cut flight stays one cluster in the inner flight bootstrap;
  under = last flight dropped (set is smaller than the target) -> lower bracket.
Pass = upper end (97.5th percentile of the inner flight bootstrap) of cand - inc flat fail diff < +1.0 pt.
Fixed before any number was seen: 'exact' is the number to print; over and under are its bracket; no agreement rule (this is a
correction of a known bias, not a replication). Toy: flights of 100 and 200 flat, target 250: over = 300, exact = 250 (second
flight cut to 150 consecutive queries), under = 100.
"""
import argparse, os
import numpy as np
from noninf_scope_check import load, ROOT, MARGIN

T49 = {2774: 61.4, 3500: 70.8, 4849: 82.3}


def flights_of(Q):
    """-> list of per-flight d arrays (flat queries in frame order)"""
    G = {}
    for f, q, d in Q:
        G.setdefault(f, []).append((q, d))
    return [np.array([d for _, d in sorted(G[f])], float) for f in sorted(G)]


def one_set(D, n, tot_d, target, mode, rng):
    """one simulated set -> arrays (n_i, d_i) of its flight clusters"""
    idx = []; tot = 0
    while tot < target:
        i = int(rng.integers(len(D))); idx.append(i); tot += n[i]
    nn = n[idx].copy(); dd = tot_d[idx].copy()
    if mode == 'under':
        nn, dd = nn[:-1], dd[:-1]
    elif mode == 'exact':
        m = int(n[idx[-1]] - (tot - target)); s = int(rng.integers(len(D[idx[-1]]) - m + 1))
        nn[-1] = m; dd[-1] = D[idx[-1]][s:s + m].sum()
    return nn, dd


def chance(D, target, mode, outer, inner, rng):
    """-> (pass chance %, median upper, mean realised flat, mean clusters, sets too small to score)"""
    n = np.array([len(x) for x in D], float); tot_d = np.array([x.sum() for x in D]); up = []; real = 0.0; m = 0; skip = 0
    for _ in range(outer):
        nn, dd = one_set(D, n, tot_d, target, mode, rng)
        if len(nn) < 2:
            skip += 1; continue
        w = rng.multinomial(len(nn), np.full(len(nn), 1.0 / len(nn)), inner).astype(float)
        up.append(np.percentile(100 * (w @ dd) / (w @ nn), 97.5)); real += nn.sum(); m += len(nn)
    up = np.array(up)
    return 100 * float((up < MARGIN).mean()), float(np.median(up)), real / len(up), m / len(up), skip


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--outer', type=int, default=4000); ap.add_argument('--inner', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--out', default=None)
    a = ap.parse_args(argv); Q, tot = load(); D = flights_of(Q); N = sum(len(x) for x in D)
    assert (len(D), N, tot[0], tot[1]) == (13, 2075, 67, 62), (len(D), N, tot)
    L = ['# T48 pass chances at the registered sizes (SUB2, tools/noninf_scope_sized.py)', '',
         'Rows: %d FRESH-1 flights, %d flat, fails cand %d vs inc %d (dppr0.317a1.6b2.01t0.9 vs dpp0.5, K100). Pass = flight-bootstrap upper '
         'bound < %+.1f pt. Outer %d sets, inner %d draws, seed %d per column.' % (len(D), N, tot[0], tot[1], MARGIN, a.outer, a.inner, a.seed), '',
         '| registered flat | over (T49 rule) chance / realised flat | T49 printed | exact chance / median upper / realised | under chance / realised flat | exact - T49 |',
         '|---|---|---|---|---|---|']
    res = {}
    for t in T49:
        r = {m: chance(D, t, m, a.outer, a.inner, np.random.default_rng(a.seed + k)) for k, m in enumerate(('over', 'exact', 'under'))}
        res[t] = r; assert abs(r['exact'][2] - t) < 1e-6 and r['over'][4] == r['exact'][4] == 0
        L.append('| %d | %.1f %% / %.0f | %.1f %% | **%.1f %%** / %+.2f / %.0f | %.1f %% / %.0f | %+.1f |'
                 % (t, r['over'][0], r['over'][2], T49[t], r['exact'][0], r['exact'][1], r['exact'][2], r['under'][0], r['under'][2], r['exact'][0] - T49[t]))
    L += ['', 'Monte Carlo error of one chance: about +-%.1f points (95 %%, %d sets).' % (196 * (0.25 / a.outer) ** 0.5, a.outer)]
    txt = '\n'.join(L) + '\n'; print(txt)
    if a.out:
        open(os.path.join(ROOT, a.out), 'w', encoding='utf-8').write(txt)
    return res


if __name__ == '__main__':
    main()
