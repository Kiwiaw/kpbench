"""SUB3 T33/T4 read-out over several arms (dispatch 2026-11-18 13:15): where do each picker's picks go, per flight, vs a reference arm.

  python -m kpbench.pickarms --arms raco_dpp0.5_K100_same_lg_picks,raco_dppr0.317a1.6b2.01t0.9_K100_same_lg_picks,\
raco_cube0.5_K100_same_lg_picks,raco_grid_K100_same_lg_picks,raco_cube0.359_K100_same_lg_picks --split dev [--runs runs] [--out md]

First arm = reference (the incumbent dpp0.5). Runs without picks are refused (gate.require_picks). Per flat query (plane_frac >= 0.85):
  off   = off-plane share of the picks with depth           (100 picks with depth, 80 on the plane -> 0.20)
  offn  = off-plane pick COUNT = off x picks with depth      (0.20 x 100 -> 20)
  dspan = (q90 - q10) / median pick depth                    (depths 5 .. 25 m around 10 m -> about 2)
Per arm vs reference, paired on the same queries: median diff, flight-cluster bootstrap 95 % CI of the mean diff (flights resampled,
10000, seed 0), flights whose own mean diff is > 0 / < 0. SUB1 column (descriptive here, its rule lives in the SUB1 prereg):
  sep = mean off on failed flat queries - mean off on posed flat queries, per arm (> 0: failures have MORE off-plane picks).
"""
import argparse, gzip, json, os
import numpy as np
from .data import slug
from .gate import require_picks, split_info, labels, RESULTS
from .stats import fail_flags
from .splits import FLAT

FIELDS = ('off', 'offn', 'dspan')


def query_fields(entry):
    """one onplane entry [share, n with depth, plane share of fit, labels, depths] -> (off, offn, dspan), None where missing"""
    if not entry or entry[0] is None:
        return None, None, None
    off = 1.0 - float(entry[0]); n = entry[1] if len(entry) > 1 else None
    offn = None if n is None else off * float(n)
    z = np.array([x for x in entry[4] if x is not None], float) if len(entry) >= 5 and entry[4] is not None else np.zeros(0)
    dspan = None
    if len(z) >= 2:
        q10, q50, q90 = np.percentile(z, [10, 50, 90]); dspan = float((q90 - q10) / q50) if q50 > 0 else None
    return off, offn, dspan


def arm_table(rid, flights, runs='runs', flat_of=None):
    """-> {(flight, q): (fail, off, offn, dspan)} over flat queries of one run with picks"""
    _, out = require_picks(rid, flights, runs); d = os.path.join(RESULTS, runs, rid); res = {}
    for f in flights:
        lab = labels(f); j = json.load(gzip.open(os.path.join(d, slug(f) + '.json.gz'), 'rt'))
        op = j.get('onplane') or {}; rows = out[f][0]
        for r, bad in zip(rows, fail_flags(rows)):
            if (lab.get(str(r[1]), {}).get('plane_frac') or -1) >= FLAT:
                res[(f, r[1])] = (int(bad),) + query_fields(op.get(str(r[1])))
    return res


def cluster_ci(d, fl, n_boot=10000, seed=0):
    """flight-cluster bootstrap 95 % CI of the mean of d (fl = flight of each value)"""
    u = sorted(set(fl)); idx = {f: np.flatnonzero(np.array(fl) == f) for f in u}
    s = np.array([d[idx[f]].sum() for f in u]); c = np.array([len(idx[f]) for f in u])
    b = np.random.default_rng(seed).integers(0, len(u), (n_boot, len(u)))
    m = s[b].sum(1) / c[b].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_field(A, B, keys, k, n_boot=10000, seed=0):
    """A - B on field k (1 off, 2 offn, 3 dspan): n, median, mean, CI, flights up / down"""
    ks = [q for q in keys if A[q][k] is not None and B[q][k] is not None]
    if not ks:
        return None
    d = np.array([A[q][k] - B[q][k] for q in ks], float); fl = [q[0] for q in ks]
    lo, hi = cluster_ci(d, fl, n_boot, seed)
    fm = {f: d[np.array(fl) == f].mean() for f in set(fl)}
    return dict(n=len(d), median=round(float(np.median(d)), 4), mean=round(float(d.mean()), 4), ci=[round(lo, 4), round(hi, 4)],
                up=sum(v > 0 for v in fm.values()), down=sum(v < 0 for v in fm.values()), flights=len(fm))


def sep(T):
    """SUB1 column: mean off on failed minus posed flat queries (None if a group is empty)"""
    x = [(v[0], v[1]) for v in T.values() if v[1] is not None]
    f = [o for b, o in x if b]; p = [o for b, o in x if not b]
    return None if not f or not p else round(float(np.mean(f) - np.mean(p)), 4)


def compare_arms(arms, split, runs='runs', n_boot=10000, seed=0):
    flights, _, _ = split_info(split); T = {a: arm_table(a, flights, runs) for a in arms}; ref = arms[0]; res = []
    for a in arms:
        row = dict(arm=a, n_flat=len(T[a]), fail=sum(v[0] for v in T[a].values()), sep=sep(T[a]),
                   median={n: (None if not (v := [x[i] for x in T[a].values() if x[i] is not None]) else round(float(np.median(v)), 4))
                           for i, n in enumerate(FIELDS, 1)})
        if a != ref:
            keys = sorted(set(T[a]) & set(T[ref]))
            row['vs_ref'] = {n: paired_field(T[a], T[ref], keys, i, n_boot, seed) for i, n in enumerate(FIELDS, 1)}
        res.append(row)
    return dict(ref=ref, split=split, arms=res)


def to_md(r):
    L = ['# pickarms read-out, split %s, reference %s' % (r['split'], r['ref']), '',
         '| arm | flat | fail | median off | offn | dspan | sep (SUB1) | off - ref [CI] up/down | offn - ref [CI] | dspan - ref [CI] |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for a in r['arms']:
        v = a.get('vs_ref') or {}
        c = ['%+.4f [%+.4f,%+.4f] %d/%d' % (p['mean'], *p['ci'], p['up'], p['down']) if (p := v.get(n)) else '-' for n in FIELDS]
        m = a['median']; L.append('| %s | %d | %d | %s | %s | %s | %s | %s |' % (a['arm'], a['n_flat'], a['fail'], m['off'], m['offn'], m['dspan'], a['sep'], ' | '.join(c)))
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--arms', required=True, help='comma list, first = reference')
    ap.add_argument('--split', default='dev'); ap.add_argument('--runs', default='runs'); ap.add_argument('--out', default='')
    a = ap.parse_args(argv); r = compare_arms([x for x in a.arms.split(',') if x], a.split, a.runs)
    print(json.dumps(r, indent=1))
    if a.out:
        open(a.out, 'w').write(to_md(r))


if __name__ == '__main__':
    main()
