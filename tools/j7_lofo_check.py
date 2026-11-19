"""J7 LOFO second method (SUB2, dispatch 2026-11-19T13:15): leave-one-flight-out p2 of dpp0.5 - cube0.5 K100 flat fails,
recomputed from the per-frame rows with code that shares nothing with tools/j7_forest.py / c43_k_check.py.

  python tools/j7_lofo_check.py            # both sets, prints PASS / FAIL against the J7 table

Differences from J7: rows are read here (own fail rule: err None or > 25 cm or > 2 deg; flat = plane_frac >= 0.85), and the
two-sided sign-flip p is a brute-force count over all 2^m sign patterns (J7 convolves counts). Toy: d = (-3, -1, +1) has
8 patterns, |sum| >= 3 in 4 of them -> p2 0.5. Expected values = the 'pooled without it' column of results/main/j7_forest_register.md.
"""
import gzip, json, os, re
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SETS = dict(board=('raco_dpp0.5_K100_same_lg', 'raco_cube0.5_K100_same_lg', (1813, 192, 227, 0.0386)),
            fresh1=('raco_dpp0.5_K100_same_lg_g-dppr0.317a1.6b2.01t0.9-fresh', 'raco_cube0.5_K100_same_lg_g-cube0.5-fresh', (2075, 62, 95, 0.0508)))


def flight_counts(run_a, run_b, pairs=False):
    """-> {flight: (n flat, fails a, fails b)} over the flat queries both runs share, straight from the row files
    (pairs=True adds a 4th entry: queries where only a fails, so only-b = b - a + only-a)"""
    out = {}
    da, db = (os.path.join(ROOT, 'results', 'runs', r) for r in (run_a, run_b))
    for fn in sorted(os.listdir(da)):
        if not fn.endswith('.json.gz') or not os.path.exists(os.path.join(db, fn)):
            continue
        lab = json.load(open(os.path.join(ROOT, 'results', 'labels', fn[:-8] + '.json')))
        bad = []
        for d in (da, db):
            rows = json.load(gzip.open(os.path.join(d, fn), 'rt'))['rows']
            bad.append({int(r[1]): (r[2] is None) or (r[2] > 25) or (r[3] > 2) for r in rows})
        assert bad[0].keys() == bad[1].keys(), fn
        flat = [q for q in bad[0] if (lab.get(str(q), {}).get('plane_frac') or -1) >= 0.85]
        if flat:
            out[fn[:-8]] = (len(flat), sum(bad[0][q] for q in flat), sum(bad[1][q] for q in flat))
            if pairs:
                out[fn[:-8]] += (sum(bad[0][q] and not bad[1][q] for q in flat),)
    return out


def p2_brute(d):
    """two-sided sign-flip p by listing all 2^m patterns of the nonzero d (m <= 20)"""
    x = np.array([abs(v) for v in d if v], dtype=np.int64)
    assert len(x) <= 20
    bits = (np.arange(2 ** len(x))[:, None] >> np.arange(len(x))) & 1
    s = ((2 * bits - 1) * x).sum(1)
    return float((np.abs(s) >= abs(int(np.sum(d)))).mean())


def j7_table(name):
    """parse the J7 register table of a set -> {flight_with_underscore: (n, a, b, pooled_without, p2_without)}"""
    txt = open(os.path.join(ROOT, 'results', 'main', 'j7_forest_register.md'), encoding='utf-8').read()
    blk = txt.split('sums: flat %d' % SETS[name][2][0])[1].split('wrote')[0]
    T = {}
    for m in re.finditer(r'\| (\S+) \| (\d+) \| (\d+) \| (\d+) \| \S+ \| \[[^\]]*\] \| (\S+) \(p2 ([\d.]+)\) \|', blk):
        T[m.group(1).replace('/', '_')] = (int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5)), float(m.group(6)))
    return T


def main():
    allok = True
    for name, (ra, rb, (N, A, B, P)) in SETS.items():
        C = flight_counts(ra, rb); J = j7_table(name)
        fl = sorted(C); n = np.array([C[f][0] for f in fl]); d = np.array([C[f][1] - C[f][2] for f in fl])
        tot = (int(n.sum()), sum(C[f][1] for f in fl), sum(C[f][2] for f in fl)); p = p2_brute(d)
        ok = tot == (N, A, B) and abs(p - P) < 5e-5 and set(fl) == set(J)
        lo = []; nbad = 0
        for j, f in enumerate(fl):
            m = np.arange(len(fl)) != j
            pw, pp = 100 * d[m].sum() / n[m].sum(), p2_brute(d[m])
            good = C[f] == J[f][:3] and abs(pw - J[f][3]) < 0.0051 and abs(pp - J[f][4]) < 0.00051 if f in J else False
            nbad += not good; lo.append((pp, f, pw))
        ok = ok and nbad == 0; allok = allok and ok
        print('%s: %d flights with flat, flat %d, dpp0.5 %d, cube0.5 %d, p2 %.4f (brute force, %d nonzero flights); LOFO p2 %.4f (drop %s) to %.4f (drop %s); '
              'rows vs J7 table: %d/%d flights match (counts, pooled-without to 0.01 pt, p2 to 0.001) -> %s'
              % (name, len(fl), *tot, p, int((d != 0).sum()), min(lo)[0], min(lo)[1], max(lo)[0], max(lo)[1], len(fl) - nbad, len(fl), 'PASS' if ok else 'FAIL'))
        for pp, f, pw in sorted(lo, reverse=True)[:3]:
            print('  largest LOFO p2: drop %s -> %+.2f pt, p2 %.4f' % (f, pw, pp))
    print('J7 LOFO second method:', 'PASS' if allok else 'FAIL')


if __name__ == '__main__':
    main()
