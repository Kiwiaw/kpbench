"""SUB3: paired repeatability test of two gate runs (same queries, same node), per flight and per 20-frame block.

  python -m kpbench.rep5_paired --cand raco_dppr0.367a0.744b2.34t0.9_K100_same_lg_g-dppr0.367a0.744b2.34t0.9 \
                                --inc raco_dpp0.5_K100_same_lg_g-dppr0.367a0.744b2.34t0.9 [--split dev] [--col rep5]

Per flight: mean rep5 over the flat queries (plane_frac >= 0.85; the TUNE objective) and over all queries.
Sign test over flights (one-sided, candidate higher), paired block bootstrap 95 % CI of the pooled mean difference
(blocks = 20 consecutive frames of a flight, as the gate). Rows with rep5 None are dropped pairwise.
"""
import argparse
import numpy as np
from .bench import COLS
from .gate import load_run, labels, split_info
from .stats import block_ids, sign_test, N_BOOT
from .splits import FLAT


def paired_ci(d, blocks, n_boot=N_BOOT, seed=0):
    _, inv = np.unique(blocks, return_inverse=True); m = inv.max() + 1
    s = np.bincount(inv, d, m); c = np.bincount(inv, None, m)
    pick = np.random.default_rng(seed).integers(0, m, (n_boot, m))
    bs = s[pick].sum(1) / c[pick].sum(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def compare(cand, inc, split='dev', col='rep5', runs='runs'):
    flights = split_info(split)[0]; j = COLS.index(col)
    _, rc = load_run(cand, flights, runs); _, ri = load_run(inc, flights, runs)
    out = dict(per_flight=[], pooled={})
    D = {'flat': ([], [], []), 'all': ([], [], [])}
    for f in flights:
        ra, rb = rc[f][0], ri[f][0]; lab = labels(f)
        assert [r[1] for r in ra] == [r[1] for r in rb], f
        row = dict(flight=f)
        for name in ('flat', 'all'):
            a, b, q = [], [], []
            for x, y in zip(ra, rb):
                pf = lab[str(x[1])].get('plane_frac')
                if name == 'flat' and not (pf is not None and pf >= FLAT):
                    continue
                if x[j] is None or y[j] is None:
                    continue
                a.append(x[j]); b.append(y[j]); q.append(x[1])
            row[name] = (len(a), float(np.mean(a)) if a else None, float(np.mean(b)) if b else None)
            D[name][0].extend(np.subtract(a, b)); D[name][1].extend([f] * len(a)); D[name][2].extend(q)
        out['per_flight'].append(row)
    for name in ('flat', 'all'):
        d = np.array(D[name][0], float); bl = block_ids(D[name][1], D[name][2])
        rows = [r[name] for r in out['per_flight'] if r[name][0]]
        better = sum(1 for n, a, b in rows if a > b); worse = sum(1 for n, a, b in rows if a < b)
        out['pooled'][name] = dict(n=len(d), cand=float(np.mean([a for n, a, b in rows for _ in range(n)])),
                                   inc=float(np.mean([b for n, a, b in rows for _ in range(n)])),
                                   diff=paired_ci(d, bl), better=better, worse=worse, tied=len(rows) - better - worse,
                                   sign_p=sign_test(better, worse))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True); ap.add_argument('--inc', required=True)
    ap.add_argument('--split', default='dev'); ap.add_argument('--col', default='rep5'); ap.add_argument('--runs', default='runs')
    a = ap.parse_args(argv)
    o = compare(a.cand, a.inc, a.split, a.col, a.runs)
    print('| flight | n flat | cand flat | inc flat | n all | cand all | inc all |'); print('|---|---|---|---|---|---|---|')
    for r in o['per_flight']:
        fl, al = r['flat'], r['all']
        print('| %s | %d | %s | %s | %d | %.4f | %.4f |' % (r['flight'], fl[0], '%.4f' % fl[1] if fl[0] else '-', '%.4f' % fl[2] if fl[0] else '-', *al))
    for name, p in o['pooled'].items():
        print('%s: n %d, %s %.4f vs %.4f, diff %+.4f [%+.4f, %+.4f], flights better/worse/tied %d/%d/%d, sign p %.3f' % (
            name, p['n'], a.col, p['cand'], p['inc'], *p['diff'], p['better'], p['worse'], p['tied'], p['sign_p']))


if __name__ == '__main__':
    main()
