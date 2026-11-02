"""SUB4 / R22: flight-level evidence for every gate of a split, from the stored rows.

  python tools/class_bootstrap.py [--split dev|fresh|held_out] [--pickers deco,ssc1.13p836] [--out results/sub4/class_table_<split>.md]

Per gate json (results/gates/*__<split>.json): per-flight flat fail % of candidate and incumbent, the paired CLUSTER bootstrap by
flight (flights resampled with replacement, 1000 draws, seed 0, frame-weighted pooled difference in points) next to the gate's own
20-frame block CI, the flight sign test and the Wilcoxon signed-rank test on the per-flight flat fail % (one-sided, candidate lower).
Rows live on Euler (results/runs/<run id>); the laptop holds the held-out dpp0.5 pair (results/runs_gate). A gate whose rows are
absent is listed from its json only, tagged [T] (table level). Nothing here changes a verdict: the gate rule stays kpbench.stats.gate.
"""
import argparse, glob, json, os, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
N_BOOT = 1000


def per_flight_fail(rows_c, rows_i, lab, flat_thr):
    """(flat fails cand, flat fails inc, n flat) of one flight from paired rows"""
    from kpbench.stats import fail_flags
    fa, fb = fail_flags(rows_c), fail_flags(rows_i)
    flat = np.array([(lab[str(r[1])].get('plane_frac') or -1) >= flat_thr for r in rows_c], bool)
    return float(fa[flat].sum()), float(fb[flat].sum()), int(flat.sum())


def cluster_ci(dc, di, n, n_boot=N_BOOT, seed=0):
    """flight-level arrays (fails cand, fails inc, flat frames) -> (diff pt, lo, hi): flights resampled with replacement"""
    dc, di, n = map(np.asarray, (dc, di, n)); m = len(n)
    pick = np.random.default_rng(seed).integers(0, m, (n_boot, m))
    bs = 100 * (dc[pick] - di[pick]).sum(1) / n[pick].sum(1)
    return float(100 * (dc - di).sum() / n.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def wilcoxon_p(pc, pi):
    """one-sided Wilcoxon signed-rank p (candidate fail % lower), tied flights dropped; None with fewer than 2 untied flights"""
    from scipy.stats import wilcoxon
    d = np.asarray(pc) - np.asarray(pi); d = d[d != 0]
    if len(d) < 2:
        return None
    return float(wilcoxon(d, alternative='less').pvalue)


def flight_table(cand, inc, split):
    """-> dict(flights, pc, pi, n, cluster, better, worse, tied, sign_p, wilcoxon_p) from rows, or None when rows are absent"""
    from kpbench.gate import load_run, labels, split_info, GateError, RESULTS
    from kpbench.splits import FLAT
    from kpbench.stats import sign_test
    flights = split_info(split)[0]
    rs = [r for r in ('runs', 'runs_gate') if os.path.isdir(os.path.join(RESULTS, r, cand)) and os.path.isdir(os.path.join(RESULTS, r, inc))]
    if not rs:
        return None
    try:
        _, rc = load_run(cand, flights, rs[0]); _, ri = load_run(inc, flights, rs[0])
    except GateError:
        return None
    dc, di, n = [], [], []
    for f in flights:
        assert [r[1] for r in rc[f][0]] == [r[1] for r in ri[f][0]], f
        a, b, k = per_flight_fail(rc[f][0], ri[f][0], labels(f), FLAT)
        if k:
            dc.append(a); di.append(b); n.append(k)
    pc, pi = 100 * np.array(dc) / np.array(n), 100 * np.array(di) / np.array(n)
    better, worse = int((pc < pi).sum()), int((pc > pi).sum())
    return dict(flights=len(n), pc=pc.tolist(), pi=pi.tolist(), n=n, cluster=cluster_ci(dc, di, n), better=better, worse=worse,
                tied=len(n) - better - worse, sign_p=sign_test(better, worse), wilcoxon_p=wilcoxon_p(pc, pi))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='dev'); ap.add_argument('--pickers', default='', help='comma list of gate pickers (default: every gate of the split)')
    ap.add_argument('--out', default=''); a = ap.parse_args()
    from kpbench.gate import GATES
    want = set(p for p in a.pickers.split(',') if p)
    G = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(GATES, '*__%s.json' % a.split)))]
    G = [g for g in G if not want or g.get('picker') in want]
    L = ['# Flight-level evidence, %s gates (%s)' % (a.split, time.strftime('%Y-%m-%d')), '',
         'tools/class_bootstrap.py: cluster CI = flights resampled with replacement (%d draws, seed 0), frame-weighted pooled flat-fail difference in points; '
         'block CI = the gate json (20-frame blocks); Wilcoxon = signed-rank on per-flight flat fail %%, one-sided (candidate lower), tied flights dropped. '
         'Verdicts are the gate rule (kpbench.stats.gate), unchanged. [T] = rows absent here, json only.' % N_BOOT, '',
         '| candidate vs incumbent | K | flat fail % cand / inc (n flat) | block CI pt | cluster CI pt | flights b/w/t | sign p | Wilcoxon p | verdict |',
         '|---|---|---|---|---|---|---|---|---|']
    out = []
    for g in sorted(G, key=lambda g: g['flat_cand']):
        t = flight_table(g['cand'], g['inc'], a.split)
        inc = g['inc'].split('_')[1]; name = '%s vs %s' % (g.get('picker', g['cand']), inc)
        row = dict(picker=g.get('picker'), inc=inc, K=g.get('K', 100), flat_cand=g['flat_cand'], flat_inc=g['flat_inc'], n_flat=g['n_flat'], block=g['flat_diff'], win=g['win'], rows=t)
        if t is None:
            L.append('| %s [T] | %d | %.2f / %.2f (%d) | %+.2f [%+.2f, %+.2f] | - | %d/%d/%d | %.3f | - | %s |' % (
                name, row['K'], g['flat_cand'], g['flat_inc'], g['n_flat'], *g['flat_diff'], g['flights_better'], g['flights_worse'], g['flights_tied'], g['sign_p'], 'KEEP' if g['win'] else 'REVERT'))
        else:
            L.append('| %s | %d | %.2f / %.2f (%d) | %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] | %d/%d/%d | %.3f | %s | %s |' % (
                name, row['K'], g['flat_cand'], g['flat_inc'], g['n_flat'], *g['flat_diff'], *t['cluster'], t['better'], t['worse'], t['tied'], t['sign_p'],
                '%.3f' % t['wilcoxon_p'] if t['wilcoxon_p'] is not None else '-', 'KEEP' if g['win'] else 'REVERT'))
        out.append(row)
    L += ['', 'Read: a cluster CI that covers 0 while the block CI does not means the frame-level gain rests on a few flights; the paper then states the gain as frame-weighted only (R22).']
    fn = a.out or os.path.join(ROOT, 'results', 'sub4', 'class_table_%s.md' % a.split)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    open(fn, 'w', encoding='utf-8').write('\n'.join(L) + '\n'); json.dump(out, open(fn[:-3] + '.json', 'w'), indent=1)
    print('\n'.join(L)); print('->', fn)


if __name__ == '__main__':
    main()
