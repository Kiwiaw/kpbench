"""Per-flight paired rep5 diff dpp0.5 - cube0.5 at K400, FRESH-1 flat frames (SUB3, dispatch 2026-11-27 13:20). [P] post hoc.

  python tools/rep5_k400_perflight.py [--boot 10000] [--seed 0] [--out results/sub3/rep5_k400_perflight_2026-11-27.md]

Frames: flat (plane_frac >= 0.85) with rep5 in both runs (n 2063 = the pooled 0.3518 vs 0.2455). Per flight: mean rep5 diff.
Intervals: paired block bootstrap (kpbench.stats.paired_diff, 20-frame blocks) and flight-cluster bootstrap (ratio of sums).
Dispatch kill: interval includes 0 or fewer than 10 of 13 flights positive -> no K400 rep5 claim.
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_e1_second import read, RUNS
from kpbench.stats import paired_diff, block_ids, sign_test


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=10000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'sub3', 'rep5_k400_perflight_2026-11-27.md')); a = ap.parse_args()
    I, C = read(RUNS['i400']), read(RUNS['c400']); fl, qs, x, y = [], [], [], []
    for f in sorted(I):
        lab = json.load(open(os.path.join(ROOT, 'results', 'labels', f + '.json')))
        for q in sorted(I[f]):
            if (lab[str(q)].get('plane_frac') or -1) >= 0.85 and I[f][q][8] is not None and C[f][q][8] is not None:
                fl.append(f); qs.append(q); x.append(I[f][q][8]); y.append(C[f][q][8])
    fl, qs, x, y = np.array(fl), np.array(qs), np.array(x), np.array(y); d = x - y
    diff, lo, hi = [v / 100 for v in paired_diff(x, y, block_ids(fl, qs), a.boot, a.seed)]
    F = sorted(set(fl)); s = np.array([d[fl == f].sum() for f in F]); n = np.array([(fl == f).sum() for f in F])
    pick = np.random.default_rng(a.seed).integers(0, len(F), (a.boot, len(F))); bs = s[pick].sum(1) / n[pick].sum(1)
    clo, chi = np.percentile(bs, [2.5, 97.5]); mf = s / n; pos, neg = int((mf > 0).sum()), int((mf < 0).sum())
    ok = lo > 0 and clo > 0 and pos >= 10
    L = ['# rep5 at K400: dpp0.5 - cube0.5 per flight, FRESH-1 flat (SUB3, 2026-11-27) [P]', '',
         'Rows: %s vs %s (mirrored, committed). Frames: flat with rep5 in both, n %d over %d flights. Script: tools/rep5_k400_perflight.py'
         ' (boot %d, seed %d). rep5 = share of picks re-detected within 5 px in the next frames (bench column).' % (RUNS['i400'], RUNS['c400'], len(d), len(F), a.boot, a.seed), '',
         'Pooled: dpp0.5 %.4f vs cube0.5 %.4f, diff %+.4f.' % (x.mean(), y.mean(), d.mean()),
         '- paired block bootstrap (20-frame blocks): [%+.4f, %+.4f]' % (lo, hi),
         '- flight-cluster bootstrap: [%+.4f, %+.4f]' % (clo, chi),
         '- flights dpp0.5 higher / lower: %d / %d of %d (one-sided sign p %.4f)' % (pos, neg, len(F), sign_test(pos, neg)), '',
         'Dispatch kill (interval includes 0 or < 10 of 13 flights positive): %s' % ('NOT MET -> K400 rep5 is a second dpp0.5 claim [P]' if ok else 'MET -> no K400 rep5 claim'), '',
         '| flight | n flat | dpp0.5 rep5 | cube0.5 rep5 | diff |', '|---|---|---|---|---|']
    L += ['| %s | %d | %.4f | %.4f | %+.4f |' % (f, k, x[fl == f].mean(), y[fl == f].mean(), m) for f, k, m in zip(F, n, mf)]
    open(a.out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
