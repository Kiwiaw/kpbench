"""SUB1 finding 4 (0 GPU): does uniformity lower error magnitude, or only the 25 cm / 2 deg fail count? From stored run rows + labels.

  python tools/error_magnitude.py [--split dev] [--out results/sub1/error_auc_dev.md]

Flat queries (plane_frac >= 0.85). No pose counts as infinite error. Median position / rotation error, recall-AUC of position error
up to 5 / 10 / 25 cm (mean over thresholds 0.1 cm steps of share within, no pose = never), no-pose vs bad-pose split of the fails.
Paired vs the next picker in the fail order: per-flight median position error (flights with flat queries), better/worse count.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpbench.audit import run_rows, labels
from kpbench.splits import load as load_splits
from kpbench.stats import fail_flags

PICKERS = ['dpp0.5', 'cube0.5', 'grid', 'sd']


def rows_of(p, flights, lab):
    """flat rows -> err, rot (nan = no pose), fail, flight per query, same query order for every picker"""
    R = [r for r in run_rows('raco_%s_K100_same_lg' % p) if r[0] in flights and (lab[r[0]][str(r[1])].get('plane_frac') or -1) >= 0.85]
    e = np.array([np.inf if r[2] is None else r[2] for r in R], float); a = np.array([np.inf if r[3] is None else r[3] for r in R], float)
    return e, a, fail_flags(R), np.array([r[0] for r in R]), [r[1] for r in R]


def auc(e, top):
    t = np.arange(0.1, top + 1e-9, 0.1)
    return float(np.mean([(e <= x).mean() for x in t]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='dev'); ap.add_argument('--out', default=None); a = ap.parse_args()
    flights = set(load_splits()[a.split]); lab = labels(); D = {p: rows_of(p, flights, lab) for p in PICKERS}
    assert all(D[p][4] == D['sd'][4] and list(D[p][3]) == list(D['sd'][3]) for p in PICKERS)
    n = len(D['sd'][0]); out = ['# Error magnitude on %s flat queries, n = %d (leaderboard runs, not gate runs)' % (a.split, n), '',
                                '| picker | fail % | median cm | median deg | AUC 5 cm | AUC 10 cm | AUC 25 cm | fails no pose | fails bad pose |', '|---|---|---|---|---|---|---|---|---|']
    for p in PICKERS:
        e, r, f, _, _ = D[p]
        out.append('| %s | %.2f (%d) | %.2f | %.3f | %.3f | %.3f | %.3f | %d | %d |' % (p, 100 * f.mean(), f.sum(), np.median(e), np.median(r), auc(e, 5), auc(e, 10), auc(e, 25), np.isinf(e).sum(), f.sum() - np.isinf(e).sum()))
    out += ['', 'Per-flight median position error (cm); pair = left better / left worse / tied, of flights with flat queries:', '', '| pair | median better/worse/tied | flights |', '|---|---|---|']
    fl = sorted(set(D['sd'][3])); P = {}
    for x, y in (('dpp0.5', 'cube0.5'), ('cube0.5', 'grid'), ('grid', 'sd'), ('cube0.5', 'sd'), ('dpp0.5', 'sd')):
        ma = np.array([np.median(D[x][0][D[x][3] == f]) for f in fl]); mb = np.array([np.median(D[y][0][D[y][3] == f]) for f in fl])
        b, w = int((ma < mb).sum()), int((ma > mb).sum()); P[(x, y)] = (b, w)
        out.append('| %s vs %s | %d/%d/%d | %d |' % (x, y, b, w, len(fl) - b - w, len(fl)))
    med = [np.median(D[p][0]) for p in PICKERS]
    r1 = all(med[i] < med[i + 1] for i in range(3)); r2 = all(P[k][0] >= 10 for k in [('dpp0.5', 'cube0.5'), ('cube0.5', 'grid'), ('grid', 'sd')])
    r3 = P[('cube0.5', 'grid')][0] >= 10; nop = [np.isinf(D[p][0]).sum() for p in PICKERS]; bad = [D[p][2].sum() - np.isinf(D[p][0]).sum() for p in PICKERS]
    r4 = bad[3] - bad[0] > nop[3] - nop[0]
    out += ['', 'Pre-registered (dispatch 2026-10-30T13:40), checked in this order:',
            '- median order dpp0.5 < cube0.5 < grid < sd: %s (%s cm)' % ('yes' if r1 else 'NO', ' / '.join('%.2f' % m for m in med)),
            '- each neighbouring pair per-flight >= 10 better (of %d flights): %s' % (len(fl), 'yes' if r2 else 'NO'),
            '- median separates cube0.5 from grid (>= 10 flights): %s (%d)' % ('yes' if r3 else 'NO', P[('cube0.5', 'grid')][0]),
            '- gain sits in bad pose, not no pose (sd - dpp0.5): %s (bad pose %d, no pose %d frames)' % ('yes' if r4 else 'NO', bad[3] - bad[0], nop[3] - nop[0]),
            '- verdict: %s' % ('passes' if r1 and r2 and r3 and r4 else 'KILL: uniformity does not lower error magnitude by this rule; SUB1 keeps only Q210')]
    print('\n'.join(out))
    if a.out:
        open(a.out, 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
