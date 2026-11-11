"""J4 depth-quota control on the cached MICRO candidates (no images, poses or GPU; dispatch 2026-11-11 13:15, SUB3 + SUB1).

  python tools/depth_quota_micro.py [--K 100] [--bins 3] [--cell 0.5] [--ref cube0.5] [--append results/sub3/depth_tercile_micro_2026-11-10.md]

Control picker dq<n> (kept here, not in kpbench.pickers: no fail outcome exists for it yet): the frame's candidates with depth are
cut into n equal-count depth bins (n = 3: terciles), every bin gets K / n picks (remainder to the nearest bins), strongest score
first inside a bin; a bin that runs short is topped up by score from the others. No voxels, no pixel term. E.g. K 100, n 3:
34 / 33 / 33 picks from the near / middle / far third.

Rows: sd, cube0.5, cube0.5r (random member per cell), dq<n>. Per frame: share of picks in the nearest candidate-depth tercile,
std of pick depths (m), on-plane share (bench.onplane), number of distinct <cell> m cubes the picks occupy, pick Jaccard vs --ref.
Paired per-frame differences vs --ref with frames above / below and the two-sided sign-test p (ties dropped).
Pre-registered kills (dispatch): cube0.5r share within 0.05 of cube0.5 -> depth spread comes from the cells, not the score;
dq<n> share within 0.05 of cube0.5 AND no paired sign difference -> the 3D cap is a depth quota (prior art).
"""
import argparse, json, os, sys, time
import numpy as np
from math import comb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from kpbench import pickers
from kpbench.bench import onplane
from kpbench.cache import Cands
from kpbench.data import to3d, DEPTH_MAX

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def pick_dq(n):
    """dq<n>: equal quota per depth bin (equal-count bins of the candidates with depth), strongest first inside each bin"""
    def f(u, v, s, z, K):
        ok = np.flatnonzero(z < DEPTH_MAX)                    # candidates arrive in score order
        if not len(ok):
            return np.arange(min(K, len(u)))
        edges = np.quantile(z[ok], np.arange(1, n) / n); b = np.searchsorted(edges, z[ok], side='left')
        quota = [K // n + (i < K % n) for i in range(n)]; sel = []
        for i in range(n):
            sel += [int(j) for j in ok[b == i][:quota[i]]]
        return pickers.fill(pickers.fill(np.array(sorted(sel), int), len(u), K, pool=ok), len(u), K)
    return f


def sign_p(up, dn):
    m = up + dn; k = min(up, dn)
    return 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)


def stats(u, v, z, idx, cell):
    zc = z[z < DEPTH_MAX]; idx = np.asarray(idx, int); d = idx[z[idx] < DEPTH_MAX]
    if len(zc) < 3 or not len(d):
        return [np.nan] * 4
    q1 = np.quantile(zc, 1 / 3); zp = z[d]
    cells = len({tuple(k) for k in np.floor(to3d(u[d], v[d], z[d]) / cell).astype(int)})
    return [float((zp <= q1).mean()), float(zp.std()), onplane(u, v, z, idx)[0][0], cells]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=100); ap.add_argument('--det', default='raco')
    ap.add_argument('--bins', type=int, default=3); ap.add_argument('--cell', type=float, default=0.5); ap.add_argument('--ref', default='cube0.5')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'sub3', 'depth_quota_micro_%s.md' % time.strftime('%Y-%m-%d')))
    ap.add_argument('--append', default=''); a = ap.parse_args()
    dq = 'dq%d' % a.bins; names = ['sd', 'cube0.5', 'cube0.5r', dq]
    fns = {n: (pick_dq(a.bins) if n == dq else pickers.parse(n)) for n in names}
    micro = json.load(open(os.path.join(ROOT, 'splits.json')))['micro']
    S = {n: [] for n in names}; J = {n: [] for n in names}; nf = 0; short = 0
    for fid, frames in micro.items():
        C = Cands(fid.replace('/', '_'), a.det)
        for q in frames:
            u, v, s, z, _ = C.frame(q); nf += 1; P = {n: np.asarray(fns[n](u, v, s, z, a.K), int) for n in names}
            short += len(P[dq]) < a.K or len(set(P[dq].tolist())) < len(P[dq])
            for n in names:
                S[n].append(stats(u, v, z, P[n], a.cell)); A_, B_ = set(P[n].tolist()), set(P[a.ref].tolist())
                J[n].append(len(A_ & B_) / max(1, len(A_ | B_)))
    A = {n: np.array(S[n], float) for n in names}; ref = A[a.ref]; cols = ['near-tercile share', 'pick-depth std (m)', 'on-plane share', '%.1f m cubes occupied' % a.cell]
    L = ['## J4 depth-quota control + cube0.5r vs cube0.5 (engineer %s, tools/depth_quota_micro.py, K %d, %d MICRO frames, cache only)' % (time.strftime('%Y-%m-%d'), a.K, nf), '',
         '%s = equal quota per depth bin (%d equal-count bins of the frame\'s candidates with depth), strongest score first inside a bin, no voxels. '
         'Paired per-frame differences vs %s; a / b = frames above / below %s (ties dropped), p = two-sided sign test. '
         '%s returned K distinct picks on %d / %d frames.' % (dq, a.bins, a.ref, a.ref, dq, nf - short, nf), '',
         '| picker | ' + ' | '.join('%s | diff vs %s (a / b, p)' % (c, a.ref) for c in cols) + ' | pick Jaccard vs %s |' % a.ref, '|---|' + '---|' * (2 * len(cols) + 1)]
    res = {}
    for n in names:
        row = [n]
        for j in range(len(cols)):
            ok = ~np.isnan(A[n][:, j]) & ~np.isnan(ref[:, j]); d = A[n][ok, j] - ref[ok, j]
            up, dn = int((d > 1e-9).sum()), int((d < -1e-9).sum()); res[(n, j)] = (float(np.nanmean(A[n][:, j])), float(d.mean()), up, dn, sign_p(up, dn), int(ok.sum()))
            row += [('%.1f' if j == 3 else '%.3f') % res[(n, j)][0], '-' if n == a.ref else '%+.4f (%d / %d, p %.2g)' % (res[(n, j)][1], up, dn, res[(n, j)][4])]
        L.append('| ' + ' | '.join(row) + ' | %.3f |' % float(np.mean(J[n])))
    r = res[('cube0.5r', 0)]; g = res[(dq, 0)]; rs = res[('sd', 0)]
    L += ['', '- cube0.5r vs cube0.5 (SUB1 row): near-tercile share %.3f vs %.3f, diff %+.4f, frames %d above / %d below, sign p %.2g; vs sd %+.4f. Verdict: %s.' % (
              r[0], res[('cube0.5', 0)][0], r[1], r[2], r[3], r[4], r[0] - rs[0],
              'within 0.05 of cube0.5 -> the depth spread comes from the cells, not from the score order inside a cell; the random-within-cell arm is predicted to tie' if abs(r[1]) < 0.05
              else 'NOT within 0.05 of cube0.5'),
          '- %s vs cube0.5 (SUB3 control): near-tercile share %.3f vs %.3f, diff %+.4f, frames %d above / %d below, sign p %.2g. Kill (within 0.05 AND no paired sign difference): %s.' % (
              dq, g[0], res[('cube0.5', 0)][0], g[1], g[2], g[3], g[4], 'MET' if abs(g[1]) < 0.05 and g[4] > 0.05 else 'NOT met'),
          '- Caveat: the %s share is %.3f by construction (K / %d per bin), so the share difference says only that cubes spend LESS than a third of the budget near, not more evenly in depth; '
          'the columns that are not fixed by construction are depth std, on-plane share, cubes occupied and the pick overlap. No fail outcomes on the laptop: descriptive, [T].' % (dq, 1 / a.bins, a.bins)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n'); print('\n'.join(L)); print('->', a.out)
    if a.append:
        open(a.append, 'a', encoding='utf-8', newline='\n').write('\n' + '\n'.join(L) + '\n'); print('appended ->', a.append)


if __name__ == '__main__':
    main()
