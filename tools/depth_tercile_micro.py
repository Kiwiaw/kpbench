"""O4 / R20 depth check on the cached MICRO candidates (no images, poses or GPU): where in depth does each picker spend K?

  python tools/depth_tercile_micro.py [--K 100] [--pickers sd,cube0.5,cube0.5r,dpp0.5,grid,dpp2d120a1] [--ref sd] [--out <md>] [--append <md>]

Per frame and picker: share of the picks (with depth) that fall in the NEAREST depth tercile of that frame's candidates with depth
(tercile edges = 33.3 / 66.7 % quantiles of the frame's candidate depths), the std of the pick depths in metres, and the on-plane
share (bench.onplane, R20). Then the paired per-frame mean difference vs --ref and the count of frames with each sign.
Pre-registered reading (dispatch 2026-11-10 12:30, SUB1 O4): ref sd nearest-tercile share exceeds cube0.5 by >= 0.05 with >= 300 / 393
frames same sign -> 'score ranking concentrates picks near-field on MICRO, voxel caps spread depth' (descriptive, 393 frames, no fail
outcomes); otherwise 'no near-field concentration' and the O4 mechanism paragraph is dropped.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from kpbench import pickers
from kpbench.bench import onplane
from kpbench.cache import Cands
from kpbench.data import DEPTH_MAX

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def frame_stats(u, v, z, idx):
    """-> (share of picks in the nearest candidate-depth tercile, std of pick depths, on-plane share); None when no pick has depth"""
    zc = z[z < DEPTH_MAX]
    idx = np.asarray(idx, int); zp = z[idx]; zp = zp[zp < DEPTH_MAX]
    if len(zc) < 3 or not len(zp):
        return None, None, None
    q1 = np.quantile(zc, 1 / 3)
    return float((zp <= q1).mean()), float(zp.std()), onplane(u, v, z, idx)[0][0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=100); ap.add_argument('--det', default='raco')
    ap.add_argument('--pickers', default='sd,cube0.5,cube0.5r,dpp0.5,grid,dpp2d120a1'); ap.add_argument('--ref', default='sd')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'sub3', 'depth_tercile_micro_%s.md' % time.strftime('%Y-%m-%d')))
    ap.add_argument('--append', default='', help='also append the table to this md (e.g. results/sub3/cube0.5r_micro_2026-11-09.md)')
    a = ap.parse_args(); names = a.pickers.split(','); fns = {n: pickers.parse(n) for n in names}
    micro = json.load(open(os.path.join(ROOT, 'splits.json')))['micro']
    S = {n: [] for n in names}; nf = 0
    for fid, frames in micro.items():
        C = Cands(fid.replace('/', '_'), a.det)
        for q in frames:
            u, v, s, z, _ = C.frame(q); nf += 1
            for n in names:
                S[n].append(frame_stats(u, v, z, fns[n](u, v, s, z, a.K)))
    A = {n: np.array(S[n], float) for n in names}      # frames x (near share, depth std, on-plane); nan where undefined
    ref = A[a.ref]
    L = ['# Depth placement of the picks on MICRO (K %d, %d frames, cache only, %s)' % (a.K, nf, time.strftime('%Y-%m-%d %H:%M')), '',
         'O4 / R20 (dispatch 2026-11-10 12:30 SUB1): nearest depth tercile = the closest third of the frame\'s candidates with depth; share = fraction of a picker\'s '
         'picks (with depth) in it. Paired per-frame differences vs %s; sign counts = frames where the picker\'s value is above / below %s (ties excluded).' % (a.ref, a.ref), '',
         '| picker | near-tercile share | diff vs %s | frames above / below | pick-depth std (m) | diff vs %s | frames above / below | on-plane share | diff vs %s |' % (a.ref, a.ref, a.ref),
         '|---|---|---|---|---|---|---|---|---|']
    res = {}
    for n in names:
        X = A[n]; row = ['%s' % n]
        for j in range(3):
            ok = ~np.isnan(X[:, j]) & ~np.isnan(ref[:, j]); d = X[ok, j] - ref[ok, j]
            up, dn = int((d > 1e-9).sum()), int((d < -1e-9).sum())
            res[(n, j)] = (float(np.nanmean(X[:, j])), float(d.mean()) if len(d) else float('nan'), up, dn, int(ok.sum()))
            row += ['%.3f' % res[(n, j)][0], '%+.4f' % res[(n, j)][1]] + (['%d / %d' % (up, dn)] if j < 2 else [])
        L.append('| ' + ' | '.join(row) + ' |')
    L += ['', 'Frames with a defined value: near share and depth std need >= 3 candidates with depth and >= 1 pick with depth; on-plane needs a plane fit (bench.onplane).']
    if 'cube0.5' in names and a.ref == 'sd':
        m, d, up, dn, nok = res[('cube0.5', 0)]; sd_share = res[('sd', 0)][0]
        gap = sd_share - m; same = max(up, dn)
        met = gap >= 0.05 and dn >= 300
        L += ['', '## Pre-registered reading', '',
              '- sd nearest-tercile share %.3f vs cube0.5 %.3f: gap %+.3f (rule >= 0.05: %s); frames with cube0.5 below sd %d / %d (rule >= 300 / 393: %s)' % (
                  sd_share, m, gap, 'met' if gap >= 0.05 else 'NOT met', dn, nok, 'met' if dn >= 300 else 'NOT met'),
              '- verdict: %s' % ('score ranking concentrates picks near-field on MICRO, voxel caps spread depth (descriptive, [T], 393 frames, no fail outcomes)' if met
                                  else 'no near-field concentration on MICRO; the O4 mechanism paragraph is dropped')]
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
    print('\n'.join(L)); print('->', a.out)
    if a.append:
        open(a.append, 'a', encoding='utf-8', newline='\n').write('\n## O4 depth placement (engineer %s, tools/depth_tercile_micro.py)\n\n' % time.strftime('%Y-%m-%d') + '\n'.join(L[4:]) + '\n')
        print('appended ->', a.append)


if __name__ == '__main__':
    main()
