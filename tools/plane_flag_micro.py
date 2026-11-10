"""O3 gated-cube spec check (no images, poses or GPU): can a plane fit on the frame's cached 3D detections stand in for the label?

  python tools/plane_flag_micro.py [--thr 0.85] [--n_fit 300] [--out results/sub3/plane_flag_micro_<date>.md] [--append <md>]

plane_frac_hat = inlier share of the RANSAC dominant plane (pickers.dominant_plane: 300 trials, 30 cm, seed 0) over the strongest
n_fit candidates with depth, i.e. bench.onplane's third output. Flag = plane_frac_hat >= thr; label = results/labels plane_frac >= FLAT
(0.85, from the depth map). Confusion counts on every labelled query of the laptop-cached flights and on the MICRO 393 frames
(all label-flat, so MICRO gives the flag's recall only). Pre-registered (dispatch 2026-11-10 13:25 SUB3): flag agreement >= 0.9
is one of the conditions for building a gated cube (cube0.5 when flagged flat, dpp0.5 otherwise); below it the gated cube is not built.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from kpbench.cache import Cands
from kpbench.data import to3d, DEPTH_MAX
from kpbench.pickers import dominant_plane
from kpbench.splits import FLAT

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def plane_hat(u, v, z, n_fit, tol=0.3):
    ok = np.flatnonzero(z < DEPTH_MAX)[:n_fit]
    if len(ok) < 10:
        return None
    return dominant_plane(to3d(u[ok], v[ok], z[ok]), tol)[2] / len(ok)


def confusion(hat, lab, thr):
    h = hat >= thr; l = lab >= FLAT
    tp, fn, fp, tn = int((h & l).sum()), int((~h & l).sum()), int((h & ~l).sum()), int((~h & ~l).sum())
    return tp, fn, fp, tn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--thr', type=float, default=0.85); ap.add_argument('--n_fit', type=int, default=300)
    ap.add_argument('--det', default='raco'); ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'sub3', 'plane_flag_micro_%s.md' % time.strftime('%Y-%m-%d')))
    ap.add_argument('--append', default=''); a = ap.parse_args()
    micro = json.load(open(os.path.join(ROOT, 'splits.json')))['micro']; mset = {(f, int(q)) for f, qs in micro.items() for q in qs}
    cached = sorted(fn[:-9] for fn in os.listdir(os.path.join(ROOT, 'cache')) if fn.endswith('_%s.npz' % a.det))
    hats, labs, keys = [], [], []
    for slug in cached:
        fid = slug.replace('_', '/', 1); C = Cands(slug, a.det); lab = json.load(open(os.path.join(ROOT, 'results', 'labels', slug + '.json')))
        for k, rec in lab.items():
            q = int(k); pf = rec.get('plane_frac')
            if pf is None or q >= len(C.n) or C.n[q] == 0:
                continue
            u, v, s, z, _ = C.frame(q); h = plane_hat(u, v, z, a.n_fit)
            if h is None:
                continue
            hats.append(h); labs.append(pf); keys.append((fid, q))
    hat, lab = np.array(hats), np.array(labs); inm = np.array([k in mset for k in keys])
    L = ['# Plane-fit flag from cached 3D detections vs the frame label (%s, %d labelled queries of %d flights, MICRO %d, %s)' % (
            a.det, len(hat), len(cached), int(inm.sum()), time.strftime('%Y-%m-%d %H:%M')), '',
         'Flag = RANSAC dominant-plane inlier share over the strongest %d candidates with depth >= %.2f; label = depth-map plane_frac >= %.2f (results/labels). '
         'Flights: %s.' % (a.n_fit, a.thr, FLAT, ', '.join(cached)), '',
         '| set | n | label flat | flag flat | both (TP) | label only (FN) | flag only (FP) | neither (TN) | agreement | flat recall | flat precision |', '|---|---|---|---|---|---|---|---|---|---|---|']
    res = {}
    for name, m in (('all labelled queries', np.ones(len(hat), bool)), ('MICRO 393 (label-flat by construction)', inm)):
        tp, fn, fp, tn = confusion(hat[m], lab[m], a.thr); n = m.sum()
        agr = (tp + tn) / n; rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1); res[name] = agr
        L.append('| %s | %d | %d | %d | %d | %d | %d | %d | %.3f | %.3f | %.3f |' % (name, n, tp + fn, tp + fp, tp, fn, fp, tn, agr, rec, prec))
    # descriptive: agreement as a function of the flag threshold (the label threshold stays 0.85)
    grid = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    L += ['', 'Agreement on all labelled queries by flag threshold (label fixed at %.2f): %s' % (FLAT, ', '.join('%.2f -> %.3f' % (t, (lambda c: (c[0] + c[3]) / len(hat))(confusion(hat, lab, t))) for t in grid)),
          'Spearman rank correlation of plane_frac_hat with the label plane_frac: %.3f; median |hat - label| %.3f.' % (
              __import__('scipy.stats', fromlist=['spearmanr']).spearmanr(hat, lab).correlation, float(np.median(np.abs(hat - lab))))]
    agr = res['all labelled queries']
    L += ['', '## Pre-registered reading', '',
          '- flag agreement on all labelled queries %.3f (condition >= 0.9: %s); on MICRO the flag recalls %.3f of the 393 label-flat frames' % (
              agr, 'met' if agr >= 0.9 else 'NOT met', res['MICRO 393 (label-flat by construction)']),
          '- %s' % ('the detection-based flag is usable for a gated cube (cube0.5 when flagged, dpp0.5 otherwise); still a spec: the oracle composite guards (SUB2 r24_strata_fresh1.md) must clear first, then a gate on a gate-capable set'
                    if agr >= 0.9 else 'the gated cube is NOT built: the cheap plane fit on detections disagrees with the label too often; R28 is answered by printing the cost')]
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n'); print('\n'.join(L)); print('->', a.out)
    if a.append:
        open(a.append, 'a', encoding='utf-8', newline='\n').write('\n## O3 gated-cube flag check (engineer %s, tools/plane_flag_micro.py)\n\n' % time.strftime('%Y-%m-%d') + '\n'.join(L[2:]) + '\n')
        print('appended ->', a.append)


if __name__ == '__main__':
    main()
