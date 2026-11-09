"""R27 control check on the cached MICRO candidates (no images, poses or GPU): cube<s>r vs cube<s> at K.

  python tools/cube_r_micro.py [--K 100] [--size 0.5] [--out results/sub3/cube0.5r_micro_<date>.md] [--old <pickers.py of HEAD>]

cube<s>r keeps the cube round robin (cubes served in the order of their strongest member) but shuffles the members inside each
cube with a per-frame seed, so the occupied 3D cells must be identical to cube<s> and only the member per cell may differ.
Per frame: occupied cells of both pick sets (must be equal), Jaccard of the K pick sets (< 1 where a cell holds > 1 candidate),
bench.onplane share of the picks (R20) for both. --old: a copy of the previous pickers.py to confirm cube<s> itself is unchanged.
Kills (dispatch 2026-11-09 13:12 SUB3): any frame with different occupied cells -> variant mis-specified; Jaccard 1.0 on every
frame -> cells singly occupied at K, the within-cell order is moot.
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from kpbench import pickers
from kpbench.bench import onplane
from kpbench.cache import Cands
from kpbench.data import to3d, DEPTH_MAX
from kpbench.splits import FLAT

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def cells(u, v, z, idx, size):
    ok = idx[z[idx] < DEPTH_MAX]
    return set(map(tuple, np.floor(to3d(u[ok], v[ok], z[ok]) / size).astype(int)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=100); ap.add_argument('--size', type=float, default=0.5)
    ap.add_argument('--det', default='raco'); ap.add_argument('--old', default='')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'sub3', 'cube%gr_micro_%s.md' % (0.5, time.strftime('%Y-%m-%d'))))
    a = ap.parse_args(); K = a.K; base = 'cube%g' % a.size; name = base + 'r'
    fb, fr = pickers.parse(base), pickers.parse(name); fo = None
    if a.old:
        # --old = a copy of the previous pickers.py with its relative imports rewritten to kpbench.<module>
        spec = importlib.util.spec_from_file_location('pickers_old', a.old); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        fo = old.parse(base)
    micro = json.load(open(os.path.join(ROOT, 'splits.json')))['micro']
    rows = []; bad_cells = 0; old_diff = 0; t = {base: 0.0, name: 0.0}
    for fid, fr_ in micro.items():
        slug = fid.replace('/', '_'); C = Cands(slug, a.det); lab = json.load(open(os.path.join(ROOT, 'results', 'labels', slug + '.json')))
        for q in fr_:
            u, v, s, z, de = C.frame(q)
            t0 = time.time(); ib = np.asarray(fb(u, v, s, z, K), int); t[base] += time.time() - t0
            t0 = time.time(); ir = np.asarray(fr(u, v, s, z, K), int); t[name] += time.time() - t0
            if fo is not None and not np.array_equal(ib, np.asarray(fo(u, v, s, z, K), int)):
                old_diff += 1
            cb, cr = cells(u, v, z, ib, a.size), cells(u, v, z, ir, a.size)
            bad_cells += cb != cr
            jac = len(set(ib) & set(ir)) / max(len(set(ib) | set(ir)), 1)
            pf = lab[str(q)].get('plane_frac')
            rows.append(dict(flight=fid, frame=int(q), flat=bool(pf is not None and pf >= FLAT), cells=len(cb), same_cells=cb == cr, jaccard=jac,
                             on_b=onplane(u, v, z, ib)[0][0], on_r=onplane(u, v, z, ir)[0][0], ndepth=int((z < DEPTH_MAX).sum())))
    n = len(rows); jac = np.array([r['jaccard'] for r in rows]); onb = np.array([r['on_b'] for r in rows], float); onr = np.array([r['on_r'] for r in rows], float)
    ok = ~np.isnan(onb) & ~np.isnan(onr)
    out = ['# %s vs %s on MICRO (K %d, %d frames, cache only, %s)' % (name, base, K, n, time.strftime('%Y-%m-%d %H:%M')), '',
           'Within-cell order control (R27, STDLoc objection): same cubes in the same order, random member per cube instead of the strongest.', '',
           '- occupied cells identical on %d / %d frames (kill "cells differ": %s)' % (n - bad_cells, n, 'MET' if bad_cells else 'not met'),
           '- pick Jaccard %s vs %s: mean %.3f, median %.3f, min %.3f, frames with Jaccard 1.0: %d / %d (kill "all 1.0": %s)' % (
               name, base, jac.mean(), np.median(jac), jac.min(), int((jac >= 1.0).sum()), n, 'MET' if (jac >= 1.0).all() else 'not met'),
           '- on-plane share of the picks (bench.onplane, frames with a plane fit %d): %s %.3f, %s %.3f, paired diff %+.4f' % (
               int(ok.sum()), base, onb[ok].mean(), name, onr[ok].mean(), (onr[ok] - onb[ok]).mean()),
           '- cells occupied per frame: mean %.1f of K %d (cells with > 1 candidate are where the order matters)' % (np.mean([r['cells'] for r in rows]), K),
           '- picker time: %s %.2f s, %s %.2f s over %d frames' % (base, t[base], name, t[name], n)]
    if fo is not None:
        out.append('- regression: %s picks identical to the previous pickers.py on %d / %d frames' % (base, n - old_diff, n))
    out += ['', '## First 5 frames', '', '| flight | frame | flat | cells | same cells | picks Jaccard | on-plane %s | on-plane %s |' % (base, name), '|---|---|---|---|---|---|---|---|']
    for r in rows[:5]:
        out.append('| %s | %d | %s | %d | %s | %.3f | %s | %s |' % (r['flight'], r['frame'], r['flat'], r['cells'], r['same_cells'], r['jaccard'], r['on_b'], r['on_r']))
    out += ['', 'Flat frames (plane_frac >= %.2f): %d of %d. Per-pick on-plane labels come from the frame\'s dominant plane (RANSAC, 30 cm), not from poses.' % (FLAT, sum(r['flat'] for r in rows), n)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    print('\n'.join(out[:9])); print('->', a.out)
    sys.exit(1 if bad_cells else 0)


if __name__ == '__main__':
    main()
