"""SUB1 E-d (EPnPU-style pose covariance trace) + SUB3 E-c (dq3 vs cube0.5 fail %) on MICRO (engineer 2026-11-12, dispatch 16:25).

  python tools/micro_epnpu_dq3.py [--pickers sd,grid,cube0.5,dpp0.5,dq3] [--K 100]

Runs the kpbench.evaluate protocol per picker (same code path as evaluate.eval_flight, copied so the matched 2D/3D points and the
true pose are kept per query) and adds epnpu_t per query: at the TRUE pose, matched points with true-pose reprojection < 3 px;
per point 2D covariance S_i = I + s_X^2 (dp/dXc)(dp/dXc)^T, s_X^2 = (z / f)^2 + (0.005 z^2)^2 (1 px keyframe noise back-projected +
depth noise ~ z^2, isotropic simplification of Vakhitov et al. CVPR 2021); information sum J_i^T S_i^-1 J_i (J = metrics.pose_cov
Jacobian); epnpu_t = log10 sqrt(trace of the camera-centre block) in metres; < 6 points or singular -> +inf.
Writes results/sub1/epnpu_micro.md and results/sub3/dq3_micro.md (appends below their pre-registered readings).
"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kpbench.data import Flight, to3d, DEPTH_MAX, FX, FY, CX, CY
from kpbench.cache import Cands, REPO
from kpbench.pickers import parse as parse_picker
from kpbench.bench import match_lightglue, pnp, repeat, COLS
from kpbench.evaluate import checked_pick, query_sets
from kpbench.splits import FLAT
from kpbench.stats import fail_flags, block_ids, paired_diff
from depth_quota_micro import pick_dq


def epnpu_t(obj, img, Rq, tq, sig_d=0.005):
    """log10 sqrt(trace centre covariance) [m] at the true pose with 2D + propagated 3D noise; inf when < 6 true inliers"""
    if len(obj) < 6:
        return np.inf
    Xc = (np.asarray(obj, float) - tq) @ Rq; R = Rq.T                     # world -> camera
    ok = Xc[:, 2] > 1e-3
    pr = np.stack([FX * Xc[:, 0] / np.where(ok, Xc[:, 2], 1) + CX, FY * Xc[:, 1] / np.where(ok, Xc[:, 2], 1) + CY], 1)
    ok &= np.linalg.norm(pr - np.asarray(img, float), axis=1) < 3
    Xc = Xc[ok]
    if len(Xc) < 6:
        return np.inf
    x, y, z = Xc.T
    dpu = np.stack([FX / z, 0 * z, -FX * x / z ** 2], 1); dpv = np.stack([0 * z, FY / z, -FY * y / z ** 2], 1)
    Ju = np.concatenate([np.cross(Xc, dpu), -dpu @ R], 1); Jv = np.concatenate([np.cross(Xc, dpv), -dpv @ R], 1)
    s2 = (z / FX) ** 2 + (sig_d * z ** 2) ** 2; info = np.zeros((6, 6))
    for k in range(len(z)):
        D = np.stack([dpu[k], dpv[k]]); S = np.eye(2) + s2[k] * D @ D.T; J = np.stack([Ju[k], Jv[k]])
        info += J.T @ np.linalg.solve(S, J)
    try:
        C = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return np.inf
    tr = np.trace(C[3:, 3:])
    return float(0.5 * np.log10(tr)) if tr > 0 else np.inf


def run(pk, K, det='raco'):
    rows, flat, ep = [], [], []
    for fid, qs in query_sets('micro').items():
        F = Flight(fid); C = Cands(F.slug, det); lab = json.load(open(os.path.join(REPO, 'results', 'labels', F.slug + '.json')))
        qs = F.queries if qs is None else qs; need = sorted(set(qs) | {k for q in qs for k in F.near(q)}); sel, MAP, trees = {}, {}, {}
        for f in need:
            u, v, s, z, de = C.frame(f); sel[f] = checked_pick(pk, u, v, s, z, de, K, (F, C, f), False)
        for k in need:
            if k % 10 == 0:
                u, v, s, z, de = C.frame(k); i = sel[k]; i = i[z[i] < DEPTH_MAX]; Rk, tk = F.cam_pose(k)
                MAP[k] = (to3d(u[i], v[i], z[i]) @ Rk.T + tk, de[i], np.stack([u[i], v[i]], 1))
        for q in qs:
            near = F.near(q); Rq, tq = F.cam_pose(q); u, v, s, z, de = C.frame(q); i = sel[q]
            mq, pairs = match_lightglue(np.stack([u[i], v[i]], 1), de[i], [(MAP[k][2], MAP[k][1]) for k in near])
            obj = np.array([MAP[near[ki]][0][pi] for ki, pi in pairs]).reshape(-1, 3); img = np.stack([u[i][mq], v[i][mq]], 1)
            err, rot, ninl, sol = pnp(obj, img, Rq, tq, full=True)
            rep = repeat(F, C, q, i, trees)
            for f in [f for f in trees if f < q]:
                del trees[f]
            rows.append([fid, q, err, rot, int(len(i)), int(len(mq)), ninl] + rep)
            pf = lab[str(q)].get('plane_frac'); flat.append(pf is not None and pf >= FLAT); ep.append(epnpu_t(obj, img, Rq, tq))
    return rows, np.array(flat, bool), np.array(ep)


def auroc(score, y):
    """P(score of a fail > score of a success), ties 1/2 (inf handled as a large value)"""
    s = np.where(np.isfinite(score), score, 1e9); a, b = s[y == 1], s[y == 0]
    if not len(a) or not len(b):
        return None
    return float(((a[:, None] > b[None]).sum() + 0.5 * (a[:, None] == b[None]).sum()) / (len(a) * len(b)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pickers', default='sd,grid,cube0.5,dpp0.5,dq3'); ap.add_argument('--K', type=int, default=100)
    a = ap.parse_args(); R = {}
    for p in a.pickers.split(','):
        t0 = time.time(); pk = pick_dq(3) if p == 'dq3' else parse_picker(p)
        rows, flat, ep = run(pk, a.K); fl = fail_flags(rows)
        R[p] = dict(rows=rows, flat=flat, ep=ep, fail=fl); print(p, 'flat fail %.2f %% of %d, %.0f s' % (100 * fl[flat].mean(), flat.sum(), time.time() - t0), flush=True)
    ps = list(R); flat = R[ps[0]]['flat']
    for p in ps:
        assert [r[1] for r in R[p]['rows']] == [r[1] for r in R[ps[0]]['rows']]
    rep5 = {p: float(np.mean([r[COLS.index('rep5')] for r, f in zip(R[p]['rows'], flat) if f and r[COLS.index('rep5')] is not None])) for p in ps}
    # E-d
    core = [p for p in ('sd', 'grid', 'cube0.5', 'dpp0.5') if p in R]
    E = np.concatenate([R[p]['ep'][flat] for p in core]); Y = np.concatenate([R[p]['fail'][flat] for p in core])
    au = auroc(E, Y); med = {p: float(np.median(np.where(np.isfinite(R[p]['ep'][flat]), R[p]['ep'][flat], 9))) for p in ps}
    order = [p for p in core]; order_ok = all(med[order[k]] > med[order[k + 1]] for k in range(len(order) - 1))
    L = ['', '## Result (%s, %d flat queries, K %d)' % (time.strftime('%Y-%m-%d %H:%M'), int(flat.sum()), a.K), '',
         '| picker | flat fail % (count) | median epnpu_t (log10 m) | share inf (< 6 true inliers) | AUROC within picker | rep5 flat |', '|---|---|---|---|---|---|']
    for p in ps:
        e = R[p]['ep'][flat]; f = R[p]['fail'][flat]; aw = auroc(e, f)
        L.append('| %s | %.2f (%d) | %.3f | %.3f | %s | %.4f |' % (p, 100 * f.mean(), int(f.sum()), med[p], float(np.mean(~np.isfinite(e))), '-' if aw is None else '%.3f' % aw, rep5[p]))
    kill = au is None or au <= 0.5 or not order_ok
    L += ['', 'Pooled AUROC over %s (%d query-arms, %d fails): %s. Median order %s: %s.' % ('/'.join(core), len(Y), int(Y.sum()), '-' if au is None else '%.3f' % au,
          ' > '.join('%s %.3f' % (p, med[p]) for p in core), 'matches sd > grid > cube0.5 > dpp0.5' if order_ok else 'does NOT match sd > grid > cube0.5 > dpp0.5'),
          'Fail order on MICRO (for reference): %s.' % ' > '.join('%s %.2f' % (p, 100 * R[p]['fail'][flat].mean()) for p in sorted(core, key=lambda p: -R[p]['fail'][flat].mean())),
          '', '**Reading: %s**' % ('kill met -> SUB1 stays KILL, no executor run.' if kill else 'MICRO pass -> one FRESH-1 K100 + K400 sub1 run (rules 1-3) may be queued by the executor.'),
          '', 'Tool: tools/micro_epnpu_dq3.py.']
    open(os.path.join(ROOT, 'results', 'sub1', 'epnpu_micro.md'), 'a', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L))
    # E-c
    if 'dq3' in R and 'cube0.5' in R:
        A, B = R['dq3'], R['cube0.5']; fl = flat
        bl = block_ids([r[0] for r in A['rows']], [r[1] for r in A['rows']])
        d = paired_diff(A['fail'][fl], B['fail'][fl], bl[fl], 1000, 0)
        only_a = int(((A['fail'] == 1) & (B['fail'] == 0) & fl).sum()); only_b = int(((A['fail'] == 0) & (B['fail'] == 1) & fl).sum())
        excl = d[1] > 0 or d[2] < 0
        M = ['', '## Result (%s, %d flat queries, K %d)' % (time.strftime('%Y-%m-%d %H:%M'), int(fl.sum()), a.K), '',
             '| arm | flat fail % (count) | rep5 flat |', '|---|---|---|',
             '| dq3 | %.2f (%d) | %.4f |' % (100 * A['fail'][fl].mean(), int(A['fail'][fl].sum()), rep5['dq3']),
             '| cube0.5 | %.2f (%d) | %.4f |' % (100 * B['fail'][fl].mean(), int(B['fail'][fl].sum()), rep5['cube0.5'])]
        M += ['| sd (ref) | %.2f (%d) | %.4f |' % (100 * R['sd']['fail'][fl].mean(), int(R['sd']['fail'][fl].sum()), rep5['sd'])] if 'sd' in R else []
        M += ['', 'Paired dq3 - cube0.5: %+.2f pt [%+.2f, %+.2f] (20-frame blocks, 1000 x seed 0); discordant frames dq3-only fails %d, cube-only fails %d.' % (d + (only_a, only_b)),
              '', '**Reading: %s**' % (('CI excludes 0: ' + ('dq3 fails less -> J4 kill met (depth quota explains the gain).' if d[2] < 0 else 'dq3 fails more -> the 3D cells do more than a depth quota on MICRO.'))
                                       if excl else 'CI covers 0 -> descriptive tie, no verdict (pre-registered; not evidence of equality at ~1.6 pt half-width).'),
              '', 'Tool: tools/micro_epnpu_dq3.py.']
        open(os.path.join(ROOT, 'results', 'sub3', 'dq3_micro.md'), 'a', encoding='utf-8').write('\n'.join(M) + '\n'); print('\n'.join(M))


if __name__ == '__main__':
    main()
