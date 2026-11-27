"""T59 smoke of the optional inlier-index field (--save-inliers, needs --save-picks), dispatch 2026-11-27 13:20 SUB3.

  python tools/inlier_smoke.py [--flight hospital/P041] [--picker dpp0.5] [--K 100] [--matcher lg]

Same placeholder flight folder as tools/picks_smoke.py (identity poses, empty depth files: pose errors are meaningless, nothing
is kept; KPBENCH_DATA / KPBENCH_RESULTS point to a temp dir). Three runs of one picker: no flag, --save-picks, --save-picks
--save-inliers. Checked: rows byte-identical in all three (json text), onplane fields 1-5 identical between the two picks runs,
the 6th field = sorted unique pick indices in [0, n_sel) with length n_inl for posed queries, None for no-pose queries, and
the indexed picks are a subset of the dumped pixels. Prints counts only, no value.
"""
import argparse, gzip, json, os, subprocess, sys, tempfile, shutil
import numpy as np
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--flight', default='hospital/P041'); ap.add_argument('--picker', default='dpp0.5')
    ap.add_argument('--K', type=int, default=100); ap.add_argument('--matcher', default='lg'); a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix='kpb_inl_smoke_'); sl = a.flight.replace('/', '_')
    N = len(np.load(os.path.join(REPO, 'cache', sl + '_raco.npz'))['n']); sys.path.insert(0, REPO)
    os.environ.update(KPBENCH_DATA=os.path.join(tmp, 'data'), KPBENCH_RESULTS=os.path.join(tmp, 'results'))
    from kpbench import data
    from kpbench.run import run_id
    env, traj = a.flight.split('/'); fd = os.path.join(tmp, 'data', data.CANDIDATES[env][0].format(t=traj)); os.makedirs(os.path.join(fd, 'depth_left'))
    np.savetxt(os.path.join(fd, 'pose_left.txt'), np.tile([0, 0, 0, 0, 0, 0, 1.0], (N, 1)))
    for f in range(N):
        open(os.path.join(fd, 'depth_left', '%06d_left_depth.npy' % f), 'wb').close()
    J = {}; ok = True
    try:
        for tag, fl in (('off', []), ('picks', ['--save-picks']), ('inl', ['--save-picks', '--save-inliers'])):
            r = subprocess.run([sys.executable, '-m', 'kpbench.run', '--det', 'raco', '--picker', a.picker, '--K', str(a.K), '--map', 'same',
                                '--flights', a.flight, '--matcher', a.matcher, '--tag', tag] + fl, cwd=REPO, capture_output=True, text=True)
            if r.returncode:
                print('RUN FAILED', tag, r.stderr[-600:]); return print('inlier smoke: FAIL')
            d = os.path.join(tmp, 'results', 'runs', run_id('raco', a.picker, a.K, 'same', tag, a.matcher))
            J[tag] = (json.load(gzip.open(os.path.join(d, sl + '.json.gz'), 'rt')), json.load(open(os.path.join(d, 'meta.json'))))
        rows = [json.dumps(J[t][0]['rows']) for t in J]; same_rows = rows[0] == rows[1] == rows[2]
        P, I = J['picks'][0], J['inl'][0]
        same5 = all(P['onplane'][q] == I['onplane'][q][:5] for q in P['onplane']) and P['picks'] == I['picks']
        R = {str(r[1]): r for r in I['rows']}; good = posed = nopose = 0
        for q, e in I['onplane'].items():
            r = R[q]; idx = e[5] if len(e) == 6 else 'missing'
            if r[2] is None:
                nopose += 1; good += idx is None
            else:
                posed += 1
                good += isinstance(idx, list) and len(idx) == r[6] == len(set(idx)) and idx == sorted(idx) and all(0 <= i < r[4] for i in idx) \
                    and len(I['picks'][q]) == r[4]
        nq = len(I['rows']); flags = (J['off'][1].get('has_inliers'), J['picks'][1].get('has_inliers'), J['inl'][1].get('has_inliers'))
        ok = same_rows and same5 and good == nq and flags == (False, False, True)
        print('%s K%d %s, %d queries (%d posed, %d no pose): rows byte-identical off/picks/inliers %s; onplane fields 1-5 + picks identical %s; '
              'valid 6th field %d of %d; meta has_inliers %s' % (a.picker, a.K, a.matcher, nq, posed, nopose, same_rows, same5, good, nq, flags))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('inlier smoke (%s, %d frames, placeholder poses):' % (a.flight, N), 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
