"""T33 plumbing smoke of the picks dump (SUB3, dispatch 2026-11-19T13:15): do rows written with --save-picks carry what
kpbench.pickarms reads, for all 5 arms at K100?

  python tools/picks_smoke.py [--flight hospital/P041] [--matcher lg]

The laptop has the candidate cache of the flight but no raw flight folder (poses, depth), so the flight folder is a
PLACEHOLDER in a temp dir: identity poses, empty depth files (only their count is read). Pose errors are therefore
meaningless and nothing is stored in the repo: KPBENCH_DATA and KPBENCH_RESULTS point to the temp dir. Checked per arm:
meta.has_picks, one row and one onplane entry per query, 5-field entries with K per-pick labels and depths,
gate.require_picks accepts the run, pickarms.arm_table / paired_field / to_md run on the flat queries. Only flags and
counts are printed; no off / offn / dspan / sep value is shown or kept (the SUB1 sep rule is not committed yet).
--sep (dispatch 2026-11-21T13:15 SUB3): the three SUB1 arms (dpp0.5, sd, cube0.5) instead, then kpbench.sepscore --check
reads the dump (format only: validates arms, picks, query sets, off column; computes and prints no value).
"""
import argparse, gzip, json, os, subprocess, sys, tempfile, shutil
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ARMS = ['dpp0.5', 'dppr0.317a1.6b2.01t0.9', 'cube0.5', 'grid', 'cube0.359']   # results/sub3/t33_prereg.md amendment 18 Nov, first = reference


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--flight', default='hospital/P041'); ap.add_argument('--matcher', default='lg'); ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--sep', action='store_true', help='SUB1 arms + kpbench.sepscore --check')
    a = ap.parse_args(); ARMS = ['dpp0.5', 'sd', 'cube0.5'] if a.sep else globals()['ARMS']; tmp = tempfile.mkdtemp(prefix='kpb_picks_smoke_'); sl = a.flight.replace('/', '_')
    N = len(np.load(os.path.join(REPO, 'cache', sl + '_raco.npz'))['n'])
    sys.path.insert(0, REPO)
    os.environ.update(KPBENCH_DATA=os.path.join(tmp, 'data'), KPBENCH_RESULTS=os.path.join(tmp, 'results'))
    from kpbench import data
    env, traj = a.flight.split('/'); fd = os.path.join(tmp, 'data', data.CANDIDATES[env][0].format(t=traj)); os.makedirs(os.path.join(fd, 'depth_left'))
    np.savetxt(os.path.join(fd, 'pose_left.txt'), np.tile([0, 0, 0, 0, 0, 0, 1.0], (N, 1)))
    for f in range(N):
        open(os.path.join(fd, 'depth_left', '%06d_left_depth.npy' % f), 'wb').close()
    ok = True
    try:
        for p in ARMS:
            r = subprocess.run([sys.executable, '-m', 'kpbench.run', '--det', 'raco', '--picker', p, '--K', str(a.K), '--map', 'same', '--flights', a.flight,
                                '--matcher', a.matcher, '--save-picks', '--tag', 'picks'], cwd=REPO, capture_output=True, text=True)
            if r.returncode:
                print('RUN FAILED', p, r.stderr[-600:]); ok = False
        from kpbench import pickarms
        from kpbench.gate import require_picks, labels
        from kpbench.run import run_id
        lab = labels(a.flight); T = {}
        for p in ARMS:
            rid = run_id('raco', p, a.K, 'same', 'picks', a.matcher); d = os.path.join(tmp, 'results', 'runs', rid)
            if not os.path.isfile(os.path.join(d, sl + '.json.gz')):
                print('%s: no row file' % rid); ok = False; continue
            meta = json.load(open(os.path.join(d, 'meta.json'))); j = json.load(gzip.open(os.path.join(d, sl + '.json.gz'), 'rt'))
            op = j.get('onplane') or {}; q = [r[1] for r in j['rows']]
            full = [e for e in op.values() if len(e) >= 5 and e[3] is not None and len(e[3]) == len(e[4])]
            kpick = sorted(set(len(e[3]) for e in full))
            require_picks(rid, [a.flight])
            T[p] = pickarms.arm_table(rid, [a.flight]); nflat = sum((lab.get(str(x), {}).get('plane_frac') or -1) >= 0.85 for x in q)
            filled = [sum(v[i] is not None for v in T[p].values()) for i in (1, 2, 3)]
            good = bool(meta.get('has_picks')) and len(op) == len(q) == N - len(range(0, N, data.KF)) and len(full) == len(q) and bool(j.get('picks')) and len(T[p]) == nflat and min(filled) > 0
            ok = ok and good
            print('%s: has_picks %s, rows %d, onplane entries %d (5-field %d, picks per entry %s..%s), flat read by pickarms %d of %d, off/offn/dspan filled %s -> %s'
                  % (rid, meta.get('has_picks'), len(q), len(op), len(full), kpick[0] if kpick else '-', kpick[-1] if kpick else '-', len(T[p]), nflat, filled, 'OK' if good else 'BAD'))
        if len(T) == len(ARMS):
            keys = sorted(set.intersection(*(set(t) for t in T.values()))); n = []
            for p in ARMS[1:]:
                pf = [pickarms.paired_field(T[p], T[ARMS[0]], keys, i, n_boot=200) for i in (1, 2, 3)]
                n.append([x['n'] if x else 0 for x in pf]); _ = pickarms.sep(T[p])     # computed, never printed
            md = pickarms.to_md(dict(ref=ARMS[0], split='smoke', arms=[dict(arm=p, n_flat=len(T[p]), fail=0, sep=None, median=dict(off=None, offn=None, dspan=None)) for p in ARMS]))
            good = all(min(x) > 0 for x in n) and md.count('\n') == 4 + len(ARMS)
            ok = ok and good
            print('paired read-out vs %s: paired n per field %s, md table rows %d -> %s' % (ARMS[0], n, len(ARMS), 'OK' if good else 'BAD'))
        else:
            ok = False
        if a.sep and ok:
            from kpbench import sepscore
            try:
                sepscore.main(['--arms', ','.join(run_id('raco', p, a.K, 'same', 'picks', a.matcher) for p in ARMS), '--flights', a.flight, '--check'])
            except SystemExit as e:
                print('sepscore --check:', e); ok = False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('picks smoke (%s, %d frames, placeholder poses, K%d, %s):' % (a.flight, N, a.K, a.matcher), 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
