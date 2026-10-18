"""FRESH split: extra TartanAir trajectories (outside the 31 flights) as kpbench flights, for bigger-sample retests.

  python -m kpbench.fresh link --flights oldtown_easy/P000,hospital_easy/P025     # build the flight folders
  python -m kpbench.fresh verify                                                   # adapter check on a known flight
  python -m kpbench.fresh select --order euler/fresh_order.json                    # pick the FRESH set -> splits_fresh.json

Source: a processed TartanAir copy (env KPBENCH_FRESH_SRC, layout <env>/<Easy|Hard>/<traj>/<i>_rgb.png, <i>_depth.npy,
<i>_cam.npz with camera_pose = 4x4 camera-to-world, x right / y down / z forward). The source is only read. `link` writes
<KPBENCH_DATA>/kpbench_fresh/<flight env>/<traj>/ with image_left/ and depth_left/ symlinks in the raw TartanAir names
and pose_left.txt (tx ty tz qx qy qz qw, body NED frame, so data.Flight.cam_pose returns camera_pose exactly).
Flight envs: data.FRESH_ENVS (e.g. hospital_easy/P025 = hospital/Easy/P025). They are not in data.FLIGHTS, so the
leaderboard, the audit, splits.json DEV/HELD_OUT and their label manifest are untouched.
select: the pre-registered rule in the order file (see select()), real flat counts from results/labels (plane_frac >= 0.85).
"""
import os, json, math, datetime
import numpy as np
from .data import FRESH_ENVS, R_BC, ROOT, Flight, slug
from .cache import REPO
from .splits import FLAT

SRC = os.environ.get('KPBENCH_FRESH_SRC', '/cluster/project/cvg/data/processed_data/processed_tartanair')
DST = os.path.join(ROOT, 'kpbench_fresh')
SPLITS_FRESH = os.path.join(REPO, 'splits_fresh.json')


def src_dir(fid):
    env, traj = fid.split('/')
    return os.path.join(SRC, FRESH_ENVS[env], traj)


def rot_to_quat(R):
    """rotation matrix -> (qx, qy, qz, qw), inverse of data.quat_to_R"""
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(t + 1); w = s / 4; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]); x = s / 4
        w = (R[2, 1] - R[1, 2]) / s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]); y = s / 4
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]); z = s / 4
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s
    return x, y, z, w


def pose_row(T):
    """4x4 camera-to-world -> pose_left.txt row (body frame: R_body = R_cam @ R_BC^T)"""
    T = np.asarray(T, np.float64)
    return [*T[:3, 3], *rot_to_quat(T[:3, :3] @ R_BC.T)]


def link(fid, src=None, dst=None):
    s = src or src_dir(fid); env, traj = fid.split('/'); d = dst or os.path.join(DST, env, traj)
    n = 0
    while all(os.path.isfile(os.path.join(s, '%06d_%s' % (n, e))) for e in ('rgb.png', 'depth.npy', 'cam.npz')):
        n += 1
    if n == 0:
        raise FileNotFoundError('no frames in %s' % s)
    for sub in ('image_left', 'depth_left'):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    rows = []
    for i in range(n):
        for a, b in ((os.path.join(s, '%06d_rgb.png' % i), os.path.join(d, 'image_left', '%06d_left.png' % i)),
                     (os.path.join(s, '%06d_depth.npy' % i), os.path.join(d, 'depth_left', '%06d_left_depth.npy' % i))):
            if not os.path.islink(b):
                os.symlink(a, b)
        c = np.load(os.path.join(s, '%06d_cam.npz' % i))
        K = c['camera_intrinsics']
        if not np.allclose(K, [[320, 0, 320], [0, 320, 240], [0, 0, 1]], atol=1e-3):
            raise RuntimeError('%s frame %d intrinsics %s' % (fid, i, K.tolist()))
        rows.append(pose_row(c['camera_pose']))
    tmp = os.path.join(d, 'pose_left.txt.tmp')
    np.savetxt(tmp, np.array(rows), fmt='%.9f'); os.replace(tmp, os.path.join(d, 'pose_left.txt'))
    print('linked', fid, n, 'frames ->', d, flush=True)
    return d


def verify(fid='oldtown/P000', src_rel='oldtown/Hard/P000', tmp=None):
    """adapter on a flight we already hold (raw layout) must give the same poses, depth and image"""
    import tempfile
    from . import data
    tmp = tmp or tempfile.mkdtemp(prefix='kpb_fresh_verify_')
    d = link('x/' + fid.split('/')[1], src=os.path.join(SRC, src_rel), dst=os.path.join(tmp, 'v'))
    ours = Flight(fid)
    poses = np.loadtxt(os.path.join(d, 'pose_left.txt')); n = len(poses); worst = 0.0
    for f in range(0, n, max(1, n // 50)):
        R, t = ours.cam_pose(f); Rn = data.quat_to_R(*poses[f, 3:7]) @ R_BC
        worst = max(worst, float(np.abs(R - Rn).max()), float(np.abs(t - poses[f, :3]).max()))
    f = min(5, n - 1)
    dd = float(np.abs(np.load(os.path.join(d, 'depth_left', '%06d_left_depth.npy' % f)) - ours.depth(f)).max())
    from PIL import Image
    di = int(np.abs(np.array(Image.open(os.path.join(d, 'image_left', '%06d_left.png' % f)).convert('RGB')).astype(int)
                    - ours.img(f).astype(int)).max())
    ok = n == ours.N and worst < 1e-4 and dd == 0 and di == 0
    print('verify %s: frames %d vs %d, pose max abs diff %.2e, depth %.1e, image %d -> %s' % (fid, n, ours.N, worst, dd, di,
                                                                                         'PASS' if ok else 'FAIL'))
    return ok


def n_flat(fid):
    d = json.load(open(os.path.join(REPO, 'results', 'labels', slug(fid) + '.json')))
    return sum(1 for v in d.values() if v.get('plane_frac') is not None and v['plane_frac'] >= FLAT), len(d)


def select(order_file):
    """Rule (fixed in the order file before any picker runs on FRESH):
    target = total flat queries wanted. 'take_all' groups enter whole. The rest R = target - their flat count is split over
    the 'fill' groups by weight (ceil(R * w / sum w) each); each fill group takes flights in its listed order until it
    reaches its share. Flat = plane_frac >= 0.85 from results/labels (the normal kpbench label pipeline)."""
    o = json.load(open(order_file)); flights, per = [], {}
    for g, fl in o['take_all'].items():
        flights += fl; per[g] = sum(n_flat(f)[0] for f in fl)
    R = max(0, o['target'] - sum(per.values())); W = sum(v['weight'] for v in o['fill'].values())
    for g, v in o['fill'].items():
        want = math.ceil(R * v['weight'] / W); per[g] = 0
        for f in v['order']:
            if per[g] >= want:
                break
            flights.append(f); per[g] += n_flat(f)[0]
        if per[g] < want:
            raise SystemExit('group %s: %d flat < share %d after its whole order list; extend the pool' % (g, per[g], want))
    nf = {f: n_flat(f) for f in flights}
    s = dict(created=datetime.date.today().isoformat(), source=SRC, order_file=os.path.relpath(order_file, REPO),
             rule=select.__doc__.split('Rule ')[1].strip(), reason=o.get('reason', ''), fresh=flights,
             flat_counts=dict(fresh=sum(v[0] for v in nf.values())), n_queries=dict(fresh=sum(v[1] for v in nf.values())),
             per_group_flat=per, per_flight={f: dict(flat=v[0], queries=v[1]) for f, v in nf.items()})
    return s


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['link', 'verify', 'select']); ap.add_argument('--flights', default='')
    ap.add_argument('--order', default=os.path.join(REPO, 'euler', 'fresh_order.json'))
    ap.add_argument('--write', action='store_true', help='select: write splits_fresh.json (refused if it exists)')
    a = ap.parse_args(argv)
    if a.cmd == 'link':
        for f in [x for x in a.flights.split(',') if x]:
            link(f)
    elif a.cmd == 'verify':
        raise SystemExit(0 if verify() else 1)
    else:
        s = select(a.order); print(json.dumps({k: s[k] for k in ('fresh', 'flat_counts', 'n_queries', 'per_group_flat')}, indent=1))
        if a.write:
            if os.path.isfile(SPLITS_FRESH):
                raise SystemExit('frozen: %s exists' % SPLITS_FRESH)
            json.dump(s, open(SPLITS_FRESH, 'w'), indent=1); print('wrote', SPLITS_FRESH)


if __name__ == '__main__':
    main()
