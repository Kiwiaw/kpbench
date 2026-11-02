"""Restore the raw frames of a split from the processed TartanAir copy: symlinks only, nothing deleted or overwritten.

  python tools/restore_frames.py --split dev|held_out [--apply]      (Euler login node; without --apply = dry run)

Why: /cluster/scratch purges files not touched for a while; on 2 Nov 2026 3 of 12 DEV and 19 of 19 HELD_OUT flights had lost depth
and image frames (pose_left.txt, candidate caches and labels intact). Source = KPBENCH_FRESH_SRC as kpbench.fresh
(<env>/Hard/<traj>/<i>_depth.npy, <i>_rgb.png, <i>_cam.npz; the 31 flights are the Hard trajectories of hospital, abandonedfactory,
oldtown). Per flight: the source frame count must equal the pose count and 21 source poses must match pose_left.txt (sign-invariant,
1e-6), else the flight is skipped with a message; then every missing depth_left/<i>_left_depth.npy and image_left/<i>_left.png
becomes a symlink into the source (the FRESH flights are built the same way, kpbench.fresh.link). Existing files are never touched.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

ENV_SRC = {'hospital': 'hospital/Hard', 'factory': 'abandonedfactory/Hard', 'oldtown': 'oldtown/Hard'}


def check(F, src, step=20, tol=1e-6):
    """-> (source frames, worst pose mismatch over every step-th frame) using kpbench.fresh.pose_row on <i>_cam.npz"""
    from kpbench.fresh import pose_row
    from kpbench.data import quat_to_R
    n = 0
    while os.path.isfile(os.path.join(src, '%06d_cam.npz' % n)):
        n += 1
    worst = 0.0
    for i in range(0, min(n, len(F.poses)), step):
        r = np.array(pose_row(np.load(os.path.join(src, '%06d_cam.npz' % i))['camera_pose'])); p = F.poses[i][:7]
        worst = max(worst, float(np.abs(r[:3] - p[:3]).max()), float(np.abs(quat_to_R(*r[3:7]) - quat_to_R(*p[3:7])).max()))
    return n, worst


def restore(fid, apply=False):
    from kpbench.data import Flight
    from kpbench.fresh import SRC
    F = Flight(fid); env, traj = fid.split('/'); src = os.path.join(SRC, ENV_SRC[env], traj)
    if not os.path.isdir(src):
        return '%-15s SKIP source %s missing' % (fid, src)
    n, worst = check(F, src)
    if n != len(F.poses) or worst > 1e-6:
        return '%-15s SKIP source %d frames vs %d poses, worst pose diff %.1e' % (fid, n, len(F.poses), worst)
    todo = []
    for sub, s_fmt, d_fmt in (('depth_left', '%06d_depth.npy', '%06d_left_depth.npy'), ('image_left', '%06d_rgb.png', '%06d_left.png')):
        d = os.path.join(F.dir, sub)
        for i in range(n):
            b = os.path.join(d, d_fmt % i)
            if not os.path.exists(b) and not os.path.islink(b):
                todo.append((os.path.join(src, s_fmt % i), b))
    if apply:
        for a, b in todo:
            os.makedirs(os.path.dirname(b), exist_ok=True); os.symlink(a, b)
    return '%-15s poses %4d source %4d match %.1e: %d links %s' % (fid, len(F.poses), n, worst, len(todo), 'made' if apply else 'to make (dry run)')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='dev', choices=['dev', 'held_out']); ap.add_argument('--apply', action='store_true')
    ap.add_argument('--flights', default='', help='comma list, default = the whole split'); a = ap.parse_args()
    from kpbench.cache import REPO
    fl = a.flights.split(',') if a.flights else json.load(open(os.path.join(REPO, 'splits.json')))[a.split]
    for f in fl:
        print(restore(f, a.apply), flush=True)


if __name__ == '__main__':
    main()
