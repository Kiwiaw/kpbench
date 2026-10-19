"""Convert a TUM RGB-D sequence into the kpbench flight layout (flight id tum/<short>, e.g. tum/fr1_desk2).

  python tools/tum_convert.py <tum_root>/rgbd_dataset_freiburg1_desk2 --out <KPBENCH_DATA>/kpbench_tum [--stride 3]

Output <out>/<short>/: image_left/%06d_left.png, depth_left/%06d_left_depth.npy (z in metres, no depth = 1e4),
pose_left.txt (TartanAir format tx ty tz qx qy qz qw, body frame, so data.cam_pose works unchanged), intrinsics.txt,
frames.txt (rgb timestamps). Every sequence is undistorted and resampled to ONE pinhole camera (fx = fy = 560, cx 320,
cy 240 at 640 x 480; ~7 % narrower than the Kinect, so no black border), so all three Freiburg cameras run in one process
with KPBENCH_INTR="560 560 320 240". Depth (registered to RGB by the dataset, factor 5000) uses the same map, nearest pixel.
Frames: every <stride>-th rgb frame that has a depth frame within 20 ms and a ground-truth pose within 10 ms
(nearest sample; mocap at 100 Hz). Ground truth = colour camera optical centre, camera-to-world (x right, y down, z fwd).
Calibration: TUM RGB-D ROS default intrinsics page (Sturm et al. 2012).
"""
import argparse, os, re, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpbench.data import R_BC, quat_to_R

W, H = 640, 480
KT = np.array([[560.0, 0, 320.0], [0, 560.0, 240.0], [0, 0, 1]])
CALIB = {'freiburg1': ([517.3, 516.5, 318.6, 255.3], [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]),
         'freiburg2': ([520.9, 521.0, 325.1, 249.7], [0.2312, -0.7849, -0.0033, -0.0001, 0.9172]),
         'freiburg3': ([535.4, 539.2, 320.1, 247.6], [0, 0, 0, 0, 0])}
NO_DEPTH = 1e4


def read_list(fn):
    """TUM list file -> (timestamps (n,), rest-of-line lists)"""
    ts, rest = [], []
    for l in open(fn):
        if l.startswith('#') or not l.strip():
            continue
        p = l.split(); ts.append(float(p[0])); rest.append(p[1:])
    return np.array(ts), rest


def nearest(t, ts):
    i = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
    i = np.where(np.abs(ts[i - 1] - t) <= np.abs(ts[i] - t), i - 1, i)
    return i, np.abs(ts[i] - t)


def R_to_quat(R):
    """rotation matrix -> (qx, qy, qz, qw)"""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1); qw = s / 4
        return (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, qw
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]); q = np.zeros(3)
    q[i] = s / 4; q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s
    return q[0], q[1], q[2], (R[k, j] - R[j, k]) / s


def short_name(seq):
    m = re.match(r'rgbd_dataset_freiburg(\d)_(.*)', os.path.basename(seq.rstrip('/')))
    return 'fr%s_%s' % (m.group(1), m.group(2)), 'freiburg' + m.group(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('seq'); ap.add_argument('--out', required=True); ap.add_argument('--stride', type=int, default=3)
    a = ap.parse_args()
    name, cam = short_name(a.seq); dst = os.path.join(a.out, name)
    if os.path.isfile(os.path.join(dst, 'pose_left.txt')):
        print('exists', dst); sys.exit(0)
    (fx, fy, cx, cy), dist = CALIB[cam]
    K0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    mx, my = cv2.initUndistortRectifyMap(K0, np.array(dist), None, KT, (W, H), cv2.CV_32FC1)
    trgb, rrgb = read_list(os.path.join(a.seq, 'rgb.txt')); tdep, rdep = read_list(os.path.join(a.seq, 'depth.txt'))
    tgt, rgt = read_list(os.path.join(a.seq, 'groundtruth.txt')); gt = np.array(rgt, float)
    idd, dd = nearest(trgb, tdep); ig, dg = nearest(trgb, tgt)
    keep = np.flatnonzero((dd <= 0.02) & (dg <= 0.01) & np.all(np.isfinite(gt[ig]), 1))[::a.stride]
    os.makedirs(os.path.join(dst, 'image_left'), exist_ok=True); os.makedirs(os.path.join(dst, 'depth_left'), exist_ok=True)
    poses = []
    for f, i in enumerate(keep):
        img = cv2.imread(os.path.join(a.seq, rrgb[i][0]))
        cv2.imwrite(os.path.join(dst, 'image_left', '%06d_left.png' % f), cv2.remap(img, mx, my, cv2.INTER_LINEAR))
        d = cv2.imread(os.path.join(a.seq, rdep[idd[i]][0]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
        z = cv2.remap(d, mx, my, cv2.INTER_NEAREST, borderValue=0)
        np.save(os.path.join(dst, 'depth_left', '%06d_left_depth.npy' % f), np.where(z > 0, z, NO_DEPTH).astype(np.float32))
        t, qc = gt[ig[i], :3], gt[ig[i], 3:7]
        Rc = quat_to_R(*qc); q = R_to_quat(Rc @ R_BC.T)
        assert np.allclose(quat_to_R(*q) @ R_BC, Rc, atol=1e-6)
        poses.append(list(t) + list(q))
    P = np.array(poses)
    np.savetxt(os.path.join(dst, 'pose_left.txt'), P, fmt='%.8f')
    open(os.path.join(dst, 'frames.txt'), 'w').write('\n'.join('%.6f' % trgb[i] for i in keep) + '\n')
    open(os.path.join(dst, 'intrinsics.txt'), 'w').write('%.6f %.6f %.6f %.6f\n' % (KT[0, 0], KT[1, 1], KT[0, 2], KT[1, 2]))
    step = np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1)
    print('done', dst, 'frames', len(keep), 'of', len(trgb), 'median step %.3f m' % np.median(step), 'path %.1f m' % step.sum())
