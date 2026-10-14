"""Convert a fetched Hypersim scene camera (tools/hypersim_fetch.py) into the kpbench flight layout.

  python tools/hypersim_convert.py ai_001_001 [--cam cam_00] [--raw local_data/hypersim_raw] [--out local_data/hypersim]

Output <out>/<scene>/: image_left/%06d_left.png (640 x 480), depth_left/%06d_left_depth.npy (planar depth z in metres,
no depth = 1e4), pose_left.txt (TartanAir format: tx ty tz qx qy qz qw, body frame, so data.cam_pose works unchanged),
intrinsics.txt (fx fy cx cy at 640 x 480; run kpbench with KPBENCH_INTR set to it), frames.txt (Hypersim frame ids).
Hypersim (standard camera, 1024 x 768, 60 deg horizontal FOV): depth_meters = distance to the optical centre,
camera axes x right, y up, z backward; positions in asset units (x meters_per_asset_unit).
"""
import argparse, glob, os, re, sys
import numpy as np, cv2, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpbench.data import R_BC, quat_to_R

W0, H0, W, H = 1024, 768, 640, 480
F0 = (W0 / 2) / np.tan(np.radians(30))          # 886.81 px at 1024 x 768
S = W / W0
NO_DEPTH = 1e4


def R_to_quat(R):
    """rotation matrix -> (qx, qy, qz, qw)"""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1); qw = s / 4
        qx, qy, qz = (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]); q = np.zeros(3)
        q[i] = s / 4; q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s
        qw = (R[k, j] - R[j, k]) / s; qx, qy, qz = q
    return qx, qy, qz, qw


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('scene'); ap.add_argument('--cam', default='cam_00')
    root = os.path.join(os.path.dirname(__file__), '..', '..', 'local_data')
    ap.add_argument('--raw', default=os.path.join(root, 'hypersim_raw')); ap.add_argument('--out', default=os.path.join(root, 'hypersim'))
    a = ap.parse_args()
    src = os.path.join(a.raw, a.scene); dst = os.path.join(a.out, a.scene)
    mpau = float(dict(l.strip().split(',') for l in open(os.path.join(src, '_detail', 'metadata_scene.csv')) if ',' in l)['meters_per_asset_unit'])
    pos = h5py.File(os.path.join(src, '_detail', a.cam, 'camera_keyframe_positions.hdf5'))['dataset'][:] * mpau
    ori = h5py.File(os.path.join(src, '_detail', a.cam, 'camera_keyframe_orientations.hdf5'))['dataset'][:]
    kfi = {int(fid): k for k, fid in enumerate(h5py.File(os.path.join(src, '_detail', a.cam, 'camera_keyframe_frame_indices.hdf5'))['dataset'][:])}
    jp = {int(re.search(r'frame\.(\d+)', p).group(1)): p for p in glob.glob(os.path.join(src, 'images', 'scene_%s_final_preview' % a.cam, '*.tonemap.jpg'))}
    dp = {int(re.search(r'frame\.(\d+)', p).group(1)): p for p in glob.glob(os.path.join(src, 'images', 'scene_%s_geometry_hdf5' % a.cam, '*.depth_meters.hdf5'))}
    ids = sorted(set(jp) & set(dp) & set(kfi))
    gaps = [i for i in range(ids[0], ids[-1] + 1) if i not in ids]
    print(a.scene, a.cam, 'frames', len(ids), 'range', ids[0], ids[-1], 'gaps', gaps[:20], 'mpau', mpau)
    os.makedirs(os.path.join(dst, 'image_left'), exist_ok=True); os.makedirs(os.path.join(dst, 'depth_left'), exist_ok=True)
    v, u = np.mgrid[0:H0, 0:W0].astype(np.float64)
    ray = np.sqrt((u + 0.5 - W0 / 2) ** 2 + (v + 0.5 - H0 / 2) ** 2 + F0 ** 2) / F0   # distance / planar depth
    poses = []
    for f, i in enumerate(ids):
        img = cv2.imread(jp[i]); cv2.imwrite(os.path.join(dst, 'image_left', '%06d_left.png' % f), cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))
        d = h5py.File(dp[i])['dataset'][:].astype(np.float64); z = d / ray
        z = np.where(np.isfinite(z) & (z > 0), z, NO_DEPTH)
        np.save(os.path.join(dst, 'depth_left', '%06d_left_depth.npy' % f), cv2.resize(z, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        Rcv = ori[kfi[i]] @ np.diag([1.0, -1.0, -1.0])                 # Hypersim (x right, y up, z back) -> (x right, y down, z forward)
        q = R_to_quat(Rcv @ R_BC.T)
        assert np.allclose(quat_to_R(*q) @ R_BC, Rcv, atol=1e-6)
        poses.append(list(pos[kfi[i]]) + list(q))
    np.savetxt(os.path.join(dst, 'pose_left.txt'), np.array(poses), fmt='%.8f')
    open(os.path.join(dst, 'frames.txt'), 'w').write('\n'.join(map(str, ids)) + '\n')
    open(os.path.join(dst, 'intrinsics.txt'), 'w').write('%.6f %.6f %.6f %.6f\n' % (F0 * S, F0 * S, W / 2, H / 2))
    step = np.linalg.norm(np.diff(pos[[kfi[i] for i in ids]], axis=0), axis=1)
    print('done', dst, 'fx %.3f' % (F0 * S), 'median step %.3f m' % np.median(step), 'path %.1f m' % step.sum())
