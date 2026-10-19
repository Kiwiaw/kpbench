"""Hypersim from the shared CVG processed copy (Euler, read-only) into the kpbench flight layout, same output as
tools/hypersim_convert.py (checked on ai_001_001 frames 0 and 5: pose, planar depth and focal identical, image +-1 grey level).

  python tools/hypersim_processed.py --select 12 [--skip ai_001_001,ai_001_002] > scenes.txt   # pre-registered scene rule
  python tools/hypersim_processed.py --scene ai_001_003 --out $KPBENCH_DATA/hypersim

Source <src>/<scene>/cam_00/%06d_{rgb.png, depth.npy, cam.npz}: rgb 1024 x 768 tonemapped, depth = planar z in metres,
cam.npz intrinsics (fx 886.81 at 1024 x 768) and pose = 4 x 4 camera-to-world, OpenCV axes (x right, y down, z forward).
Output <out>/<scene>/: image_left/%06d_left.png (640 x 480), depth_left/%06d_left_depth.npy (no depth = 1e4),
pose_left.txt (TartanAir body-frame quaternions), intrinsics.txt (554.256258 554.256258 320 240), frames.txt.
Scene rule (--select N): scenes in sorted order, camera cam_00 only, frames 0..99 all present (no gaps), frame-0 image
mean >= 20 (drops black renders such as ai_003_001) and finite positive depth on >= 90 % of frame-0 pixels; first N
scenes not in --skip.
"""
import argparse, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpbench.data import R_BC, quat_to_R
from tum_convert import R_to_quat

SRC = '/cluster/project/cvg/data/processed_data/processed_hypersim'
W, H, S, NO_DEPTH, NF = 640, 480, 640 / 1024, 1e4, 100


def ok_scene(src, sc):
    d = os.path.join(src, sc, 'cam_00')
    if not os.path.isdir(d) or any(not os.path.isfile(os.path.join(d, '%06d_rgb.png' % i)) for i in range(NF)):
        return False
    img = cv2.imread(os.path.join(d, '000000_rgb.png')); z = np.load(os.path.join(d, '000000_depth.npy'))
    return img is not None and img.mean() >= 20 and np.mean(np.isfinite(z) & (z > 0)) >= 0.9


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC); ap.add_argument('--select', type=int, default=0)
    ap.add_argument('--skip', default=''); ap.add_argument('--scene'); ap.add_argument('--out')
    a = ap.parse_args()
    if a.select:
        skip, n = set(a.skip.split(',')), 0
        for sc in sorted(os.listdir(a.src)):
            if sc not in skip and ok_scene(a.src, sc):
                print(sc); n += 1
                if n == a.select:
                    break
        sys.exit(0)
    src, dst = os.path.join(a.src, a.scene, 'cam_00'), os.path.join(a.out, a.scene)
    if os.path.isfile(os.path.join(dst, 'pose_left.txt')):
        print('exists', dst); sys.exit(0)
    os.makedirs(os.path.join(dst, 'image_left'), exist_ok=True); os.makedirs(os.path.join(dst, 'depth_left'), exist_ok=True)
    poses = []
    for f in range(NF):
        c = np.load(os.path.join(src, '%06d_cam.npz' % f)); K, T = c['intrinsics'], c['pose'].astype(np.float64)
        assert abs(K[0, 0] * S - 554.256258) < 1e-3 and abs(K[0, 2] - 512) < 1e-3, (a.scene, f, K)
        img = cv2.imread(os.path.join(src, '%06d_rgb.png' % f))
        cv2.imwrite(os.path.join(dst, 'image_left', '%06d_left.png' % f), cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))
        z = np.load(os.path.join(src, '%06d_depth.npy' % f)).astype(np.float64)
        z = np.where(np.isfinite(z) & (z > 0), z, NO_DEPTH)
        np.save(os.path.join(dst, 'depth_left', '%06d_left_depth.npy' % f), cv2.resize(z, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        q = R_to_quat(T[:3, :3] @ R_BC.T)
        assert np.allclose(quat_to_R(*q) @ R_BC, T[:3, :3], atol=1e-5)
        poses.append(list(T[:3, 3]) + list(q))
    P = np.array(poses)
    np.savetxt(os.path.join(dst, 'pose_left.txt'), P, fmt='%.8f')
    open(os.path.join(dst, 'frames.txt'), 'w').write('\n'.join(map(str, range(NF))) + '\n')
    open(os.path.join(dst, 'intrinsics.txt'), 'w').write('554.256258 554.256258 320.000000 240.000000\n')
    step = np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1)
    print('done', dst, 'median step %.3f m' % np.median(step), 'path %.1f m' % step.sum())
