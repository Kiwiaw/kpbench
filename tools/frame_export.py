"""Export a few frames of any sequence into a small kpbench flight folder, for the frame pages (kpbench.viewer --dir).

  python tools/frame_export.py --src kp:hospital/P042 --frame 120 --past 3 --out $SCRATCH/kpbench_frames/tartanair_hospital_P042
  python tools/frame_export.py --src proc:<processed seq dir> --score 20          # list frames by depth variety + flat part

Sources:
  kp:<flight id>   a kpbench flight (data.flight_dir: TartanAir raw, FRESH, hypersim/<scene>, tum/<seq>), copied as is
  proc:<dir>       a CVG processed sequence, either <i:06d>_{rgb.png,depth.npy,cam.npz} (TartanAir) or rgb/ depth/ cam/
                   subfolders (Spring, Point Odyssey; cam npz 'intrinsics'). 960 x 540 frames are centre-cropped to 4:3
                   (720 x 540) and resized to 640 x 480 (depth: nearest pixel); intrinsics follow.
Output <out>/: image_left/%06d_left.png (640 x 480), depth_left/%06d_left_depth.npy (float32 metres, no depth = 1e4),
intrinsics.txt (fx fy cx cy), pose_left.txt (identity rows, the frame pages use no pose), source.txt.
Frames f-past .. f are written (dppmr's track bonus looks at the 3 previous frames).
--score N: every N-th frame (with >= 3 predecessors) -> depth spread (90th / 10th percentile of valid depth), share of
valid pixels on the dominant plane (RANSAC, 5 % of depth), valid share; sorted by spread x (plane share in 0.2..0.8).
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

W, H, NO_DEPTH = 640, 480, 1e4


class Seq:
    def __init__(self, src):
        kind, path = src.split(':', 1)
        self.kind = kind
        if kind == 'kp':
            from kpbench.data import flight_dir
            self.dir = flight_dir(path)
            assert self.dir, 'no flight %s' % path
            ip = os.path.join(self.dir, 'intrinsics.txt')
            self.intr = np.loadtxt(ip) if os.path.isfile(ip) else np.array([320.0, 320.0, 320.0, 240.0])
            self.n = len(glob.glob(os.path.join(self.dir, 'image_left', '*.png')))
        else:
            self.dir = path
            if os.path.isdir(os.path.join(path, 'rgb')):
                self.rgb = sorted(glob.glob(os.path.join(path, 'rgb', '*')))
                self.dep = sorted(glob.glob(os.path.join(path, 'depth', '*.npy')))
                self.cam = sorted(glob.glob(os.path.join(path, 'cam', '*.npz')))
            else:
                self.rgb = sorted(glob.glob(os.path.join(path, '*_rgb.png')))
                self.dep = [p.replace('_rgb.png', '_depth.npy') for p in self.rgb]
                self.cam = [p.replace('_rgb.png', '_cam.npz') for p in self.rgb]
            self.n = len(self.rgb)

    def frame(self, f):
        """-> rgb (480, 640, 3) uint8, depth (480, 640) float32 (no depth = 1e4), intrinsics (fx, fy, cx, cy)"""
        if self.kind == 'kp':
            img = np.array(Image.open(os.path.join(self.dir, 'image_left', '%06d_left.png' % f)).convert('RGB'))
            z = np.load(os.path.join(self.dir, 'depth_left', '%06d_left_depth.npy' % f)).astype(np.float32)
            K = self.intr
        else:
            img = np.array(Image.open(self.rgb[f]).convert('RGB'))
            z = np.load(self.dep[f]).astype(np.float32)
            c = np.load(self.cam[f]); M = c['intrinsics'] if 'intrinsics' in c else c['camera_intrinsics']
            K = np.array([M[0, 0], M[1, 1], M[0, 2], M[1, 2]], float)
            h, w = z.shape
            if (w, h) != (W, H):
                cw = int(round(h * W / H)); x0 = (w - cw) // 2; s = W / cw
                img = np.array(Image.fromarray(img[:, x0:x0 + cw]).resize((W, H), Image.BICUBIC))
                yy = np.clip(((np.arange(H) + 0.5) / s - 0.5).round().astype(int), 0, h - 1)
                xx = np.clip(((np.arange(W) + 0.5) / s - 0.5).round().astype(int), 0, cw - 1) + x0
                z = z[yy][:, xx]
                K = np.array([K[0] * s, K[1] * s, (K[2] - x0 + 0.5) * s - 0.5, (K[3] + 0.5) * s - 0.5])
        z = np.where(np.isfinite(z) & (z > 0) & (z < NO_DEPTH), z, NO_DEPTH).astype(np.float32)
        return img, z, K


def score(seq, f, rng):
    _, z, K = seq.frame(f)
    vv, uu = np.mgrid[0:H:4, 0:W:4]; zz = z[vv, uu]; ok = zz < 1000
    if ok.mean() < 0.5:
        return None
    zz, uu, vv = zz[ok], uu[ok], vv[ok]
    P = np.stack([(uu - K[2]) * zz / K[0], (vv - K[3]) * zz / K[1], zz], 1)
    best = 0
    for _ in range(200):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        nrm = np.cross(b - a, c - a); nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        best = max(best, int((np.abs((P - a) @ nrm) / nn < 0.05 * zz).sum()))
    p10, p90 = np.percentile(zz, [10, 90])
    return p90 / max(p10, 1e-3), best / len(P), ok.mean(), float(np.median(zz))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--frame', type=int); ap.add_argument('--past', type=int, default=3)
    ap.add_argument('--out'); ap.add_argument('--score', type=int, default=0)
    a = ap.parse_args(); seq = Seq(a.src)
    if a.score:
        rng = np.random.default_rng(0); rows = []
        for f in range(3, seq.n, a.score):
            r = score(seq, f, rng)
            if r:
                rows.append((f,) + r)
        rows.sort(key=lambda r: -r[1] * (0.2 <= r[2] <= 0.8))
        for r in rows[:8]:
            print('frame %4d  spread %5.1f  plane %.2f  valid %.2f  median %.2f m' % r)
        sys.exit()
    os.makedirs(os.path.join(a.out, 'image_left'), exist_ok=True); os.makedirs(os.path.join(a.out, 'depth_left'), exist_ok=True)
    for f in range(a.frame - a.past, a.frame + 1):
        img, z, K = seq.frame(f)
        Image.fromarray(img).save(os.path.join(a.out, 'image_left', '%06d_left.png' % f))
        np.save(os.path.join(a.out, 'depth_left', '%06d_left_depth.npy' % f), z)
    np.savetxt(os.path.join(a.out, 'intrinsics.txt'), K[None], fmt='%.6f')
    np.savetxt(os.path.join(a.out, 'pose_left.txt'), np.tile([0, 0, 0, 0, 0, 0, 1.0], (a.frame + 1, 1)), fmt='%g')
    open(os.path.join(a.out, 'source.txt'), 'w').write('%s frame %d (frames %d..%d exported)\n' % (a.src, a.frame, a.frame - a.past, a.frame))
    print(a.out, 'intrinsics', K)
