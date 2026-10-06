"""TartanAir flights: where they live, how to read images, depth and poses.

A flight id is "<env>/<traj>", e.g. "oldtown/P001". The data root is the environment variable
KPBENCH_DATA (default: the laptop's local_data folder). Each environment is looked up under a few
known folder layouts (laptop and Euler differ), see CANDIDATES.
"""
import os
import numpy as np
from PIL import Image

W, H = 640, 480
FX = FY = 320.0
CX, CY = 320.0, 240.0
KMAT = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
# TartanAir poses are body (NED) frame; this maps camera axes (x right, y down, z forward) into the body frame.
R_BC = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
KF = 10          # a keyframe every 10 frames
WIN = 20         # a query uses the keyframes within +-20 frames
DEPTH_MAX = 1000.0   # depth >= this = no depth (sky)
MIN_FRAMES = 50

DEFAULT_ROOT = r'C:\Users\Kinga\Desktop\ETH_lab\local_data'
ROOT = os.environ.get('KPBENCH_DATA', DEFAULT_ROOT)

ENVS = {
    'oldtown': ['P%03d' % i for i in range(0, 9)],
    'hospital': ['P%03d' % i for i in range(37, 50)],
    'factory': ['P%03d' % i for i in range(0, 9)],
}
CANDIDATES = {
    'oldtown': ['tartanair_hard/{t}', 'tartanair_hard/oldtown/Hard/{t}'],
    'hospital': ['tartanair_envs/hospital/{t}', 'tartanair_hospital/{t}'],
    'factory': ['tartanair_envs/abandonedfactory/{t}', 'tartanair_factory/{t}'],
}
FLIGHTS = ['%s/%s' % (e, t) for e, ts in ENVS.items() for t in ts]


def slug(fid):
    return fid.replace('/', '_')


def flight_dir(fid):
    env, traj = fid.split('/')
    for pat in CANDIDATES[env]:
        d = os.path.join(ROOT, pat.format(t=traj))
        if os.path.isfile(os.path.join(d, 'pose_left.txt')):
            return d
    return None


def quat_to_R(qx, qy, qz, qw):
    n = qx*qx + qy*qy + qz*qz + qw*qw
    s = 2.0 / n
    wx, wy, wz = s*qw*qx, s*qw*qy, s*qw*qz
    xx, xy, xz = s*qx*qx, s*qx*qy, s*qx*qz
    yy, yz, zz = s*qy*qy, s*qy*qz, s*qz*qz
    return np.array([[1-(yy+zz), xy-wz, xz+wy], [xy+wz, 1-(xx+zz), yz-wx], [xz-wy, yz+wx, 1-(xx+yy)]])


def to3d(u, v, z):
    """pixel + depth -> camera-frame 3D points (n, 3)"""
    return np.stack([(u - CX) * z / FX, (v - CY) * z / FY, z], 1)


class Flight:
    def __init__(self, fid):
        self.id = fid
        self.env, self.traj = fid.split('/')
        self.slug = slug(fid)
        self.dir = flight_dir(fid)
        if self.dir is None:
            raise FileNotFoundError('flight %s not found under %s' % (fid, ROOT))
        self.poses = np.loadtxt(os.path.join(self.dir, 'pose_left.txt'))
        n_img = len([f for f in os.listdir(os.path.join(self.dir, 'image_left')) if f.endswith('.png')])
        n_dep = len([f for f in os.listdir(os.path.join(self.dir, 'depth_left')) if f.endswith('.npy')])
        self.N = min(len(self.poses), n_img, n_dep)
        self.keyframes = list(range(0, self.N, KF))
        self.queries = [q for q in range(self.N) if q % KF]

    def img_path(self, f):
        return os.path.join(self.dir, 'image_left', '%06d_left.png' % f)

    def img(self, f):
        return np.array(Image.open(self.img_path(f)).convert('RGB'))

    def depth(self, f):
        return np.load(os.path.join(self.dir, 'depth_left', '%06d_left_depth.npy' % f)).astype(np.float64)

    def cam_pose(self, f):
        """camera-to-world rotation R and camera centre t: X_world = R @ X_cam + t"""
        return quat_to_R(*self.poses[f, 3:7]) @ R_BC, self.poses[f, :3]

    def near(self, q):
        return [k for k in self.keyframes if abs(k - q) <= WIN]


def available():
    """flights present under ROOT with enough frames"""
    out = []
    for fid in FLIGHTS:
        d = flight_dir(fid)
        if d is None:
            continue
        try:
            if Flight(fid).N >= MIN_FRAMES:
                out.append(fid)
        except Exception:
            pass
    return out


def parse_flights(spec):
    if spec in ('all', ''):
        return available()
    out = []
    for s in spec.split(','):
        s = s.strip()
        if s in ENVS:
            out += ['%s/%s' % (s, t) for t in ENVS[s]]
        elif s:
            out.append(s)
    return out
