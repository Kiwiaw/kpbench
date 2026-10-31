"""R19 smoke on synthetic scenes (the laptop has no flight poses): AP3P vs AP3P + planar hypothesis, bench.pnp, no flight data.

  python tools/solver_smoke.py [--out results/main/r19_solver_smoke.md]

Scenes: 100 points, depth 2-6 m ahead of a tilted plane (or spread in 3D), 1 px noise, outliers as a share of the matches. n matches is
low on purpose (flat frames have about 50). Prints fail % (> 25 cm or > 2 deg or no pose), median cm, and how often ippe differs.
"""
import argparse, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpbench.bench import pnp
from kpbench.data import KMAT, W, H


def scene(rng, n, thick, tilt_deg, out):
    """n matches of a plane tilted by tilt_deg, +- thick m off the plane; true camera at the origin looking down +z, rotated and moved"""
    u, v = rng.uniform(0, W, n), rng.uniform(0, H, n); ray = np.stack([(u - KMAT[0, 2]) / KMAT[0, 0], (v - KMAT[1, 2]) / KMAT[1, 1], np.ones(n)], 1)
    t = np.radians(tilt_deg); nrm = np.array([0, np.sin(t), np.cos(t)]); z = 3.0 / (ray @ nrm)
    Xc = ray * z[:, None] + nrm * rng.uniform(-thick, thick, n)[:, None]
    uv = (Xc @ KMAT.T); uv = uv[:, :2] / uv[:, 2:] + rng.normal(0, 1, (n, 2))
    k = int(out * n); uv[:k] = np.stack([rng.uniform(0, W, k), rng.uniform(0, H, k)], 1)
    R = cv2.Rodrigues(rng.normal(0, 0.3, 3))[0]; c = rng.normal(0, 1, 3)    # world -> camera; c = true centre
    return (Xc @ R) + c, uv, R.T, c    # Rq = camera -> world, as Flight.cam_pose


def run(n, thick, tilt, out, reps=300):
    res = {s: [] for s in ('ap3p', 'ippe')}; same = 0
    for i in range(reps):
        obj, uv, R, c = scene(np.random.default_rng(i), n, thick, tilt, out); a = {}
        for s in res:
            e, r, ni = pnp(obj, uv, R, c, solver=s); a[s] = (e, r, ni); res[s].append((e if e is not None else 1e9, r if r is not None else 1e9))
        same += a['ap3p'] == a['ippe']
    row = lambda v: (100 * np.mean([e > 25 or r > 2 for e, r in v]), np.median([e for e, _ in v]))
    return row(res['ap3p']), row(res['ippe']), same / reps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=None); a = ap.parse_args()
    L = ['| scene | matches | off-plane m | tilt | outliers | ap3p fail % / median cm | ippe fail % / median cm | identical result |', '|---|---|---|---|---|---|---|---|']
    for name, n, th, t, o in (('planar', 12, 0, 30, 0.3), ('planar', 25, 0, 30, 0.3), ('planar', 50, 0, 30, 0.5), ('planar', 25, 0, 60, 0.3), ('near-planar', 25, 0.03, 30, 0.3),
                              ('near-planar', 50, 0.03, 30, 0.5), ('3D', 25, 1.0, 30, 0.3), ('3D', 50, 1.0, 30, 0.5)):
        x, y, s = run(n, th, t, o)
        L.append('| %s | %d | %.2f | %d | %.0f %% | %.1f / %.1f | %.1f / %.1f | %.0f %% |' % (name, n, th, t, 100 * o, *x, *y, 100 * s))
    print('\n'.join(L))
    if a.out:
        open(a.out, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
