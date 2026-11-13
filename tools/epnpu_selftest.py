"""SUB1 E-d self-test (engineer 2026-11-13, dispatch 13:15): does epnpu_t (tools/micro_epnpu_dq3.py) order pick sets as the true pose error does?

  python tools/epnpu_selftest.py [--scenes 10] [--trials 200]

Synthetic near-planar oblique scene, no flight data: camera centre at 0, looking along +z, pitched ground plane y = 3 m (in camera
coordinates y points down) seen from 5 to 30 m, relief +-0.05 m. 400 candidate points; three pick sets of 40 at the true pose:
  uniform  = 40 of the candidates spread over the image (farthest-point in pixels),
  cluster  = the 40 candidates nearest a random pixel (one 2D clump),
  near     = 40 random candidates from the nearest depth tercile (the sd-like near-field concentration seen on MICRO, 0.479 share).
True error: 1 px image noise + depth noise along the ray (sigma 0.005 z^2, the same model epnpu_t assumes), kpbench.bench.pnp
(AP3P RANSAC + LM), RMS camera-centre error over trials. Pre-registered pass rule (written before the first run, results/sub1/epnpu_selftest.md):
  for both contrasts (cluster vs uniform, near vs uniform) epnpu_t is larger for the worse set in >= 9 of 10 scenes where the
  true RMS error is larger for that set, and log10 RMS error vs epnpu_t over all scene-sets has Spearman rho >= 0.7.
Fail -> E-d dropped unrun, SUB1 stays KILL.
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kpbench.data import FX, FY, CX, CY, W, H
from kpbench.bench import pnp
from micro_epnpu_dq3 import epnpu_t


def scene(rng, n=400):
    """camera-frame points on the plane y = 3 (z 5..30 m) with relief, kept inside the image -> (X world = camera, u, v)"""
    P = []
    while len(P) < n:
        z = rng.uniform(5, 30); x = rng.uniform(-1, 1) * z * CX / FX; y = 3.0 + rng.normal(0, 0.05)
        u, v = FX * x / z + CX, FY * y / z + CY
        if 0 <= u < W and 0 <= v < H:
            P.append((x, y, z, u, v))
    P = np.array(P); return P[:, :3], P[:, 3:]


def picks(rng, X, uv, k=40):
    """-> dict name -> indices"""
    s = [int(rng.integers(len(X)))]; d = np.linalg.norm(uv - uv[s[0]], axis=1)
    while len(s) < k:
        j = int(np.argmax(d)); s.append(j); d = np.minimum(d, np.linalg.norm(uv - uv[j], axis=1))
    c = uv[int(rng.integers(len(X)))]; cl = np.argsort(np.linalg.norm(uv - c, axis=1))[:k]
    near = np.flatnonzero(X[:, 2] <= np.quantile(X[:, 2], 1 / 3)); nr = rng.choice(near, k, replace=False)
    return dict(uniform=np.array(s), cluster=cl, near=nr)


def true_rms(rng, X, uv, trials, sig_d=0.005):
    """RMS camera-centre error [m] over noisy trials (None-pose trials count as 10 m)"""
    R, t = np.eye(3), np.zeros(3); e = []
    for _ in range(trials):
        z = X[:, 2]; Xn = X * ((z + rng.normal(0, sig_d * z ** 2)) / z)[:, None]         # depth noise along the ray
        Xn = Xn + rng.normal(0, 1, X.shape) * (z / FX)[:, None] * np.array([1, 1, 0])      # 1 px keyframe noise back-projected
        img = uv + rng.normal(0, 1, uv.shape)
        err = pnp(Xn, img, R, t)[0]
        e.append(10.0 if err is None else err / 100.0)
    return float(np.sqrt(np.mean(np.square(e))))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scenes', type=int, default=10); ap.add_argument('--trials', type=int, default=200)
    a = ap.parse_args(); rng = np.random.default_rng(0); rows = []
    for sc in range(a.scenes):
        X, uv = scene(rng); P = picks(rng, X, uv)
        r = {k: (epnpu_t(X[i], uv[i], np.eye(3), np.zeros(3)), true_rms(rng, X[i], uv[i], a.trials)) for k, i in P.items()}
        rows.append(r); print(sc, ' '.join('%s ep %.3f rms %.3f m' % (k, v[0], v[1]) for k, v in r.items()), flush=True)
    L = ['', '## Result (%d scenes x %d trials, seed 0)' % (a.scenes, a.trials), '', '| scene | ' + ' | '.join('%s epnpu_t / log10 RMS' % k for k in rows[0]) + ' |',
         '|---|' + '---|' * len(rows[0])]
    for sc, r in enumerate(rows):
        L.append('| %d | ' % sc + ' | '.join('%.3f / %.3f' % (v[0], np.log10(v[1])) for v in r.values()) + ' |')
    ok = True
    for bad in ('cluster', 'near'):
        agree = sum((r[bad][0] > r['uniform'][0]) == (r[bad][1] > r['uniform'][1]) for r in rows)
        truth = sum(r[bad][1] > r['uniform'][1] for r in rows); ep = sum(r[bad][0] > r['uniform'][0] for r in rows)
        L.append('- %s vs uniform: true error larger in %d / %d scenes, epnpu_t larger in %d / %d, same direction in %d / %d' % (bad, truth, len(rows), ep, len(rows), agree, len(rows)))
        ok &= agree >= 9 * len(rows) // 10
    E = [v[0] for r in rows for v in r.values()]; T = [np.log10(v[1]) for r in rows for v in r.values()]
    rho = spearman(np.array(E), np.array(T)); ok &= rho >= 0.7
    L += ['- Spearman rho(epnpu_t, log10 RMS) over %d scene-sets: %.3f' % (len(E), rho), '',
          '**Reading: %s**' % ('PASS: epnpu_t orders pick sets as the true pose error does -> E-d may be queued on Euler MICRO (euler/sub1_epnpu.sbatch).' if ok
                               else 'FAIL: kill met -> E-d dropped unrun, SUB1 stays KILL.'), '', 'Tool: tools/epnpu_selftest.py.']
    open(os.path.join(ROOT, 'results', 'sub1', 'epnpu_selftest.md'), 'a', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
