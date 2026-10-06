"""Unit tests that need no data: pickers, plane fraction, PnP on a synthetic scene, round trip of the pose convention."""
import numpy as np, cv2
from kpbench import pickers, labels, data, bench


def toy_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 640, n); v = rng.uniform(0, 480, n); s = -np.arange(n, dtype=float)
    z = rng.uniform(2, 30, n); z[::7] = 1e9       # some sky
    return u, v, s, z


def test_pickers_sizes_and_depth():
    u, v, s, z = toy_frame()
    for name in ['score', 'sd', 'cube0.5', 'cube2', 'grid', 'grid10x8', 'fps', 'fps0']:
        idx = pickers.parse(name)(u, v, s, z, 50)
        assert len(idx) == 50, name
        assert len(set(idx.tolist())) == 50, name
        if name != 'score':
            assert (z[idx] < 1000).all(), name


def test_sd_is_strongest_with_depth():
    u, v, s, z = toy_frame()
    idx = pickers.parse('sd')(u, v, s, z, 20)
    assert idx.tolist() == np.flatnonzero(z < 1000)[:20].tolist()


def test_grid_spreads():
    u, v, s, z = toy_frame(2000)
    g = pickers.parse('grid')(u, v, s, z, 100); d = pickers.parse('sd')(u, v, s, z, 100)
    cells = lambda i: len(set(zip((u[i] * 10 // 640).astype(int), (v[i] * 10 // 480).astype(int))))
    assert cells(g) > cells(d)


def test_fps_first_is_strongest_and_far_apart():
    P = np.array([[0, 0], [1, 1], [100, 100], [50, 50.]])
    assert pickers.fps_2d(P, 3, False).tolist() == [0, 2, 3]


def test_plane_frac():
    rng = np.random.default_rng(1)
    P = np.c_[rng.uniform(-5, 5, (300, 2)), np.zeros(300)]      # flat
    assert labels.plane_frac(P) > 0.99
    P2 = rng.uniform(-5, 5, (300, 3))
    assert labels.plane_frac(P2) < 0.5


def test_pnp_recovers_pose():
    rng = np.random.default_rng(2)
    u = rng.uniform(50, 590, 80); v = rng.uniform(50, 430, 80); z = rng.uniform(3, 20, 80)
    R = data.quat_to_R(0.1, 0.2, -0.1, 0.96); t = np.array([1.0, -2.0, 0.5])
    Xw = data.to3d(u, v, z) @ R.T + t
    err, rot, ninl = bench.pnp(Xw, np.stack([u, v], 1), R, t)
    assert err < 0.5 and rot < 0.05 and ninl >= 70


def test_quat_is_rotation():
    R = data.quat_to_R(0.3, -0.2, 0.1, 0.9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)
