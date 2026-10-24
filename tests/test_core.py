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


def test_nms0_is_sd_and_nms_spreads():
    u, v, s, z = toy_frame(2000)
    assert pickers.parse('nms0')(u, v, s, z, 100).tolist() == pickers.parse('sd')(u, v, s, z, 100).tolist()
    i = pickers.parse('nms16')(u, v, s, z, 100)
    assert len(set(i.tolist())) == 100 and (z[i] < 1000).all()
    d = (u[i, None] - u[None, i]) ** 2 + (v[i, None] - v[None, i]) ** 2; d[np.diag_indices(100)] = np.inf
    assert d.min() >= 16 ** 2


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


def test_dpp_depth_adaptive():
    u, v, s, z = toy_frame(600, seed=3)
    ok = z < 1000; med = float(np.median(z[ok]))
    # dppm at c = sigma / median depth is dpp<sigma> on that frame
    assert pickers.parse('dppm%ra1.3' % (0.5 / med))(u, v, s, z, 80).tolist() == pickers.parse('dpp0.5a1.3')(u, v, s, z, 80).tolist()
    # dppl with one depth everywhere is dpp<c * z>
    zc = np.where(ok, 8.0, z)
    assert pickers.parse('dppl0.05')(u, v, s, zc, 80).tolist() == pickers.parse('dpp0.4')(u, v, s, zc, 80).tolist()
    # both are scale-free: all depths x 3 -> same picks (dpp0.5 is not)
    z3 = np.where(ok, 3 * z, z)
    for name in ('dppm0.1', 'dppl0.05a0.7'):
        a = pickers.parse(name)(u, v, s, z, 80); b = pickers.parse(name)(u, v, s, z3, 80)
        assert a.tolist() == b.tolist(), name
        assert len(set(a.tolist())) == 80 and (z[a] < 1000).all(), name
    # degenerate frames as dpp: no depth -> strongest K; fewer with depth than K -> all of them, then strongest without
    zn = np.full_like(z, 1e9)
    for name in ('dppm0.1', 'dppl0.05', 'dpp0.5'):
        assert pickers.parse(name)(u, v, s, zn, 30).tolist() == list(range(30)), name
        zf = zn.copy(); zf[[5, 50, 500]] = [2.0, 4.0, 6.0]
        idx = pickers.parse(name)(u, v, s, zf, 10)
        assert sorted(idx[:3].tolist()) == [5, 50, 500] and idx[3:].tolist() == [0, 1, 2, 3, 4, 6, 7], name
        assert len(pickers.parse(name)(u[:20], v[:20], s[:20], z[:20], 50)) == 20, name
    # the Gibbs kernel of dppl is positive semi-definite for mixed depths
    P = data.to3d(u[ok][:200], v[ok][:200], z[ok][:200]); l = 0.05 * z[ok][:200]
    s2 = l[:, None] ** 2 + l[None] ** 2
    Kg = (2 * l[:, None] * l[None] / s2) ** 1.5 * np.exp(-((P[:, None] - P[None]) ** 2).sum(2) / s2)
    assert np.linalg.eigvalsh(Kg).min() > -1e-9


def test_dpps_slices():
    # T12 (docs/t12_dpps.md): one slice = dpp0.5 exactly; n slices cap each slice at ceil(K/n) while candidates remain
    for seed in (0, 3, 5):
        u, v, s, z = toy_frame(600, seed=seed)
        for K in (30, 100):
            ref = pickers.parse('dpp0.5')(u, v, s, z, K).tolist()
            for name in ('dpps1', 'dpps1q', 'dpps1l', 'dpps1e'):
                assert pickers.parse(name)(u, v, s, z, K).tolist() == ref, (name, seed, K)
    u, v, s, z = toy_frame(600, seed=3); ok = np.flatnonzero(z < 1000); zp = z[ok][:400]
    edges = np.quantile(zp, [0.25, 0.5, 0.75])
    for name in ('dpps4q', 'dpps4l', 'dpps4e', 'dpps8q'):
        idx = pickers.parse(name)(u, v, s, z, 100)
        assert len(set(idx.tolist())) == 100 and (z[idx] < 1000).all(), name
    idx = pickers.parse('dpps4q')(u, v, s, z, 100)
    assert np.bincount(np.searchsorted(edges, z[idx], side='right'), minlength=4).max() <= 25
    zn = np.full_like(z, 1e9)
    assert pickers.parse('dpps4q')(u, v, s, zn, 30).tolist() == list(range(30))


def test_dppmr_combines_dppm_and_dppr():
    """dppmr<c>a<alpha>b<beta>t<tau> = dppm's depth-scaled sigma + dppr's past-frame track bonus."""
    u, v, s, z = toy_frame(600, seed=4)
    ok = z < 1000; med = float(np.median(z[ok]))
    rng = np.random.default_rng(9); de = rng.normal(size=(600, 32)).astype(np.float32)

    class F_:
        slug = 'toy'

    class C_:      # past frames: half of the candidates re-appear with small descriptor noise, in shuffled order
        def frame(self, g):
            r = np.random.default_rng(100 + g); d = de.copy(); d[::2] += 0.05 * r.normal(size=d[::2].shape)
            d[1::2] = r.normal(size=d[1::2].shape); return None, None, None, None, d[r.permutation(600)]
    ctx = (F_(), C_(), 5)
    pr = pickers.parse('dppr0.5a0.45b1.5t0.9'); pr(u, v, s, z, 80, ctx=ctx, de=de)
    assert max(pr.rstd) > 0                          # the track bonus is active on this toy frame
    # beta 0 is dppm exactly
    for a in ('1', '0.5'):
        assert pickers.parse('dppmr0.03a%sb0t0.9' % a)(u, v, s, z, 80, ctx=ctx, de=de).tolist() ==             pickers.parse('dppm0.03a%s' % a)(u, v, s, z, 80).tolist(), a
    # c x median depth = sigma is dppr exactly (same track bonus)
    for b, t in (('1.02', '0.9'), ('1.5', '0.8')):
        m = pickers.parse('dppmr%ra0.45b%st%s' % (0.5 / med, b, t))(u, v, s, z, 80, ctx=ctx, de=de)
        r = pickers.parse('dppr0.5a0.45b%st%s' % (b, t))(u, v, s, z, 80, ctx=ctx, de=de)
        assert m.tolist() == r.tolist(), (b, t)
    # the bonus changes the picks, and dppmr is scale-free in depth like dppm
    pk = pickers.parse('dppmr0.03a0.45b1.5t0.9')
    a = pk(u, v, s, z, 80, ctx=ctx, de=de)
    assert a.tolist() != pickers.parse('dppm0.03a0.45')(u, v, s, z, 80).tolist()
    assert np.isclose(pk.sig[-1], 0.03 * med) and pk.needs_past and pk.needs_desc and not getattr(pk, 'needs_ctx', False)
    z3 = np.where(ok, 3 * z, z)
    assert pk(u, v, s, z3, 80, ctx=ctx, de=de).tolist() == a.tolist()
    assert len(set(a.tolist())) == 80 and (z[a] < 1000).all()


def test_fresh_pose_roundtrip():
    from kpbench import fresh
    rng = np.random.default_rng(5)
    for _ in range(50):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        R = q * np.sign(np.linalg.det(q))
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = rng.normal(size=3) * 10
        row = fresh.pose_row(T)
        assert np.allclose(data.quat_to_R(*row[3:7]) @ data.R_BC, R, atol=1e-9) and np.allclose(row[:3], T[:3, 3])
    assert 'oldtown_easy/P000' not in data.FLIGHTS and data.CANDIDATES['factory_hard'] == ['kpbench_fresh/factory_hard/{t}']
