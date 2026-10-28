"""Pick-level determinism probe on the cached MICRO candidates (no images, poses or GPU needed).

  python tools/determinism_probe.py dpp1a2 dpp0.601a0.283 --K 200 [--det raco] [--out results/audit/determinism_probe_K200.json]

Each picker runs over every MICRO frame in fresh processes: run A and B with the default BLAS threads, run C with
OMP/MKL/OPENBLAS_NUM_THREADS=1 (Euler nodes differ in core count, so float sums can round differently). Per picker:
frames whose pick sets differ A vs B (same setup) and A vs C (thread count), and the mean picks changed on those frames.
Same picks -> same PnP up to LightGlue / RANSAC, so 0 differing frames puts any fail-count drift (e.g. dpp1a2 165 vs 167
of 936 across studies, SUB2 28 Oct) on the pose side or on commit drift, not on the picker.
"""
import argparse, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ROOT = os.path.join(os.path.dirname(__file__), '..')
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')


def dump(names, K, det, out):
    """picks of every picker on every MICRO frame -> npz (key '<picker>|<flight>|<frame>' = sorted indices)"""
    from kpbench import pickers
    micro = json.load(open(os.path.join(ROOT, 'splits.json')))['micro']; res = {}
    for p in names:
        fn = pickers.parse(p)
        for fid, fr in micro.items():
            c = np.load(os.path.join(ROOT, 'cache', '%s_%s.npz' % (fid.replace('/', '_'), det)))
            for f in fr:
                n = int(c['n'][f]); u, v, s, z = (c[k][f, :n].astype(float) for k in 'uvsz')
                res['%s|%s|%d' % (p, fid, f)] = np.sort(np.asarray(fn(u, v, s, z, K)))
    np.savez(out, **res)


def run(names, K, det, out, one_thread):
    env = dict(os.environ, **({k: '1' for k in THREADS} if one_thread else {}))
    if not one_thread:
        for k in THREADS: env.pop(k, None)
    subprocess.run([sys.executable, __file__, *names, '--K', str(K), '--det', det, '--dump', out], env=env, check=True)
    return dict(np.load(out))


def diff(a, b, p):
    ks = [k for k in a if k.startswith(p + '|')]; d = [k for k in ks if not np.array_equal(a[k], b[k])]
    ch = [len(np.setdiff1d(a[k], b[k])) for k in d]
    return dict(frames=len(ks), differ=len(d), mean_changed=round(float(np.mean(ch)), 1) if ch else 0, first=d[:3])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pickers', nargs='+'); ap.add_argument('--det', default='raco')
    ap.add_argument('--K', type=int, default=200); ap.add_argument('--out', default=None); ap.add_argument('--dump', default=None)
    a = ap.parse_args()
    if a.dump:
        return dump(a.pickers, a.K, a.det, a.dump)
    tmp = os.path.join(ROOT, 'results', 'audit', '_probe_%s.npz')
    A, B, C = (run(a.pickers, a.K, a.det, tmp % x, x == 'C') for x in 'ABC')
    out = dict(det=a.det, K=a.K, runs='A, B default BLAS threads; C 1 thread; each a fresh process',
               pickers={p: dict(same_setup=diff(A, B, p), one_thread=diff(A, C, p)) for p in a.pickers})
    for x in 'ABC':
        os.remove(tmp % x)
    fn = os.path.join(ROOT, a.out or 'results/audit/determinism_probe_K%d.json' % a.K)
    json.dump(out, open(fn, 'w'), indent=1); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
