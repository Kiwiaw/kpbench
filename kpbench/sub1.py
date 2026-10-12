"""SUB1 analysis: the pre-registered selection rule of docs/sub1_metrics.md applied to saved evaluate rows (--metrics).

  python -m kpbench.sub1 --split dev        -> results/sub1/<split>.md + .json
Inputs: results/eval/<split>/raco_<picker>_K100_lg_m.json.gz for sd, cube0.5 and grid (euler/sub1.sbatch writes them).
Per metric: oriented so positive = uniform better, no-pose values filled as pre-registered, d = cube0.5 - sd per query,
z = mean(d) / block-bootstrap SE (20-frame blocks, 2000 resamples), AUROC vs the fail flag on pooled sd + cube0.5 rows.
Boring checks (interpretation fixed here, before DEV data): A = mean of d after regressing it on the n_inl difference
(intercept z), B = the same on the inlier depth-IQR difference; grid share = mean d(grid - sd) / mean d(cube - sd).
"""
import os, json, gzip
import numpy as np
from .cache import REPO
from .stats import fail_flags
from .metrics import MCOLS

PICKERS = ('sd', 'cube0.5', 'grid')
CAND = ['crlb', 'n_inl', 'inl_ratio', 'hull', 'reproj_med', 'log_t', 'log_r', 'thr_score']
SIGN = dict(crlb=-1, n_inl=1, inl_ratio=1, hull=1, reproj_med=-1, log_t=-1, log_r=-1, thr_score=1)
NOPOSE = dict(n_inl=0.0, inl_ratio=0.0, hull=0.0, reproj_med=3.0, log_t=2.0, log_r=float(np.log10(20)), thr_score=0.0)
FAIL_DEF = ('log_t', 'log_r', 'thr_score')
OUT = os.path.join(REPO, 'results', 'sub1')


def table(split):
    R, M, Fl = {}, {}, {}
    for p in PICKERS:
        d = json.load(gzip.open(os.path.join(REPO, 'results', 'eval', split, 'raco_%s_K100_lg_m.json.gz' % p), 'rt'))
        c = d['cols']; key = [(str(r[0]), r[1]) for r in d['rows']]; o = sorted(range(len(key)), key=key.__getitem__)
        R[p] = [d['rows'][i] for i in o]; Fl[p] = np.array(d['flat'], bool)[o]
        M[p] = {m: np.array([np.nan if R[p][i][c.index(m)] is None else R[p][i][c.index(m)] for i in range(len(o))], float)
                for m in MCOLS + ['n_inl']}
    keys = [[(str(r[0]), r[1]) for r in R[p]] for p in PICKERS]
    assert keys[0] == keys[1] == keys[2], 'pickers must share the query set'
    return R, M, Fl


def blocks(rows, b=20):
    _, inv = np.unique(['%s:%d' % (r[0], r[1] // b) for r in rows], return_inverse=True)
    return inv


def boot_se(d, inv, n=2000, seed=0):
    m = inv.max() + 1; s = np.bincount(inv, d, m); c = np.bincount(inv, None, m)
    pick = np.random.default_rng(seed).integers(0, m, (n, m))
    return float(np.std(s[pick].sum(1) / c[pick].sum(1)))


def auroc(score, fail):
    """P(score of a success > score of a failure), ties count half"""
    from scipy.stats import rankdata
    r = rankdata(score); n1 = (fail == 0).sum(); n0 = (fail == 1).sum()
    return float((r[fail == 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def resid_z(d, x, inv):
    """intercept of d = a + b x (x = a paired difference, NaN -> 0): the part of d left when x does not differ; z with b fixed"""
    x = np.nan_to_num(x); A = np.stack([np.ones_like(x), x], 1)
    a, b = np.linalg.lstsq(A, d, rcond=None)[0]; r = d - b * x
    return float(r.mean() / boot_se(r, inv)), float(r.mean())


def analyse(split):
    R, M, Fl = table(split); inv = blocks(R['sd'])
    fail = {p: fail_flags(R[p]) for p in PICKERS}
    pooled = np.concatenate([M['sd']['crlb'], M['cube0.5']['crlb']])
    cap = float(np.nanpercentile(pooled, 99))
    def val(p, m):
        v = M[p][m].copy(); v[np.isnan(v)] = cap if m == 'crlb' else NOPOSE[m]      # NaN = no pose (n_inl is 0 there)
        return SIGN[m] * v
    both_ok = (fail['sd'] == 0) & (fail['cube0.5'] == 0)
    dn = M['cube0.5']['n_inl'] - M['sd']['n_inl']; ddep = M['cube0.5']['depth_iqr'] - M['sd']['depth_iqr']
    res = []
    for m in CAND:
        a, b, g = val('cube0.5', m), val('sd', m), val('grid', m); d = a - b
        z = d.mean() / boot_se(d, inv)
        au = auroc(np.concatenate([b, a]), np.concatenate([fail['sd'], fail['cube0.5']]))
        zok = None
        if m in FAIL_DEF:
            ib = blocks([r for r, k in zip(R['sd'], both_ok) if k]); zok = float(d[both_ok].mean() / boot_se(d[both_ok], ib))
        eligible = (zok is not None and zok >= 2) if m in FAIL_DEF else (au is not None and au >= 0.75)
        dg = g - b
        zA = resid_z(d, dn, inv)[0] if m != 'n_inl' else None
        zB, mB = resid_z(d, ddep, inv)
        res.append(dict(metric=m, mean_d=round(float(d.mean()), 4), z=round(float(z), 2), auroc=None if au is None else round(au, 3),
                        z_both_ok=None if zok is None else round(zok, 2), eligible=bool(eligible),
                        z_grid=round(float(dg.mean() / boot_se(dg, inv)), 2),
                        grid_share=round(float(dg.mean() / d.mean()), 2) if d.mean() else None,
                        mean_d_flat=round(float(d[Fl['sd']].mean()), 4) if Fl['sd'].any() else None,
                        zA_resid_ninl=None if zA is None else round(zA, 2), zB_resid_depth=round(zB, 2),
                        B_kept=round(mB / d.mean(), 2) if d.mean() else None))
    ok = [r for r in res if r['eligible'] and r['z'] >= 2.5]
    pick = max(ok, key=lambda r: r['z'])['metric'] if ok else None
    return dict(split=split, n=len(R['sd']), n_flat=int(Fl['sd'].sum()), crlb_cap=round(cap, 4),
                fail_pct={p: round(100 * fail[p].mean(), 3) for p in PICKERS}, metrics=res, chosen=pick,
                verdict=('chosen: %s' % pick) if pick else 'KILL: no eligible metric with z >= 2.5')


def write(out):
    os.makedirs(OUT, exist_ok=True); base = os.path.join(OUT, out['split'])
    json.dump(out, open(base + '.json', 'w'), indent=1)
    L = ['# SUB1 metrics on %s (%d queries, %d flat; rule: docs/sub1_metrics.md)' % (out['split'], out['n'], out['n_flat']), '',
         'fail %: ' + ', '.join('%s %.2f' % kv for kv in out['fail_pct'].items()) + '; crlb no-pose cap %.3f' % out['crlb_cap'], '',
         '| metric | mean d (cube-sd, oriented) | z | AUROC | z both ok | eligible | z grid-sd | grid share | mean d flat | z after n_inl | z after depth IQR | share kept after depth |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in out['metrics']:
        L.append('| ' + ' | '.join(str(r[k]) for k in ['metric', 'mean_d', 'z', 'auroc', 'z_both_ok', 'eligible', 'z_grid', 'grid_share',
                                                       'mean_d_flat', 'zA_resid_ninl', 'zB_resid_depth', 'B_kept']) + ' |')
    L += ['', '**%s**' % out['verdict']]
    open(base + '.md', 'w').write('\n'.join(L) + '\n')
    return base + '.md'


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='dev'); a = ap.parse_args(argv)
    out = analyse(a.split); print(write(out)); print(out['verdict'])


if __name__ == '__main__':
    main()
