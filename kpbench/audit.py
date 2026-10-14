"""SUB2 audit: verify every part of the published kpbench numbers, step by step. Writes reports/audit_<date>.md.

  python -m kpbench.audit [--quick]

Each check prints PASS / FAIL / WARN with the numbers behind it. FAIL = a published number or text is wrong;
WARN = correct but fragile or misleading. Checks:
  C1  counts       queries per flight = labels = rows of every 31-flight run; totals 18571 / flat 1813 / very flat 1058
  C2  K            every row of the 31-flight runs kept exactly K points
  C3  keyframes    keyframes every 10th frame, queries = the rest, +-20 window gives 4 keyframes (fewer only at the ends)
  C4  reproduce    re-running stored configurations on MICRO gives bit-identical rows (seeded RANSAC, deterministic picks)
  C5  ground truth under the TRUE pose, LightGlue inlier matches reproject onto their query pixels (depth, pose, intrinsics
                   conventions consistent with the images); error formulas recomputed independently (scipy rotations)
  C6  fail rule    python (export/stats) and page (app.js isFail) agree on every query; thresholds 25 cm / 2 deg
  C7  page = runs  docs/data/runs/<id>.json rows equal results/runs rows; index.json fail25 equals recomputation
  C8  bootstrap    python block bootstrap and the page's JavaScript bootstrap give the same interval (run in node)
  C9  paired test  the page's per-query sign test vs block-aware tests (neighbouring frames fail together)
  C10 block length paired difference cubes - strongest with 10 / 20 / 40-frame blocks
  C11 headline     the numbers quoted for the 31-flight result (2.4 / 1.7 / 1.8, flat 16.3 / 12.5 / 13.6, -0.67 [-1.02,-0.38]; rerun 11 Oct)
  C12 planarity    plane_frac labels recomputed with a stronger independent plane fit; how many queries change bin
  C13 canaries     random picker worse than strongest; rule-breaking pickers refused
  C14 texts        README / page / data spec describe the protocol the code runs
  C15 incumbent    dpp0.5 (incumbent since 13 Oct): 31-flight row = leaderboard.json = quoted (18571 queries, K 100, all 1.38,
                   flat 10.59, vflat 16.07, rep5 0.5997); held-out gate vs cube0.5 recomputed (877 flat, 6.50 vs 9.24, -2.74, 6/0)
"""
import os, re, json, gzip, glob, subprocess, tempfile, datetime
import numpy as np
from .data import Flight, FLIGHTS, slug, to3d, DEPTH_MAX, KMAT, WIN
from .cache import REPO, Cands
from .stats import fail_flags, block_ids, boot_ci, paired_diff, per_flight, sign_test
from .splits import load as load_splits

RUNS31 = ['raco_sd_K100_same_lg', 'raco_cube0.5_K100_same_lg', 'raco_grid_K100_same_lg',
          'raco_sd_K100_ret5_lg', 'raco_cube0.5_K100_ret5_lg', 'raco_grid_K100_ret5_lg']
RES = []


def report(cid, status, title, detail):
    RES.append((cid, status, title, detail)); print('%-4s %-4s %s | %s' % (cid, status, title, detail.replace('\n', ' ')[:300]), flush=True)


def labels():
    out = {}
    for f in FLIGHTS:
        out[f] = json.load(open(os.path.join(REPO, 'results', 'labels', slug(f) + '.json')))
    return out


def run_rows(rid):
    rows = []
    for f in FLIGHTS:
        fn = os.path.join(REPO, 'results', 'runs', rid, slug(f) + '.json.gz')
        if os.path.isfile(fn):
            for r in json.load(gzip.open(fn, 'rt'))['rows']:
                rows.append([f] + r[1:])
    return rows


def flat_of(rows, lab, lo=0.85):
    return np.array([(lab[r[0]][str(r[1])].get('plane_frac') or -1) >= lo for r in rows])


def c1_counts(lab, R):
    nlab = {f: len(lab[f]) for f in FLIGHTS}
    tot = sum(nlab.values()); flat = sum(1 for f in FLIGHTS for v in lab[f].values() if (v.get('plane_frac') or -1) >= 0.85)
    vflat = sum(1 for f in FLIGHTS for v in lab[f].values() if (v.get('plane_frac') or -1) >= 0.95)
    bad = []
    for rid, rows in R.items():
        per = {}
        for r in rows: per[r[0]] = per.get(r[0], 0) + 1
        bad += ['%s %s %d vs %d' % (rid, f, per.get(f, 0), nlab[f]) for f in FLIGHTS if per.get(f, 0) != nlab[f]]
    nF = {f: Flight(f).queries for f in FLIGHTS if os.path.isdir(os.path.dirname(Flight(f).img_path(0)) or '')} if False else {}
    ok = not bad and tot == 18571 and flat == 1813 and vflat == 1058
    report('C1', 'PASS' if ok else 'FAIL', 'counts', 'total %d flat %d very flat %d; mismatches: %s' % (tot, flat, vflat, bad[:5] or 'none'))


def c2_k(R):
    bad = {rid: sum(1 for r in rows if r[4] != 100) for rid, rows in R.items()}
    report('C2', 'PASS' if not any(bad.values()) else 'FAIL', 'K = 100', 'rows with n_sel != 100 per run: %s' % bad)


def c3_keyframes(lab):
    issues = []
    for f in FLIGHTS:
        qs = sorted(int(q) for q in lab[f]); N = max(qs) + 1
        if any(q % 10 == 0 for q in qs): issues.append(f + ' keyframe among queries')
        exp = [q for q in range(N) if q % 10]
        if qs != exp[:len(qs)]: issues.append(f + ' query list differs from every-non-10th rule')
    # window rule on one flight with images available
    F = Flight('oldtown/P001'); sizes = [len(F.near(q)) for q in F.queries]
    hist = {s: sizes.count(s) for s in sorted(set(sizes))}
    ok = not issues and all(abs(k - q) <= WIN for q in F.queries for k in F.near(q))
    report('C3', 'PASS' if ok else 'FAIL', 'keyframe rule', 'issues %s; keyframes per query on oldtown/P001: %s' % (issues[:3] or 'none', hist))


def c4_reproduce():
    """same machine twice must be identical; the stored runs are compared too (LightGlue output depends on the device)"""
    from .evaluate import evaluate
    sp = load_splits(); diffs, flips, twice = {}, {}, {}
    for picker, rid in (('sd', 'raco_sd_K100_same_lg'), ('cube0.5', 'raco_cube0.5_K100_same_lg')):
        _, rows, _ = evaluate(picker, 'micro'); _, rows2, _ = evaluate(picker, 'micro')
        twice[picker] = sum(a != b for a, b in zip(rows, rows2))
        mine = {(r[0], r[1]): r for r in rows}; d = fl = 0
        for f, qs in sp['micro'].items():
            st = {r[1]: r for r in json.load(gzip.open(os.path.join(REPO, 'results', 'runs', rid, slug(f) + '.json.gz'), 'rt'))['rows']}
            d += sum(1 for q in qs if mine[(f, q)][2:] != st[q][2:])
            fl += sum(1 for q in qs if fail_flags([mine[(f, q)]])[0] != fail_flags([st[q]])[0])
        diffs[picker] = d; flips[picker] = fl
    ok_same = not any(twice.values())
    from .data import FLIGHTS
    mixed = [f for f in FLIGHTS if len({json.load(gzip.open(os.path.join(REPO, 'results', 'runs', r, slug(f) + '.json.gz'), 'rt')).get('device')
                                        for r in ('raco_sd_K100_same_lg', 'raco_cube0.5_K100_same_lg', 'raco_grid_K100_same_lg')} - {None}) != 1]
    status = 'PASS' if ok_same and not mixed else 'FAIL'
    report('C4', status, 'reproducibility',
           'same machine twice: rows differing %s. Stored baselines sd / cube0.5 / grid: flights whose three runs do not share one '
           'recorded device: %s (rerun 11 Oct, job 16816317: one node per flight). Laptop re-run vs stored Euler rows on MICRO '
           '(information, different device): rows differing %s, pass/fail flips %s. LightGlue output depends on the device, so pickers '
           'are compared only on one device per flight (gate: same job).' % (twice, mixed or 'none', diffs, flips))


def c5_ground_truth(n_q=60):
    from scipy.spatial.transform import Rotation as Rot
    from .bench import match_lightglue
    from .pickers import parse
    import cv2
    pk = parse('cube0.5'); F = Flight('oldtown/P001'); C = Cands(F.slug, 'raco')
    qs = F.queries[::max(1, len(F.queries) // n_q)][:n_q]
    sel = {}
    def pick(f):
        if f not in sel:
            u, v, s, z, de = C.frame(f); sel[f] = np.asarray(pk(u, v, s, z, 100), int)
        return sel[f]
    # independent pose reader: TartanAir pose_left.txt = tx ty tz qx qy qz qw (NED body); camera x=right(+y body), y=down(+z), z=fwd(+x)
    P = np.loadtxt(os.path.join(os.path.dirname(os.path.dirname(F.img_path(0))), 'pose_left.txt'))
    B = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float).T     # columns = camera axes in body frame
    def gt(f):
        Rwb = Rot.from_quat(P[f, 3:7]).as_matrix(); return Rwb @ B, P[f, :3]     # camera-to-world, centre
    reproj, drot, derr, conv = [], [], [], []
    for q in qs:
        near = F.near(q); u, v, s, z, de = C.frame(q); i = pick(q)
        kfs, MAP = [], []
        for k in near:
            uk, vk, sk, zk, dk = C.frame(k); j = pick(k); j = j[zk[j] < DEPTH_MAX]
            Rk, ck = gt(k); MAP.append(to3d(uk[j], vk[j], zk[j]) @ Rk.T + ck); kfs.append((np.stack([uk[j], vk[j]], 1), dk[j]))
            Rk2, ck2 = F.cam_pose(k); conv.append(max(np.abs(Rk2 - Rk).max(), np.abs(ck2 - ck).max()))
        mq, pairs = match_lightglue(np.stack([u[i], v[i]], 1), de[i], kfs)
        if len(mq) < 6: continue
        obj = np.array([MAP[a][b] for a, b in pairs]); img = np.stack([u[i][mq], v[i][mq]], 1)
        Rq, cq = gt(q); Xc = (obj - cq) @ Rq; pr = Xc[:, :2] / Xc[:, 2:3] * KMAT[0, 0] + KMAT[:2, 2]
        e = np.linalg.norm(pr - img, axis=1); reproj.append(np.median(e))
        cv2.setRNGSeed(0)
        ok, rv, tv, inl = cv2.solvePnPRansac(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2).astype(np.float64), KMAT, None,
                                             reprojectionError=3.0, iterationsCount=2000, confidence=0.9999, flags=cv2.SOLVEPNP_AP3P)
        if not ok or inl is None or len(inl) < 6: continue
        inl = inl.ravel(); rv, tv = cv2.solvePnPRefineLM(obj[inl].reshape(-1, 1, 3), img[inl].reshape(-1, 1, 2).astype(np.float64), KMAT, None, rv, tv)
        Rcw = Rot.from_rotvec(rv.ravel()).as_matrix()                   # world -> camera (estimate)
        rot_ind = np.degrees(Rot.from_matrix(Rcw @ Rq).magnitude())    # Rq = camera->world true, so Rcw Rq = I if exact
        err_ind = np.linalg.norm(-Rcw.T @ tv.ravel() - cq) * 100
        from .bench import pnp
        err_b, rot_b, _ = pnp(obj, img, F.cam_pose(q)[0], F.cam_pose(q)[1])
        if err_b is not None:
            derr.append(abs(err_b - err_ind)); drot.append(abs(rot_b - rot_ind))
    med = float(np.median(reproj))
    ok = med < 2.0 and max(derr) < 0.01 and max(drot) < 0.01 and max(conv) < 1e-6
    report('C5', 'PASS' if ok else 'FAIL', 'ground truth and error formulas',
           'median reprojection of matches under the TRUE pose %.2f px (%d queries); independent pose reader vs code max diff %.1e; '
           'independent err/rot vs code max |diff| %.4f cm / %.4f deg' % (med, len(reproj), max(conv), max(derr), max(drot)))


def c6_fail_rule(R):
    js = open(os.path.join(REPO, 'docs', 'app.js'), encoding='utf-8').read()
    m = re.search(r'function isFail\(r, T\) \{\s*return (.+?);', js)
    rule_js = m.group(1) if m else ''
    idx = json.load(open(os.path.join(REPO, 'docs', 'data', 'index.json')))
    rot25 = (idx.get('rot_fail') or {}).get('25', idx.get('rot_fail_deg'))
    exp_js = "r.err == null || r.err > T || (r.rot != null && r.rot > rotFail(T))"
    # python and page agree row by row when rot is null exactly when err is null
    odd = sum(1 for rows in R.values() for r in rows if (r[2] is None) != (r[3] is None))
    ok = rule_js == exp_js and rot25 == 2 and odd == 0
    report('C6', 'PASS' if ok else 'FAIL', 'fail rule', 'page isFail = "%s"; rotation threshold at 25 cm in index.json = %s deg; rows with err/rot null mismatch %d'
           % (rule_js, rot25, odd))


def c7_page(R):
    idx = json.load(open(os.path.join(REPO, 'docs', 'data', 'index.json'))); bad = []
    fl = [f['id'] for f in idx['flights']]
    for rid, rows in R.items():
        d = json.load(open(os.path.join(REPO, 'docs', 'data', 'runs', rid + '.json')))
        page = [[fl[r[0]]] + r[1:] for r in d['rows']]
        if page != rows: bad.append(rid + ' rows differ')
        ent = [r for r in idx['runs'] if r['id'] == rid][0]
        f25 = round(100 * fail_flags(rows).mean(), 2)
        if abs(ent['fail25'] - f25) > 0.005: bad.append('%s fail25 %.2f vs %.2f' % (rid, ent['fail25'], f25))
    report('C7', 'PASS' if not bad else 'FAIL', 'page data = run data', 'issues: %s' % (bad or 'none'))


def c8_bootstrap(R, lab):
    rows = R['raco_cube0.5_K100_same_lg']; fails = fail_flags(rows)
    blocks = block_ids([r[0] for r in rows], [r[1] for r in rows])
    p, lo, hi = boot_ci(fails, blocks)
    js = open(os.path.join(REPO, 'docs', 'app.js'), encoding='utf-8').read()
    parts = [re.search(r'function hashStr[\s\S]+?\n\}', js).group(0), re.search(r'function mulberry32[\s\S]+?\n\}', js).group(0),
             'const N_BOOT = 1000;', re.search(r'function bootstrapCI[\s\S]+?\n\}', js).group(0)]
    data = dict(fails=fails.astype(int).tolist(), blocks=blocks.tolist())
    with tempfile.TemporaryDirectory() as td:
        jf = os.path.join(td, 'd.json'); json.dump(data, open(jf, 'w'))
        sf = os.path.join(td, 'b.js')
        open(sf, 'w').write('\n'.join(parts) + "\nconst d=require(%s);console.log(JSON.stringify(bootstrapCI(d.fails,'audit',d.blocks)));" % json.dumps(jf))
        out = json.loads(subprocess.check_output(['node', sf]).decode())
    ok = abs(out['lo'] - lo) < 0.1 and abs(out['hi'] - hi) < 0.1
    report('C8', 'PASS' if ok else 'FAIL', 'bootstrap python = page',
           'cube0.5 all frames %.2f %%: python [%.2f, %.2f], page JS [%.2f, %.2f]' % (p, lo, hi, out['lo'], out['hi']))


def c9_paired_test(R, lab):
    a = R['raco_cube0.5_K100_same_lg']; b = R['raco_sd_K100_same_lg']
    fa, fb = fail_flags(a), fail_flags(b); fl = np.array([r[0] for r in a])
    saves = int(((fb == 1) & (fa == 0)).sum()); breaks = int(((fa == 1) & (fb == 0)).sum())
    p_query = min(1.0, 2 * sign_test(saves, breaks))
    blocks = block_ids(fl, [r[1] for r in a])
    inv = np.unique(blocks, return_inverse=True)[1]
    bd = np.bincount(inv, fa - fb)
    bs, bb = int((bd < 0).sum()), int((bd > 0).sum()); p_block = min(1.0, 2 * sign_test(bs, bb))
    b_, w_, t_ = per_flight(fa, fb, fl); p_flight = min(1.0, 2 * sign_test(b_, w_))
    report('C9', 'WARN', 'page paired p-value is per query',
           'cubes vs strongest, all frames: page-style per-query sign test saves %d breaks %d p=%.1e; per 20-frame block %d vs %d p=%.1e; '
           'per flight %d vs %d (tied %d) p=%.3f. Neighbouring queries fail together, so the per-query p overstates certainty; '
           'the page should show the block or flight test.' % (saves, breaks, p_query, bs, bb, p_block, b_, w_, t_, p_flight))


def c10_block_length(R, lab):
    a = R['raco_cube0.5_K100_same_lg']; b = R['raco_sd_K100_same_lg']; fa, fb = fail_flags(a), fail_flags(b)
    fl = [r[0] for r in a]; qs = [r[1] for r in a]; flat = flat_of(a, lab); out = []
    for L in (1, 10, 20, 40, 80):
        bl = block_ids(fl, qs, L); dall = paired_diff(fa, fb, bl); dfl = paired_diff(fa[flat], fb[flat], bl[flat])
        out.append('%d: all %.2f [%.2f, %.2f] flat %.2f [%.2f, %.2f]' % (L, *dall, *dfl))
    report('C10', 'INFO', 'block length sensitivity (cubes - strongest, points)', '; '.join(out))
    return out


def c11_headline(R, lab):
    q = {}
    for rid in ('raco_sd_K100_same_lg', 'raco_cube0.5_K100_same_lg', 'raco_grid_K100_same_lg'):
        rows = R[rid]; f = fail_flags(rows); fl = flat_of(rows, lab); bl = block_ids([r[0] for r in rows], [r[1] for r in rows])
        p, lo, hi = boot_ci(f, bl); q[rid] = (p, (hi - lo) / 2, 100 * f[fl].mean())
    a, b = R['raco_cube0.5_K100_same_lg'], R['raco_sd_K100_same_lg']
    d = paired_diff(fail_flags(a), fail_flags(b), block_ids([r[0] for r in a], [r[1] for r in a]))
    bw = per_flight(fail_flags(a), fail_flags(b), np.array([r[0] for r in a]))
    s = ', '.join('%s %.2f +-%.2f flat %.1f' % (k.split('_')[1], *v) for k, v in q.items())
    exp = dict(sd=(2.4, 16.3), cube0=(1.7, 12.5), grid=(1.8, 13.6))   # device-consistent rerun 11 Oct (job 16816317)
    ok = all(abs(round(v[0], 1) - exp[k.split('_')[1].split('.')[0]][0]) < 0.051 and abs(round(v[2], 1) - exp[k.split('_')[1].split('.')[0]][1]) < 0.051
             for k, v in q.items()) and abs(d[0] + 0.67) < 0.01
    report('C11', 'PASS' if ok else 'FAIL', 'headline numbers', '%s; paired cubes-strongest %.2f [%.2f, %.2f]; flights better/worse/tied %s' % (s, *d, bw))


def c12_planarity(lab, n_per=40):
    rng = np.random.default_rng(1); changed = []; nq = 0; diffs = []
    for f in ['oldtown/P001', 'oldtown/P000', 'hospital/P041']:
        C = Cands(slug(f), 'sp'); qs = sorted(int(q) for q in lab[f])
        pfs = np.array([lab[f][str(q)].get('plane_frac') or 0 for q in qs])
        near = [q for q, p in zip(qs, pfs) if abs(p - 0.85) < 0.08]            # the queries whose bin could flip
        pick = list(rng.choice(near, min(n_per, len(near)), replace=False)) if near else []
        for q in pick:
            u, v, s, z, _ = C.frame(int(q)); ok = np.flatnonzero(z < DEPTH_MAX)[:300]; P = to3d(u[ok], v[ok], z[ok])
            best = 0; r2 = np.random.default_rng(int(q))
            for _ in range(3000):                         # 10x the trials of labels.py, then least-squares refit of the best plane
                i = r2.choice(len(P), 3, replace=False); n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]]); nn = np.linalg.norm(n)
                if nn < 1e-9: continue
                n /= nn; inl = np.abs((P - P[i[0]]) @ n) < 0.3
                if inl.sum() > best:
                    best = inl.sum(); c = P[inl].mean(0); _, _, vt = np.linalg.svd(P[inl] - c); n2 = vt[2]
                    best = max(best, int((np.abs((P - c) @ n2) < 0.3).sum()))
            new = best / len(P); old = lab[f][str(q)]['plane_frac']; nq += 1; diffs.append(new - old)
            if (old >= 0.85) != (new >= 0.85): changed.append('%s/%d %.3f->%.3f' % (f, q, old, new))
    report('C12', 'WARN' if changed else 'PASS', 'planarity labels',
           '%d near-threshold queries refit with 3000 trials + LS: mean change %+.3f, max %+.3f; bin flips %d: %s'
           % (nq, np.mean(diffs), np.max(diffs), len(changed), changed[:6]))


def c13_canaries():
    from .evaluate import evaluate, PickerError
    def rnd(u, v, s, z, K):
        r = np.random.default_rng(len(u) * 7919 + int(s[0] * 1e6) % 1000); return r.choice(len(u), min(K, len(u)), replace=False)
    def dup(u, v, s, z, K): return np.zeros(K, int)
    def few(u, v, s, z, K): return np.arange(K - 1)
    o_r, _, _ = evaluate(rnd, 'micro', name='random'); o_s, _, _ = evaluate('sd', 'micro')
    refused = []
    for fn in (dup, few):
        try: evaluate(fn, 'micro')
        except PickerError as e: refused.append('%s: %s' % (fn.__name__, e))
    from .pickers import parse
    try: evaluate('orep', 'micro'); refused.append('orep NOT refused')
    except PickerError as e: refused.append('orep: %s' % e)
    ok = o_r['fail_flat'] > o_s['fail_flat'] and len([x for x in refused if 'NOT' not in x]) == 3
    report('C13', 'PASS' if ok else 'FAIL', 'canaries', 'MICRO flat fail: random %.1f %% vs strongest %.1f %%; refused: %s' % (o_r['fail_flat'], o_s['fail_flat'], refused))


def c14_texts():
    issues = []
    js = open(os.path.join(REPO, 'docs', 'app.js'), encoding='utf-8').read()
    rd = open(os.path.join(REPO, 'README.md'), encoding='utf-8').read()
    spec = open(os.path.join(REPO, 'docs', 'data_spec.md'), encoding='utf-8').read()
    if 'rotation above 5 deg' in js: issues.append('app.js fail-% tooltip says "rotation above 5 deg" (code: 2 deg at 25 cm, 5 at 50 cm, 10 at 5 m)')
    if re.search(r"n_match: 'mean number of mutual nearest neighbour", js): issues.append('app.js n_match tooltip says mutual NN (LightGlue runs use LightGlue matches)')
    if 'rotation error is above 5 degrees' in rd: issues.append('README protocol: "rotation error above 5 degrees" (code: 2 deg at 25 cm)')
    if 'Matching = mutual nearest neighbour' in rd: issues.append('README protocol: matching = mutual NN only (LightGlue runs are the headline)')
    if 'AP3P' not in rd: issues.append('README protocol: solver lacks AP3P minimal solver + LM refinement and camera-centre error')
    if 'mutual nearest-neighbour matches to the map' in spec: issues.append('data_spec n_match: mutual NN only')
    report('C14', 'FAIL' if issues else 'PASS', 'protocol texts', '; '.join(issues) or 'consistent')
    return issues


def c15_incumbent(lab):
    rid = 'raco_dpp0.5_K100_same_lg'; rows = run_rows(rid); f = fail_flags(rows)
    pf = np.array([lab[r[0]][str(r[1])].get('plane_frac') or -1 for r in rows])
    from .bench import COLS
    rep = [r[COLS.index('rep5')] for r in rows if r[COLS.index('rep5')] is not None]
    got = dict(n=len(rows), k_bad=sum(1 for r in rows if r[4] != 100), all=round(float(100 * f.mean()), 2), flat=round(float(100 * f[pf >= 0.85].mean()), 2),
               vflat=round(float(100 * f[pf >= 0.95].mean()), 2), rep5=round(float(np.mean(rep)), 4))
    lb = [e for e in json.load(open(os.path.join(REPO, 'results', 'leaderboard.json')))['runs'] if e['id'] == rid]
    lbv = dict(all=round(lb[0]['all'][0], 2), flat=round(lb[0]['flat'][0], 2), vflat=round(lb[0]['vflat'][0], 2), rep5=lb[0]['rep5']) if lb else None
    exp = dict(n=18571, k_bad=0, all=1.38, flat=10.59, vflat=16.07, rep5=0.5997)
    ok = all(abs(got[k] - exp[k]) < 0.0051 for k in exp) and lbv is not None and all(abs(lbv[k] - got[k]) < 0.0051 for k in lbv)
    from .gate import compare, load_run, RESULTS
    try:   # the gate runs (same node per flight) live in results/runs on Euler, results/runs_gate in the laptop mirror
        gr = 'runs' if os.path.isdir(os.path.join(RESULTS, 'runs', rid + '_g-dpp0.5')) else 'runs_gate'
        g = compare(rid + '_g-dpp0.5', 'raco_cube0.5_K100_same_lg_g-dpp0.5', 'held_out', gr)
        ho = load_splits()['held_out']; _, gc = load_run('raco_cube0.5_K100_same_lg_g-dpp0.5', ho, gr); _, hc = load_run('raco_cube0.5_K100_same_lg', ho)
        flips = sum(int(np.sum(fail_flags(gc[f][0]) != fail_flags(hc[f][0]))) for f in ho)
        g['flips'] = flips
        gs = 'held-out flat %.2f vs %.2f, diff %.2f [%.2f, %.2f], %d flat, flights %d/%d, win %s, devices %s, gate cube vs headline cube fail flips %d' % (
            g['flat_cand'], g['flat_inc'], *g['flat_diff'], g['n_flat'], g['flights_better'], g['flights_worse'], g['win'], g['device_check'], flips)
        ok = ok and round(g['flat_cand'], 2) == 6.50 and round(g['flat_inc'], 2) == 9.24 and round(g['flat_diff'][0], 2) == -2.74             and g['n_flat'] == 877 and (g['flights_better'], g['flights_worse']) == (6, 0) and g['win'] and flips == 0
    except Exception as e:
        gs = 'gate recompute failed: %s' % e; ok = False
    report('C15', 'PASS' if ok else 'FAIL', 'incumbent dpp0.5 numbers', 'run %s; leaderboard %s; %s' % (got, lbv, gs))


def write_report(path):
    lines = ['# kpbench audit %s' % datetime.date.today().isoformat(), '',
             'Generated by `python -m kpbench.audit`. FAIL = a published number or text is wrong; WARN = correct but misleading or fragile.', '',
             '| check | status | what | result |', '|---|---|---|---|']
    for cid, st, t, d in RES:
        lines.append('| %s | %s | %s | %s |' % (cid, st, t, d.replace('|', '/').replace('\n', ' ')))
    open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true', help='skip checks that re-run the pipeline (C4, C5, C12, C13)')
    ap.add_argument('--only', default='', help='comma list, e.g. C3,C4 (laptop without flight data / Euler without node); report tagged')
    a = ap.parse_args(argv)
    lab = labels(); R = {rid: run_rows(rid) for rid in RUNS31}
    C = dict(C1=lambda: c1_counts(lab, R), C2=lambda: c2_k(R), C3=lambda: c3_keyframes(lab), C6=lambda: c6_fail_rule(R),
             C7=lambda: c7_page(R), C8=lambda: c8_bootstrap(R, lab), C9=lambda: c9_paired_test(R, lab),
             C10=lambda: c10_block_length(R, lab), C11=lambda: c11_headline(R, lab), C14=c14_texts, C15=lambda: c15_incumbent(lab),
             C4=c4_reproduce, C5=c5_ground_truth, C12=lambda: c12_planarity(lab), C13=c13_canaries)
    only = [c.strip() for c in a.only.split(',') if c.strip()]
    for cid in only or [c for c in C if not (a.quick and c in ('C4', 'C5', 'C12', 'C13'))]:
        C[cid]()
    os.makedirs(os.path.join(REPO, 'reports'), exist_ok=True)
    tag = ('_' + '-'.join(only)) if only else ''
    fn = os.path.join(REPO, 'reports', 'audit_%s%s.md' % (datetime.date.today().isoformat(), tag)); write_report(fn); print('report', fn)


if __name__ == '__main__':
    main()
