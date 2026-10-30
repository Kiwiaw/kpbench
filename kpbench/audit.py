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
  C24 provisional  summaries tagged PROVISIONAL-936 (hand summaries with --allow-label-mismatch while 6 Euler DEV label files
                   are stale): repo labels match the manifest and give the splits.json DEV flat count; every provisional file
                   names that count; no provisional picker or number in leaderboard.json / page data; dpp2d / ssc rows say
                   'baseline' (dispatch 2026-10-18)
  C37 rep5        every gate file with a 'rep5' block recomputed from its rows with an independent paired mean, flight sign count
                   and block bootstrap (the gate code path is not used); rows on Euler, the laptop has the held-out dpp0.5 pair only;
                   per-query rep5 must be whole counts / K, and files with rows absent are listed, not passed
  C36 nms0 = sd  nms0 picks = sd picks on the MICRO candidates (results/micro/nms_smoke.json, tools/nms_smoke.py), nms16 fills K
                   with no pair closer than 16 px; any stored nms0 run equals the sd run flight by flight
  C23 fresh        FRESH-1 (results/gates/FRESH_set.json) shares no trajectory with the 31 flights (DEV + HELD_OUT): no
                   identical pose_left.txt and no fresh pose within 1 cm of a 31-flight pose of the same environment;
                   splits_fresh.json / fresh manifest hashes = FRESH_set.json; flat count >= 2 x DEV flat (needs flight data)
  C25 gates        every DEV gate file results/gates/*__dev.json recomputed from its rows with gate.compare (flat diff + CI,
                   flights better/worse, verdict); needs the gate runs (Euler) (dispatch 2026-10-19 13:10)
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
    js = open(os.path.join(REPO, 'docs', 'app.js'), encoding='utf-8').read()
    ht = open(os.path.join(REPO, 'docs', 'index.html'), encoding='utf-8').read()
    i_f, i_b, i_q = (js.find(h) for h in ('flights better/worse, p (main)', 'p, 20-frame blocks', 'p per query (too optimistic)'))
    ok = 0 <= i_f < i_b < i_q and 'p = two-sided sign test on saves versus breaks' not in ht
    report('C9', 'PASS' if ok else 'WARN', 'page leads with the per-flight sign test' if ok else 'page paired p-value is per query',
           'cubes vs strongest, all frames: per-query sign test saves %d breaks %d p=%.1e; per 20-frame block %d vs %d p=%.1e; '
           'per flight %d vs %d (tied %d) p=%.3f. Neighbouring queries fail together, so the per-query p overstates certainty; '
           'page column order flight / block / per-query %s.' % (saves, breaks, p_query, bs, bb, p_block, b_, w_, t_, p_flight,
                                                                'OK' if ok else 'NOT OK'))


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


PROV = 'PROVISIONAL-936'
BASELINE_ARMS = ('dpp2d', 'ssc')


def c24_provisional():
    from .gate import label_guard
    from .splits import flat_queries
    dev = load_splits()['dev']; n_flat = sum(len(flat_queries(f)) for f in dev)
    bad = label_guard(dev, n_flat, 'dev')
    files = [fn for fn in glob.glob(os.path.join(REPO, 'results', '**', '*.md'), recursive=True) + glob.glob(os.path.join(REPO, 'docs', '*.md'))
             if PROV in ''.join(open(fn, encoding='utf-8', errors='replace').readlines()[:3])]
    for fn in files:
        t = open(fn, encoding='utf-8', errors='replace').read()
        if str(n_flat) not in t:
            bad.append('%s does not name the %d flat frames' % (os.path.relpath(fn, REPO), n_flat))
        for line in t.splitlines():
            if line.startswith('|') and any(re.search(r'%s[0-9.]' % a, line) for a in BASELINE_ARMS) and 'baseline' not in line.lower():
                bad.append('%s: baseline-arm row without label: %s' % (os.path.relpath(fn, REPO), line[:60]))
    pub = [os.path.join(REPO, 'results', 'leaderboard.json')] + glob.glob(os.path.join(REPO, 'docs', 'data', '*.json'))
    for fn in pub:
        t = open(fn, encoding='utf-8', errors='replace').read()
        if PROV in t or any(re.search(r'"raco_%s[0-9.]' % a, t) for a in BASELINE_ARMS):
            bad.append('%s carries a provisional number or baseline-arm run' % os.path.relpath(fn, REPO))
    report('C24', 'PASS' if not bad else 'FAIL', 'provisional-936 summaries',
           'DEV flat %d from repo labels; %d provisional files %s; %d published files clean%s' % (
               n_flat, len(files), [os.path.relpath(f, REPO) for f in files], len(pub) - sum(1 for b in bad if 'published' in b),
               ('; problems: ' + '; '.join(bad[:6])) if bad else ''))


def c23_fresh():
    import hashlib
    from .data import Flight, FLIGHTS, flight_dir
    from .splits import load as ls
    fs = json.load(open(os.path.join(REPO, 'results', 'gates', 'FRESH_set.json'))); bad = []; info = []
    for fn, key in (('splits_fresh.json', 'splits_fresh_sha256'), (os.path.join('results', 'labels_manifest_fresh.json'), 'manifest_sha256')):
        p = os.path.join(REPO, fn)
        if not os.path.isfile(p):
            bad.append('%s missing' % fn); continue
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()   # raw bytes or canonical JSON (sorted keys), as gate.label_hash
        hc = hashlib.sha256(json.dumps(json.load(open(p)), sort_keys=True).encode()).hexdigest()
        if fs[key] not in (h, hc):
            bad.append('%s sha %s != FRESH_set %s' % (fn, h[:10], fs[key][:10]))
    s = ls(); want = 2 * s['flat_counts']['dev']
    n_dev_flat = sum(len(flat_queries_(f)) for f in s['dev'])
    if fs['flat_counts']['fresh'] < want and fs['flat_counts']['fresh'] < 1872:
        bad.append('FRESH flat %d < target %d' % (fs['flat_counts']['fresh'], want))
    ref = {}
    for f in FLIGHTS:
        d = flight_dir(f)
        if d:
            P = np.loadtxt(os.path.join(d, 'pose_left.txt')); ref[f] = (hashlib.sha256(open(os.path.join(d, 'pose_left.txt'), 'rb').read()).hexdigest(), P[:, :3])
    if len(ref) < len(FLIGHTS):
        bad.append('only %d of %d reference flights have data here' % (len(ref), len(FLIGHTS)))
    near_min = []
    for f in fs['fresh']:
        d = flight_dir(f)
        if not d:
            bad.append('%s: no flight data' % f); continue
        P = np.loadtxt(os.path.join(d, 'pose_left.txt')); h = hashlib.sha256(open(os.path.join(d, 'pose_left.txt'), 'rb').read()).hexdigest()
        env = f.split('/')[0].split('_')[0]
        for g, (hg, Q) in ref.items():
            if hg == h:
                bad.append('%s pose file identical to %s' % (f, g))
            if g.split('/')[0] == env and len(P) == len(Q):
                dist = float(np.abs(P[:, :3] - Q).max())
                near_min.append((dist, f, g))
                if dist < 0.01:
                    bad.append('%s = %s (all positions within %.3f m)' % (f, g, dist))
    near_min.sort()
    info.append('%d fresh flights vs %d reference flights; DEV flat %d, FRESH flat %d / %d queries; closest same-length same-env pair %s' % (
        len(fs['fresh']), len(ref), n_dev_flat, fs['flat_counts']['fresh'], fs['n_queries']['fresh'],
        ('%s vs %s max |dpos| %.2f m' % (near_min[0][1], near_min[0][2], near_min[0][0])) if near_min else 'none'))
    report('C23', 'PASS' if not bad else 'FAIL', 'FRESH-1 disjoint from the 31 flights', '; '.join(info + bad[:8]))


def flat_queries_(f):
    from .splits import flat_queries
    return flat_queries(f)


def c25_gates():
    from .gate import compare, GATES, GateError
    rows = []; bad = []
    for fn in sorted(glob.glob(os.path.join(GATES, '*__dev.json'))):
        g0 = json.load(open(fn))
        try:
            g = compare(g0['cand'], g0['inc'], 'dev')
        except (GateError, OSError) as e:
            bad.append('%s: %s' % (g0.get('picker'), str(e)[:80])); continue
        d0, d1 = g0['flat_diff'], g['flat_diff']
        same = all(abs(a - b) < 0.006 for a, b in zip(d0, d1)) and (g0['flights_better'], g0['flights_worse']) == (g['flights_better'], g['flights_worse'])             and g0['win'] == g['win'] and g0['n_flat'] == g['n_flat']
        rows.append('%s %+.2f [%+.2f, %+.2f] %d/%d %s n_flat %d%s' % (g['picker'], *d1, g['flights_better'], g['flights_worse'],
                    'KEEP' if g['win'] else 'REVERT', g['n_flat'], '' if same else ' (file: %+.2f [%+.2f, %+.2f] %d/%d n_flat %d)' % (
                        *d0, g0['flights_better'], g0['flights_worse'], g0['n_flat'])))
        if not same:   # a file from before the 18 Oct label sync (937 flat) with the same verdict is stale, not wrong
            bad.append('%s %s' % (g['picker'], 'pre-sync file (n_flat %d), same verdict' % g0['n_flat']
                                  if g0['n_flat'] != g['n_flat'] and g0['win'] == g['win'] else 'differs from its file'))
    st = 'FAIL' if any('differs' in b for b in bad) else ('WARN' if bad else 'PASS')
    report('C25', st, 'DEV gate verdicts recomputed from rows', '%d recomputed: %s%s' % (len(rows), '; '.join(rows), ('; problems: ' + '; '.join(bad)) if bad else ''))


REP5_PINNED = {'dppr0.388a1.91b2.85t0.9': (0.0229, 0.0200, 0.0259, 11, 1), 'dppr0.317a1.6b2.01t0.9': (0.0236, 0.0209, 0.0263, 12, 0)}


def rep5_pair(cand, inc, flights, runs, flat_only, K=100):
    """independent rep5 recompute: pairs where both have rep5 -> dict(n, cand, inc, diff, lo, hi, better, worse) (seed 0, 1000 boots)"""
    from .bench import COLS
    from .gate import load_run, labels as gate_labels
    from .splits import FLAT
    j = COLS.index('rep5'); _, rc = load_run(cand, flights, runs); _, ri = load_run(inc, flights, runs); A, B, bl = [], [], []; better = worse = 0
    for f in flights:
        lab = gate_labels(f); a, b = [], []
        for x, y in zip(rc[f][0], ri[f][0]):
            assert x[1] == y[1], f
            pf = lab[str(x[1])].get('plane_frac')
            if (flat_only and not (pf is not None and pf >= FLAT)) or x[j] is None or y[j] is None:
                continue
            a.append(x[j]); b.append(y[j]); bl.append('%s:%d' % (f, x[1] // 20))
        if a:
            better += np.mean(a) > np.mean(b); worse += np.mean(a) < np.mean(b)
        A += a; B += b
    A, B = np.array(A), np.array(B); blocks = np.unique(bl, return_inverse=True)[1]; m = blocks.max() + 1
    s = np.bincount(blocks, A - B, m); c = np.bincount(blocks, None, m); pick = np.random.default_rng(0).integers(0, m, (1000, m))
    bs = s[pick].sum(1) / c[pick].sum(1)
    return dict(n=len(A), cand=float(A.mean()), inc=float(B.mean()), diff=float((A - B).mean()), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), better=int(better), worse=int(worse),
                whole=bool(np.all(np.abs(A * K - np.round(A * K)) < 1e-6) and np.all(np.abs(B * K - np.round(B * K)) < 1e-6)))


def c37_rep5():
    """gate rep5 blocks = recompute from rows; the stored CI has to come back within 0.001 (same seed, same scheme: normally equal)"""
    from .gate import GATES, RESULTS, split_info, GateError
    ok = []; bad = []; absent = []
    for fn in sorted(glob.glob(os.path.join(GATES, '*.json'))):
        g0 = json.load(open(fn))
        if 'rep5' not in g0 or not isinstance(g0.get('rep5'), dict) or 'flat' not in g0['rep5']:
            continue
        rs = [r for r in ('runs', 'runs_gate') if os.path.isdir(os.path.join(RESULTS, r, g0['cand'])) and os.path.isdir(os.path.join(RESULTS, r, g0['inc']))]
        if not rs:
            absent.append(os.path.basename(fn)[:40]); continue
        try:
            flights = split_info(g0['split'])[0]
            for name, fo in (('flat', True), ('all', False)):
                e = g0['rep5'][name]; r = rep5_pair(g0['cand'], g0['inc'], flights, rs[0], fo, g0.get('K', 100))
                same = r['n'] == e['n'] and abs(r['diff'] - e['diff'][0]) < 1e-6 and abs(r['lo'] - e['diff'][1]) < 1e-3 and abs(r['hi'] - e['diff'][2]) < 1e-3 and (r['better'], r['worse']) == (e['better'], e['worse']) and r['whole']
                (ok if same else bad).append('%s %s %+.4f [%+.4f,%+.4f] %d/%d n %d' % (g0.get('picker'), name, r['diff'], r['lo'], r['hi'], r['better'], r['worse'], r['n']))
        except (GateError, OSError, KeyError) as e:
            bad.append('%s: %s' % (g0.get('picker'), str(e)[:80]))
    for picker, exp in REP5_PINNED.items():   # gates older than the rep5 block (28 Oct): values quoted in NIGHT_2026-10-21_r7 / 22_r1
        cand, inc = 'raco_%s_K100_same_lg_g-%s' % (picker, picker), 'raco_dpp0.5_K100_same_lg_g-%s' % picker
        if not os.path.isdir(os.path.join(RESULTS, 'runs', cand)) or not os.path.isdir(os.path.join(RESULTS, 'runs', inc)):
            absent.append(picker); continue
        r = rep5_pair(cand, inc, split_info('dev')[0], 'runs', True)
        same = abs(r['diff'] - exp[0]) < 5e-5 and abs(r['lo'] - exp[1]) < 1e-3 and abs(r['hi'] - exp[2]) < 1e-3 and (r['better'], r['worse']) == exp[3:]
        (ok if same else bad).append('%s pinned %+.4f [%+.4f,%+.4f] %d/%d' % (picker, r['diff'], r['lo'], r['hi'], r['better'], r['worse']))
    st = 'FAIL' if bad else ('WARN' if absent or not ok else 'PASS')
    report('C37', st, 'gate rep5 recomputed from rows', '%d match: %s; mismatch %s; rows absent (Euler) %d' % (len(ok), '; '.join(ok), bad, len(absent)))


def c36_nms_identity():
    """nms0 = sd: picks on the MICRO candidates (tools/nms_smoke.py) and, where stored, nms0 run rows = sd run rows"""
    fn = os.path.join(REPO, 'results', 'micro', 'nms_smoke.json'); bad = []; det = []
    if not os.path.isfile(fn):
        return report('C36', 'WARN', 'nms0 = sd (nms family)', 'no results/micro/nms_smoke.json: run tools/nms_smoke.py')
    s = json.load(open(fn))
    det.append('MICRO %d frames: identity fails %d, fill fails %d, spread fails %d' % (s['frames'], s['identity_fail'], s['fill_fail'], s['spread_fail']))
    if not s['pass']: bad.append('smoke')
    for d in sorted(glob.glob(os.path.join(REPO, 'results', 'runs', '*_nms0_*'))):
        rid = os.path.basename(d); ref = rid.replace('_nms0_', '_sd_')
        if not os.path.isdir(os.path.join(REPO, 'results', 'runs', ref)): continue
        by = lambda rows: {f: [r for r in rows if r[0] == f] for f in set(r[0] for r in rows)}
        a, b = by(run_rows(rid)), by(run_rows(ref)); same = [f for f in a if f in b and a[f] == b[f]]
        det.append('%s = %s on %d/%d flights' % (rid, ref, len(same), len(a)))
        if len(same) < len(a): bad.append(rid)
    report('C36', 'FAIL' if bad else 'PASS', 'nms0 = sd (nms family)', '; '.join(det))


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
             C10=lambda: c10_block_length(R, lab), C11=lambda: c11_headline(R, lab), C14=c14_texts, C15=lambda: c15_incumbent(lab), C23=c23_fresh, C24=c24_provisional, C25=c25_gates, C36=c36_nms_identity, C37=c37_rep5,
             C4=c4_reproduce, C5=c5_ground_truth, C12=lambda: c12_planarity(lab), C13=c13_canaries)
    only = [c.strip() for c in a.only.split(',') if c.strip()]
    for cid in only or [c for c in C if not (a.quick and c in ('C4', 'C5', 'C12', 'C13'))]:
        C[cid]()
    os.makedirs(os.path.join(REPO, 'reports'), exist_ok=True)
    tag = ('_' + '-'.join(only)) if only else ''
    fn = os.path.join(REPO, 'reports', 'audit_%s%s.md' % (datetime.date.today().isoformat(), tag)); write_report(fn); print('report', fn)


if __name__ == '__main__':
    main()
