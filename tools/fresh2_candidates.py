"""Candidate flights for a FRESH-2 extension (MAIN, dispatch 2026-11-20): what is left outside DEV 12, HELD_OUT 19 and
FRESH-1 14, with the flat count each flight can be expected to give. Reads no picker row and no FRESH-2 frame.

  python tools/fresh2_candidates.py --pool <agent repo>/kpbench/fresh_pool.json > results/main/fresh2_extension_candidates.md

Inputs: fresh_pool.json (Euler census of 18 Oct: 369 trajectories with frame counts, and a depth-only flat estimate for the
56 same-environment ones), results/gates/FRESH_set.json (FRESH-1 flights with their true flat counts), splits.json.
Queries of a flight = frames - ceil(frames / 10) (every 10th frame is a keyframe). Flat = plane_frac >= 0.85.
Calibration: the depth-only estimate is compared with the true count on the 14 FRESH-1 flights (the only flights that have
both); ratio = true / estimate, interval = flight bootstrap of that ratio (10000, seed 0). Toy: estimates (100, 50), true
(120, 30) -> ratio 150 / 150 = 1.0, but single flights are off by +20 % and -40 %.
"""
import argparse, json, math, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ENV = dict(oldtown='oldtown', hospital='hospital', factory='abandonedfactory')
LOW, HIGH = 0.0976, 0.154            # flat share of the 31 board flights / of FRESH-1 (brackets where no depth estimate exists)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pool', required=True); a = ap.parse_args()
    P = json.load(open(a.pool)); S = json.load(open(os.path.join(ROOT, 'splits.json')))
    F1 = json.load(open(os.path.join(ROOT, 'results', 'gates', 'FRESH_set.json')))
    key = lambda k: '%s/%s/%s' % (ENV[k.split('/')[0].split('_')[0]], k.split('/')[0].split('_')[1].capitalize(), k.split('/')[1])
    fresh1 = {key(k): v for k, v in F1['per_flight'].items()}
    used = {k for k, v in P['counts'].items() if v['ours']} | set(fresh1)
    assert sum(v['ours'] for v in P['counts'].values()) == len(S['dev']) + len(S['held_out']) == 31 and len(used) == 45
    q = lambda k: P['counts'][k]['rgb'] - math.ceil(P['counts'][k]['rgb'] / 10)

    # calibration of the depth-only estimate on FRESH-1
    est = np.array([P['pool'][k]['est_flat'] for k in fresh1]); true = np.array([fresh1[k]['flat'] for k in fresh1], float)
    rng = np.random.default_rng(0); I = rng.integers(0, len(est), (10000, len(est)))
    R = true[I].sum(1) / est[I].sum(1); r0 = true.sum() / est.sum(); rlo, rhi = np.percentile(R, [2.5, 97.5])
    same = sorted(k for k in P['pool'] if k not in used)
    e_same = sum(P['pool'][k]['est_flat'] for k in same)
    print('# FRESH-2 extension: candidate flights and reachable flat counts (MAIN, engineer 2026-11-20)\n')
    print('Script: `python tools/fresh2_candidates.py --pool <agent repo>/kpbench/fresh_pool.json`. No picker row and no FRESH-2 frame was read; nothing ran on Euler.\n')
    print('## Calibration of the flat estimate (14 FRESH-1 flights, the only ones with estimate and truth)')
    print('- true flat %d vs depth-only estimate %.0f -> ratio %.3f, flight-bootstrap 95 %% interval [%.3f, %.3f].' % (true.sum(), est.sum(), r0, rlo, rhi))
    print('- single flights: true / estimate from %.2f to %.2f (flights with estimate >= 20); largest miss %s (%d true vs %.0f).'
          % (min(t / e for t, e in zip(true, est) if e >= 20), max(t / e for t, e in zip(true, est) if e >= 20),
             list(fresh1)[int(np.argmax(np.abs(true - est)))], true[np.argmax(np.abs(true - est))], est[np.argmax(np.abs(true - est))]))
    print('\n## A. Registered pool: same environments, never run (%d flights)' % len(same))
    print('- queries %d, flat estimate %.0f; with the calibration ratio: %.0f [%.0f, %.0f].' % (sum(q(k) for k in same), e_same, e_same * r0, e_same * rlo, e_same * rhi))
    print('- same-environment flights beyond these: 0 (every oldtown / hospital / abandonedfactory trajectory is in DEV, HELD_OUT, FRESH-1 or this pool).\n')
    print('| flight | queries | flat est |'); print('|---|---|---|')
    for k in same:
        print('| %s | %d | %.0f |' % (k, q(k), P['pool'][k]['est_flat']))

    # new environments: no labels, no depth estimate -> brackets
    new = sorted(k for k in P['counts'] if k not in used and k not in P['pool'])
    envs = {}
    for k in new:
        envs.setdefault(k.rsplit('/', 1)[0], []).append(k)
    print('\n## B. Extension candidates: new environments (%d flights in %d environment/difficulty groups; no labels, no depth estimate)' % (len(new), len(envs)))
    print('Flat is bracketed at %.2f %% (share on the 31 board flights) and %.1f %% (share on FRESH-1) of the queries.\n' % (100 * LOW, 100 * HIGH))
    print('| group | flights | queries | flat at %.2f %% | flat at %.1f %% | flights (queries) |' % (100 * LOW, 100 * HIGH)); print('|---|---|---|---|---|---|')
    for g, ks in envs.items():
        n = sum(q(k) for k in ks)
        print('| %s | %d | %d | %.0f | %.0f | %s |' % (g, len(ks), n, n * LOW, n * HIGH, ' '.join('%s(%d)' % (k.rsplit('/', 1)[1], q(k)) for k in ks)))
    nq = sum(q(k) for k in new)
    print('| **total** | **%d** | **%d** | **%.0f** | **%.0f** | |' % (len(new), nq, nq * LOW, nq * HIGH))

    print('\n## Reachability')
    print('| target | from pool A alone (est %.0f, calibrated low %.0f) | extra flat needed (est / calibrated low) | new-env queries needed at %.2f %% (est / low) | share of the %d new-env queries |'
          % (e_same, e_same * rlo, 100 * LOW, nq)); print('|---|---|---|---|---|')
    for t in (3000, 3500, 4150):
        need = [max(0.0, t - e_same), max(0.0, t - e_same * rlo)]
        print('| %d flat | %s | %.0f / %.0f | %.0f / %.0f | %.1f %% / %.1f %% |' % (t, 'yes' if e_same >= t else 'NO', need[0], need[1], need[0] / LOW, need[1] / LOW, 100 * need[0] / LOW / nq, 100 * need[1] / LOW / nq))
    print('| >= 20 flights | yes (%d) | 0 | 0 | 0 |' % len(same))
    print('\nAll three flat targets are reachable only by adding NEW-environment flights (0 same-environment flights are left); the new-environment pool holds %.0f-%.0f flat by the brackets, %.0fx-%.0fx the largest need.'
          % (nq * LOW, nq * HIGH, nq * LOW / max(1.0, 4150 - e_same * rlo), nq * HIGH / max(1.0, 4150 - e_same * rlo)))


if __name__ == '__main__':
    main()
