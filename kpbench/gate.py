"""Keep/revert gate: does a candidate run beat the incumbent run on a frozen split? (rule in kpbench.stats)

  python -m kpbench.gate --cand raco_cube0.5_K100_same_lg --inc raco_sd_K100_same_lg --split held_out [--log]

Both runs must hold every flight of the split, with the same queries, and each flight of both runs must come from the same
device (LightGlue output differs between devices: audit 11 Oct, C4). euler/gate.sbatch produces such pairs: candidate and
incumbent of one flight run back to back in one array task.
HELD_OUT discipline (--log): one held-out verdict per candidate picker, only after it beat the incumbent on DEV
(flat difference < 0); every verdict is appended to results/gates/log.jsonl and kept. --force-held-out overrides, logged.
Label guard (audit C15 at run time): before any verdict, every flight's label file must match results/labels_manifest.json
(canonical JSON sha256) and the flat frame count must equal splits.json flat_counts; else GUARD-FAIL, nothing written.
--allow-label-mismatch "<reason>" overrides, kept in the verdict. Rebuild the manifest only after an audit: --write-manifest.
Split fresh (near-miss retests, kpbench.fresh): flights and flat count from splits_fresh.json, label hashes from
results/labels_manifest_fresh.json (written once by --write-manifest --split fresh); held-out discipline does not apply.
The verdict json also carries 'rep5' (paired rep5 on flat and all queries, kpbench.rep5_paired): information, not rule.
"""
import os, json, gzip, hashlib, datetime
import numpy as np
from .data import slug
from .cache import REPO
from .splits import load as load_splits, FLAT
from .stats import fail_flags, gate as gate_rule

RESULTS = os.environ.get('KPBENCH_RESULTS', os.path.join(REPO, 'results'))
GATES = os.path.join(RESULTS, 'gates')
LOG = os.path.join(GATES, 'log.jsonl')
MANIFEST = os.path.join(REPO, 'results', 'labels_manifest.json')
MANIFEST_FRESH = os.path.join(REPO, 'results', 'labels_manifest_fresh.json')
SPLITS_FRESH = os.path.join(REPO, 'splits_fresh.json')


def split_info(split):
    """-> (flights, flat count the labels must give, manifest path)"""
    if split == 'fresh':
        s = json.load(open(SPLITS_FRESH)); return s['fresh'], s['flat_counts']['fresh'], MANIFEST_FRESH
    s = load_splits(); return s[split], s['flat_counts'].get(split), MANIFEST


class GateError(Exception):
    pass


def load_run(rid, flights, runs='runs'):
    d = os.path.join(RESULTS, runs, rid); meta = json.load(open(os.path.join(d, 'meta.json')))
    out = {}
    for f in flights:
        fn = os.path.join(d, slug(f) + '.json.gz')
        if not os.path.isfile(fn):
            raise GateError('%s: flight %s missing' % (rid, f))
        j = json.load(gzip.open(fn, 'rt')); out[f] = (j['rows'], j.get('device'))
    return meta, out


def labels(f):
    return json.load(open(os.path.join(REPO, 'results', 'labels', slug(f) + '.json')))


def label_hash(f, d=None):
    """sha256 of the label file as canonical JSON (sorted keys), so CRLF/LF checkouts hash the same"""
    fn = os.path.join(d or os.path.join(REPO, 'results', 'labels'), slug(f) + '.json')
    return hashlib.sha256(json.dumps(json.load(open(fn)), sort_keys=True).encode()).hexdigest()


def label_guard(flights, n_flat, split, d=None):
    """[] if labels match the manifest and the flat count matches splits.json, else the list of problems"""
    _, want, mf = split_info(split); man = json.load(open(mf)); bad = []
    for f in flights:
        h = label_hash(f, d)
        if man.get(f) != h:
            bad.append('%s label hash %s != manifest %s' % (f, h[:10], str(man.get(f))[:10]))
    if want is not None and n_flat != want:
        bad.append('%s flat frames %d != splits.json %d' % (split, n_flat, want))
    return bad


def log_entries():
    return [json.loads(l) for l in open(LOG)] if os.path.isfile(LOG) else []


def compare(cand, inc, split, runs='runs', allow_mismatch=''):
    flights = split_info(split)[0]
    mc, rc = load_run(cand, flights, runs); mi, ri = load_run(inc, flights, runs)
    for k in ('det', 'K', 'map', 'matcher'):
        if mc.get(k) != mi.get(k):
            raise GateError('runs differ in %s: %s vs %s' % (k, mc.get(k), mi.get(k)))
    fa, fb, fl, qs, flat, envs, dev = [], [], [], [], [], [], {}
    for f in flights:
        (ra, da), (rb, db) = rc[f], ri[f]
        if [r[1] for r in ra] != [r[1] for r in rb]:
            raise GateError('%s: query lists differ' % f)
        dev[f] = 'same' if da and da == db else ('unknown' if not (da and db) else 'DIFFERENT %s | %s' % (da, db))
        lab = labels(f)
        fa.append(fail_flags(ra)); fb.append(fail_flags(rb))
        for r in ra:
            pf = lab[str(r[1])].get('plane_frac')
            fl.append(f); qs.append(r[1]); flat.append(pf is not None and pf >= FLAT); envs.append(f.split('/')[0])
    bad = [f for f, v in dev.items() if v.startswith('DIFFERENT')]
    if bad:
        raise GateError('device mismatch on %d flights, e.g. %s: %s' % (len(bad), bad[0], dev[bad[0]]))
    guard = label_guard(flights, int(np.sum(flat)), split)
    if guard and not allow_mismatch:
        raise GateError('GUARD-FAIL (labels differ from the audited manifest, no verdict written): ' + '; '.join(guard))
    g = gate_rule(np.concatenate(fa), np.concatenate(fb), np.array(fl), np.array(qs), np.array(flat), np.array(envs))
    if guard:
        g['label_guard'] = dict(problems=guard, allowed=allow_mismatch)
    g.update(cand=cand, inc=inc, split=split, n=len(fl), n_flat=int(np.sum(flat)),
             device_check='same' if all(v == 'same' for v in dev.values()) else
             'unknown on %d flights' % sum(v == 'unknown' for v in dev.values()),
             date=datetime.datetime.now().isoformat(timespec='seconds'), picker=mc.get('picker'), K=mc.get('K'))
    try:                                   # SUB3 info beside the verdict (never part of the rule): paired rep5, kpbench.rep5_paired
        from .rep5_paired import compare as rep5_compare
        g['rep5'] = rep5_compare(cand, inc, split, runs=runs)['pooled']
    except Exception as e:
        g['rep5'] = dict(error=repr(e)[:200])
    return g


def held_out_allowed(picker):
    E = log_entries()
    if any(e['split'] == 'held_out' and e.get('picker') == picker for e in E):
        return 'candidate %s already had its held-out verdict' % picker
    if not any(e['split'] == 'dev' and e.get('picker') == picker and e['flat_diff'][0] < 0 for e in E):
        return 'candidate %s has no logged DEV gate with flat difference < 0' % picker
    return None


def fmt(g):
    d, a = g['flat_diff'], g['all_diff']
    s = ['%s vs %s on %s (%d frames, %d flat; devices %s)' % (g['cand'], g['inc'], g['split'], g['n'], g['n_flat'], g['device_check']),
         '  flat fail      %.2f %% vs %.2f %%  (relative drop %.0f %%)' % (g['flat_cand'], g['flat_inc'], 100 * g['rel_drop']),
         '  A size         flat diff %+.2f [%+.2f, %+.2f] points -> %s' % (d[0], d[1], d[2], 'PASS' if g['route_A'] else 'no'),
         '  B consistency  flights better/worse/tied %d/%d/%d, p %.3f -> %s' % (
             g['flights_better'], g['flights_worse'], g['flights_tied'], g['sign_p'], 'PASS' if g['route_B'] else 'no'),
         '  guard rel>=10%% %s; guard all-frames hi %+.2f <= +0.2 %s' % (g['guard_rel'], a[2], g['guard_all']),
         '  per env flat   ' + ', '.join('%s %.1f vs %.1f' % (e, c, i) for e, (c, i) in g.get('per_env', {}).items()),
         *['  rep5 (info)    %s %.4f vs %.4f, diff %+.4f [%+.4f, %+.4f], flights better/worse/tied %d/%d/%d, p %.3f' % (
             k, r['cand'], r['inc'], *r['diff'], r['better'], r['worse'], r['tied'], r['sign_p'])
           for k, r in g.get('rep5', {}).items() if isinstance(r, dict) and 'diff' in r],
         '  VERDICT        %s' % ('KEEP candidate' if g['win'] else 'REVERT (incumbent stays)')]
    return '\n'.join(s)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True); ap.add_argument('--inc', required=True)
    ap.add_argument('--split', default='dev', choices=['dev', 'held_out', 'fresh']); ap.add_argument('--log', action='store_true')
    ap.add_argument('--force-held-out', default='', help='reason; allows a repeated or DEV-less held-out verdict (logged)')
    ap.add_argument('--allow-label-mismatch', default='', help='reason; gate despite a label guard failure (kept in the verdict)')
    ap.add_argument('--write-manifest', action='store_true', help='hash the current labels of all 31 flights (--split fresh: the FRESH flights) into the manifest and exit')
    a = ap.parse_args(argv)
    if a.write_manifest and a.split == 'fresh':
        if os.path.isfile(MANIFEST_FRESH):
            raise SystemExit('exists: %s (move it away first)' % MANIFEST_FRESH)
        json.dump({f: label_hash(f) for f in split_info('fresh')[0]}, open(MANIFEST_FRESH, 'w'), indent=1); print('wrote', MANIFEST_FRESH); return
    if a.write_manifest:
        from .data import FLIGHTS
        json.dump({f: label_hash(f) for f in FLIGHTS}, open(MANIFEST, 'w'), indent=1); print('wrote', MANIFEST); return
    if a.log and a.split == 'held_out':
        picker = json.load(open(os.path.join(RESULTS, 'runs', a.cand, 'meta.json'))).get('picker')
        why = held_out_allowed(picker)
        if why and not a.force_held_out:
            raise SystemExit('refused: %s (use --force-held-out "<reason>")' % why)
    g = compare(a.cand, a.inc, a.split, allow_mismatch=a.allow_label_mismatch)
    if a.force_held_out:
        g['forced'] = a.force_held_out
    print(fmt(g))
    os.makedirs(GATES, exist_ok=True)
    fn = os.path.join(GATES, '%s__vs__%s__%s.json' % (a.cand, a.inc, a.split))
    json.dump(g, open(fn, 'w'), indent=1, default=float)
    if a.log:
        open(LOG, 'a').write(json.dumps(g, default=float) + '\n')
    print('saved', os.path.relpath(fn, REPO))


if __name__ == '__main__':
    main()
