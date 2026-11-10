"""148 = 148 audit, Euler half as a laptop tool (SUB2 r29, dispatch 2026-11-10 13:25): score the pre-registered rules of
results/audit/r29_148_prep_2026-11-09.md once the sd K400 rows are mirrored.

  python tools/audit_148.py --run raco_sd_K400_same_lg_g-cube0.5-fresh-K400 [--ref results/audit/sd_K100_failset_2026-11-09.json]
                            [--k400 raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400] [--out results/audit/r29_148_verdict_<date>.md] [--no-verdict]

Reads --run with tools/failset_prep.flat_stats (fail ids on the 2075 FRESH-1 flat rows, n_sel per flat row, sha256 per flight file)
and compares with the stored sd K100 fail set (148 ids, 14 hashes, n_sel 100): Jaccard of the two fail sets, n_sel histogram, hash
collisions, then the verdict by the four rules (1 saturation, 2 row reuse, 3 recount error, 4 implausible identity -> re-run once).
--no-verdict prints the numbers only (self-check on a dir that is not the sd K400 arm).
"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from failset_prep import flat_stats
from kpbench.gate import split_info

RULES = {1: 'saturation: the sd K400 cell stands, tag [V]',
         2: 'row reuse: sd K400 void (57 vs 148, -4.39 [-7.01,-2.01] struck), cube0.5 K400 57 re-checked the same way',
         3: 'recount error: the two 148 are different frame sets, both sd cells frozen until the recount agrees',
         4: 'identical fail sets at n_sel 400 on distinct files: implausible, sd K400 arm re-run once (~1 GPU-h) before the cell is cited; [T] until then'}


def verdict(jac, nsel_le100_all, nsel_400_all, same_hash, same_nsel_rows):
    if jac < 1.0:
        return 3
    if same_hash:                      # identical row files decide reuse before saturation (both have n_sel <= 100 row by row)
        return 2
    if nsel_le100_all:
        return 1
    if nsel_400_all:
        return 4
    return 2 if same_nsel_rows else 4


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='raco_sd_K400_same_lg_g-cube0.5-fresh-K400')
    ap.add_argument('--ref', default=os.path.join(ROOT, 'results', 'audit', 'sd_K100_failset_2026-11-09.json'))
    ap.add_argument('--k400', default='raco_cube0.5_K400_same_lg_g-cube0.5-fresh-K400', help='mirrored K400 arm whose n_sel the sd arm should match (~400)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'audit', 'r29_148_verdict_%s.md' % time.strftime('%Y-%m-%d')))
    ap.add_argument('--no-verdict', action='store_true'); a = ap.parse_args()
    ref = json.load(open(a.ref)); sd = ref['sd_K100']; flights = split_info('fresh')[0]
    d = os.path.join(ROOT, 'results', 'runs', a.run)
    if not os.path.isdir(d):
        sys.exit('run dir not mirrored: %s (euler/pull_runs.sh --keep-old-files once ssh answers)' % a.run)
    t = flat_stats(a.run, flights)
    A, B = set(sd['fail_ids']), set(t['fail_ids']); jac = len(A & B) / max(len(A | B), 1)
    ns = t['n_sel_flat']; hist = ns['hist'] if ns else {}
    le100_all = bool(ns and ns['max'] <= 100); n400_all = bool(ns and ns['min'] >= 400 and ns['max'] <= 400)
    same_hash = [k for k, v in t['files'].items() if k in sd['files'] and v['sha256'] == sd['files'][k]['sha256']]
    same_nsel_rows = bool(ns and sd['n_sel_flat'] and hist == sd['n_sel_flat']['hist'])
    k4 = flat_stats(a.k400, flights) if a.k400 and os.path.isdir(os.path.join(ROOT, 'results', 'runs', a.k400)) else None
    L = ['# 148 = 148 audit: %s vs the stored sd K100 fail set (%s)' % (a.run, time.strftime('%Y-%m-%d %H:%M')), '',
         'Reference: %s (run %s, %d flat fails of %d, n_sel %s). Rules: results/audit/r29_148_prep_2026-11-09.md.' % (
             os.path.relpath(a.ref, ROOT).replace(os.sep, '/'), sd['run'], sd['flat_fails'], sd['flat_rows'], sd['n_sel_flat']['hist'] if sd['n_sel_flat'] else 'none'), '',
         '| quantity | value |', '|---|---|',
         '| run meta | K %s, picker %s, date %s, device %s, has_picks %s |' % (t['K'], t['picker'], t['date'], t['device'], t['has_picks']),
         '| rows / flat rows / flat fails | %d / %d / %d |' % (t['rows'], t['flat_rows'], t['flat_fails']),
         '| Jaccard(fail sets) | %.4f (common %d, only K100 %d, only this run %d) |' % (jac, len(A & B), len(A - B), len(B - A)),
         '| n_sel histogram on flat rows | %s |' % (json.dumps(hist) if hist else 'no n_sel column'),
         '| n_sel <= 100 on every flat row | %s |' % le100_all, '| n_sel == 400 on every flat row | %s |' % n400_all,
         '| flight files with a sha256 identical to the sd K100 file | %d of %d%s |' % (len(same_hash), len(t['files']), (': ' + ', '.join(same_hash)) if same_hash else ''),
         '| n_sel histogram identical to sd K100 | %s |' % same_nsel_rows]
    if k4:
        L.append('| reference K400 arm %s | flat fails %d, n_sel %s |' % (a.k400, k4['flat_fails'], json.dumps(k4['n_sel_flat']['hist']) if k4['n_sel_flat'] else 'none'))
    if a.no_verdict:
        L += ['', 'Numbers only (--no-verdict): this run is not the sd K400 arm.']
    else:
        r = verdict(jac, le100_all, n400_all, bool(same_hash), same_nsel_rows)
        L += ['', '## Verdict', '', '- rule %d fires: %s' % (r, RULES[r])]
    os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n'); print('\n'.join(L)); print('->', a.out)


if __name__ == '__main__':
    main()
