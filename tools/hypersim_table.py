"""Hypersim table: fail % (25 cm / 2 deg), matches, inliers, rep, paired block-bootstrap diff vs the sd baseline."""
import gzip, json, os, sys
import numpy as np
from kpbench.stats import fail_flags, paired_diff, block_ids
RES = os.environ.get('KPBENCH_RESULTS', 'results_hypersim')
FL = ['hypersim_ai_001_001', 'hypersim_ai_001_002']
runs = sys.argv[1:] or ['raco_dpp0.5_K100_same_lg', 'raco_sd_K100_same_lg', 'raco_cube0.5_K100_same_lg']
def load(r):
    rows = []
    for f in FL:
        rows += [[f] + x[1:] for x in json.load(gzip.open(os.path.join(RES, 'runs', r, f + '.json.gz')))['rows']]
    return rows
base = load('raco_sd_K100_same_lg'); fb = fail_flags(base); blocks = block_ids([r[0] for r in base], [r[1] for r in base])
for r in runs:
    R = load(r); fa = fail_flags(R); assert [x[1] for x in R] == [x[1] for x in base]
    per = '  '.join('%s %.1f' % (f[-3:], 100 * fa[np.array([x[0] == f for x in R])].mean()) for f in FL)
    num = lambda c: np.nanmean([np.nan if x[c] is None else x[c] for x in R])
    d = paired_diff(fa, fb, blocks) if r != 'raco_sd_K100_same_lg' else (0, 0, 0)
    print('%-28s n %3d  fail %5.1f  (%s)  vs sd %+5.1f [%+5.1f, %+5.1f]  match %5.1f  inl %5.1f  rep1 %.2f rep5 %.2f' % (
        r, len(R), 100 * fa.mean(), per, d[0], d[1], d[2], num(5), num(6), num(7), num(8)))
