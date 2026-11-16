"""C42 (SUB2): kpbench.pickstats.compare end to end on synthetic --save-picks run files with a known answer (no flight data).

Two runs on 2 flights x 10 queries, queries 0-5 flat (plane_frac 0.9), 6-9 not flat (0.5).
  cand: fails on q 0 only;          off-plane share 0.30, depths 10 x [1, 2, 3] -> dstd 8.165, dspan 1.6 / 2 = 0.8
  inc : fails on q 0, 1, 2;         off-plane share 0.10, depths 10 x [1, 1, 1] -> dstd 0, dspan 0
Known answer: n 20, n_flat 12, rescued = flat queries cand posed, inc failed = q 1, 2 per flight = 4;
paired off diff +0.2 on every query (median 0.2, up = n, down 0), kill False (median > 0).
Second case: cand off 0.05 -> rescued median -0.05 -> kill True. Third case: run without picks is refused.
"""
import gzip, json, os
import pytest
from kpbench import pickstats, gate

FLIGHTS = ['envA/P000', 'envB/P001']
FLAT_Q = range(6)


def write_run(root, rid, fail_q, off, depths, picks=True):
    d = os.path.join(root, 'runs', rid); os.makedirs(d, exist_ok=True)
    json.dump(dict(has_picks=picks), open(os.path.join(d, 'meta.json'), 'w'))
    for f in FLIGHTS:
        rows = [[f, q, 300.0 if q in fail_q else 5.0, 0.5] for q in range(10)]
        j = dict(rows=rows, picks={str(q): [1, 2, 3] for q in range(10)} if picks else {},
                 onplane={str(q): [1 - off, 3, 0.9, [1, 1, 0], depths] for q in range(10)})
        with gzip.open(os.path.join(d, pickstats.slug(f) + '.json.gz'), 'wt') as fh:
            json.dump(j, fh)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'RESULTS', str(tmp_path)); monkeypatch.setattr(pickstats, 'RESULTS', str(tmp_path))
    monkeypatch.setattr(pickstats, 'split_info', lambda split: (FLIGHTS, None, None))
    lab = {str(q): dict(plane_frac=0.9 if q in FLAT_Q else 0.5) for q in range(10)}
    monkeypatch.setattr(pickstats, 'labels', lambda f: lab)
    write_run(str(tmp_path), 'inc', {0, 1, 2}, 0.10, [10.0, 10.0, 10.0])
    return tmp_path


def test_compare_known_answer(fake):
    write_run(str(fake), 'cand', {0}, 0.30, [10.0, 20.0, 30.0])
    r = pickstats.compare('cand', 'inc', 'fresh')
    assert (r['n'], r['n_flat'], r['n_rescued']) == (20, 12, 4)
    assert r['flat']['off'] == dict(n=12, median=0.2, mean=0.2, up=12, down=0)
    assert r['rescued']['off'] == dict(n=4, median=0.2, mean=0.2, up=4, down=0)
    assert r['rescued']['dstd']['median'] == 8.165 and r['rescued']['dspan']['median'] == 0.8
    assert r['kill'] is False


def test_compare_kill_when_no_offplane_gain(fake):
    write_run(str(fake), 'cand', {0}, 0.05, [10.0, 10.0, 10.0])
    r = pickstats.compare('cand', 'inc', 'fresh')
    assert r['rescued']['off']['median'] == -0.05 and r['rescued']['off']['down'] == 4 and r['kill'] is True


def test_run_without_picks_refused(fake):
    write_run(str(fake), 'cand', {0}, 0.30, [10.0, 20.0, 30.0], picks=False)
    with pytest.raises(gate.GateError):
        pickstats.compare('cand', 'inc', 'fresh')
