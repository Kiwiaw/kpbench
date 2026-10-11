"""Code search for pickers on MICRO (week-1 step 5): an LLM rewrites picker code, kpbench.evaluate scores it, an archive keeps
every program. ShinkaEvolve-style loop (fitness-weighted parent + inspirations, novelty by code hash), small enough to audit.

  python -m kpbench.evolve init  --run night1                 # score the seeds (kpbench/evolved/evo_seed_*.py)
  python -m kpbench.evolve next  --run night1                 # print the mutation brief; the child goes to the printed path
  python -m kpbench.evolve eval  --run night1 <child.py>      # score a child (subprocess, timeout) and add it to the archive
  python -m kpbench.evolve loop  --run night1 --n 100         # next + `claude -p` (Claude Code login) + eval, n times
  python -m kpbench.evolve status --run night1                # top programs
  python -m kpbench.evolve promote --run night1 <id>          # copy to kpbench/evolved/evo_<run>_<id>.py (gate-able picker)

Fitness = flat fail % on MICRO (393 flat queries of oldtown/P001 + hospital/P041, laptop GPU, LightGlue; deterministic on one
machine), lower is better; rep5 breaks ties. Rules are enforced by evaluate.checked_pick (K unique in-range integer indices, no
oracle). MICRO is 2 DEV flights, so a MICRO winner is only a candidate: promote it, then DEV gate, then the held-out exam.
"""
import os, re, sys, json, time, hashlib, subprocess, argparse, importlib.util
import numpy as np
from .cache import REPO

RESULTS = os.environ.get('KPBENCH_RESULTS', os.path.join(REPO, 'results'))
SEEDS = os.path.join(REPO, 'kpbench', 'evolved')
EVAL_TIMEOUT = 420

BRIEF = """You improve a keypoint picker for camera relocalisation. Write ONE complete Python file.

Task. Each frame has up to 2000 candidate keypoints from the RaCo detector, sorted by score (index 0 = strongest). The picker
returns exactly K = 100 indices (unique integers into the candidate arrays). The same picker runs on map keyframes and on
query frames; the query is matched to nearby keyframes with LightGlue and its pose solved by PnP-RANSAC. A query FAILS when
there is no pose, the position error is > 25 cm or the rotation error > 2 degrees. Fitness = fail % on 393 near-planar query
frames (>= 85 %% of the 3D points on one plane), lower is better; the incumbent cube picker scores about 22.6 %%.
Known so far: spreading points in 3D (cubes, DPP) beats strongest-K (34 %%); repeatability (the same physical points picked in
neighbouring frames) and strength also matter; on planes, points off the dominant plane and wide image coverage help pose.

Signature: def pick(u, v, s, z, K) -> np.ndarray of int
  u, v: pixel coordinates (float arrays, image 640 x 480); s: detector scores (descending); z: depth in metres per candidate
  (values >= kpbench.data.DEPTH_MAX = 1000 mean no depth: sky). Camera: fx = fy = 320, cx = 320, cy = 240;
  kpbench.data.to3d(u, v, z) gives camera-frame 3D points (n, 3).
Rules: numpy / scipy / standard library only; deterministic; no file, network or global state; must return exactly
min(K, len(u)) unique indices; under ~0.05 s per frame on a laptop CPU (the evaluation picks ~1500 frames). First line of
the file: '# idea: <one sentence: what changed vs the parent and why it should lower the fail %%>'.

Scoreboard (fail %% on the 393 frames, lower is better; rep5 = repeatability at lag 5, higher is better):
{board}

Recently tried children (do not repeat these ideas):
{recent}

Inspiration programs:
{insp}

PARENT program (fail {pfail:.2f} %%, rep5 {prep}) - modify it:
```python
{pcode}
```
Reply with only the new file in one ```python block."""


def rdir(run):
    d = os.path.join(RESULTS, 'evolve', run); os.makedirs(os.path.join(d, 'cand'), exist_ok=True); return d


def archive(run):
    f = os.path.join(rdir(run), 'archive.jsonl')
    return [json.loads(l) for l in open(f)] if os.path.isfile(f) else []


def add(run, e):
    open(os.path.join(rdir(run), 'archive.jsonl'), 'a').write(json.dumps(e) + '\n')


def code_hash(code):
    return hashlib.sha1(re.sub(r'\s+|#.*', '', code).encode()).hexdigest()[:12]


def scored(A):
    return sorted([e for e in A if e.get('fail_flat') is not None], key=lambda e: (e['fail_flat'], -(e.get('rep5') or 0)))


def eval_one(path, name):
    """child process: import pick() from path and score it on MICRO -> json line on stdout"""
    spec = importlib.util.spec_from_file_location('cand_' + name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from .evaluate import evaluate
    out, _, _ = evaluate(m.pick, split='micro', name=name)
    print('EVAL ' + json.dumps(out))


def do_eval(run, path, parent=None):
    code = open(path).read(); h = code_hash(code); A = archive(run)
    idea = (re.match(r'#\s*idea:\s*(.*)', code) or [None, ''])[1][:200]
    nid = 'c%03d' % len(A) if not os.path.basename(path).startswith('evo_seed') else os.path.basename(path)[4:-3]
    if parent is None:
        nx = os.path.join(rdir(run), 'next.json')
        parent = json.load(open(nx)).get('parent') if os.path.isfile(nx) else None
    e = dict(id=nid, parent=parent, hash=h, idea=idea, time=time.strftime('%Y-%m-%dT%H:%M:%S'))
    if any(x['hash'] == h for x in A):
        e['error'] = 'duplicate of %s' % next(x['id'] for x in A if x['hash'] == h)
    else:
        dst = os.path.join(rdir(run), 'cand', nid + '.py'); open(dst, 'w', newline='\n').write(code)
        try:
            r = subprocess.run([sys.executable, '-m', 'kpbench.evolve', 'eval_one', dst, nid], cwd=REPO, capture_output=True,
                               text=True, timeout=EVAL_TIMEOUT)
            line = [l for l in r.stdout.splitlines() if l.startswith('EVAL ')]
            if line:
                o = json.loads(line[-1][5:]); e.update(fail_flat=o['fail_flat'], fail_all=o['fail_all'], rep5=o['rep5'], secs=o['secs'])
            else:
                e['error'] = (r.stderr.strip().splitlines() or ['no output'])[-1][:300]
        except subprocess.TimeoutExpired:
            e['error'] = 'timeout > %d s' % EVAL_TIMEOUT
    add(run, e)
    print(json.dumps(e))
    return e


def pick_parent(S, rng):
    """power-law rank selection over the scored archive (rank 1 most likely), 20 % uniform for diversity"""
    if rng.random() < 0.2:
        return S[rng.integers(len(S))]
    w = 1.0 / (1 + np.arange(len(S))) ** 1.2
    return S[rng.choice(len(S), p=w / w.sum())]


def brief(run, seed=None):
    A = archive(run); S = scored(A)
    if not S:
        raise SystemExit('archive empty: run init first')
    rng = np.random.default_rng(seed if seed is not None else len(A)); par = pick_parent(S, rng)
    others = [e for e in S[:8] if e['id'] != par['id']]
    insp = [others[i] for i in sorted(rng.choice(len(others), min(2, len(others)), replace=False))] if others else []
    code = lambda e: open(os.path.join(rdir(run), 'cand', e['id'] + '.py')).read()
    board = '\n'.join('  %s fail %.2f rep5 %s  %s' % (e['id'], e['fail_flat'], e.get('rep5'), e.get('idea', '')[:90]) for e in S[:8])
    recent = '\n'.join('  %s: %s -> %s' % (e['id'], e.get('idea', '')[:110], ('fail %.2f' % e['fail_flat']) if e.get('fail_flat') is not None else e.get('error', '')[:60])
                       for e in A[-10:]) or '  none'
    ins = '\n'.join('--- %s (fail %.2f, rep5 %s)\n```python\n%s\n```' % (e['id'], e['fail_flat'], e.get('rep5'), code(e)) for e in insp) or 'none'
    txt = BRIEF.replace('%%', '%').format(board=board, recent=recent, insp=ins, pfail=par['fail_flat'], prep=par.get('rep5'), pcode=code(par))
    child = os.path.join(rdir(run), 'child.py')
    json.dump(dict(parent=par['id'], child=child), open(os.path.join(rdir(run), 'next.json'), 'w'))
    return txt, child, par['id']


def extract(text):
    m = re.findall(r'```python\s*\n(.*?)```', text, re.S)
    return m[-1] if m else None


def loop(run, n, model):
    for i in range(n):
        txt, child, par = brief(run)
        try:
            r = subprocess.run(['claude', '-p', '--model', model, '--tools', '', '--no-session-persistence', '--output-format', 'text',
                                '--system-prompt', 'You are an expert in multi-view geometry and Python. Output only the requested file.'],
                               input=txt, capture_output=True, text=True, timeout=600, encoding='utf-8')
        except subprocess.TimeoutExpired:
            print('llm timeout'); continue
        code = extract(r.stdout)
        if not code:
            print('no code from llm:', (r.stdout or r.stderr)[:200]);
            if 'authenticate' in (r.stdout + r.stderr): raise SystemExit('claude CLI not logged in: run `claude auth login`')
            continue
        open(child, 'w', newline='\n').write(code)
        do_eval(run, child, par)


def status(run, k=12):
    S = scored(archive(run)); A = archive(run)
    print('%d programs, %d scored, %d errors' % (len(A), len(S), sum(1 for e in A if e.get('error'))))
    for e in S[:k]:
        print('%-14s fail %6.2f  all %5.2f  rep5 %s  %5.0fs  parent %-8s %s' % (e['id'], e['fail_flat'], e['fail_all'], e.get('rep5'), e.get('secs') or 0, e.get('parent'), e.get('idea', '')[:70]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['init', 'next', 'eval', 'eval_one', 'loop', 'status', 'promote'])
    ap.add_argument('args', nargs='*'); ap.add_argument('--run', default='night1'); ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--model', default='sonnet')
    a = ap.parse_args(argv)
    if a.cmd == 'eval_one':
        return eval_one(a.args[0], a.args[1])
    if a.cmd == 'init':
        for f in sorted(os.listdir(SEEDS)):
            if f.startswith('evo_seed_') and f.endswith('.py') and not any(e['id'] == f[4:-3] for e in archive(a.run)):
                do_eval(a.run, os.path.join(SEEDS, f), None)
    elif a.cmd == 'next':
        txt, child, par = brief(a.run); print(txt); print('\n=== write the child file to: %s (parent %s), then: python -m kpbench.evolve eval --run %s %s' % (child, par, a.run, child))
    elif a.cmd == 'eval':
        do_eval(a.run, a.args[0])
    elif a.cmd == 'loop':
        loop(a.run, a.n, a.model)
    elif a.cmd == 'status':
        status(a.run)
    elif a.cmd == 'promote':
        src = os.path.join(rdir(a.run), 'cand', a.args[0] + '.py'); name = 'evo_%s_%s' % (a.run, a.args[0])
        open(os.path.join(SEEDS, name + '.py'), 'w', newline='\n').write(open(src).read()); print('picker', name)


if __name__ == '__main__':
    main()
