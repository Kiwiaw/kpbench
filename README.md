# kpbench: keypoint selection benchmark on TartanAir

One fixed task, one fixed protocol, every result stored and compared on one page.

**Task.** A drone flies through a TartanAir scene (old town, hospital, abandoned factory; 31 flights, about 22 000 frames).
Every 10th frame is a map keyframe. Every other frame is a query that must find its camera pose against the map built
from the keyframes within 20 frames (usually 4), using ground-truth depth and poses for the map. The thing under test is
the **picker**: which K of a detector's candidates are kept. Strongest-K is the baseline.

**Protocol (C23).** Candidates = the detector's strongest 2000 (RaCo ranker score or SuperPoint score), all with
SuperPoint descriptors. Matching = mutual nearest neighbour. Solver = `cv2.solvePnPRansac`, 3 px, 2000 iterations,
at least 6 inliers. A query fails at threshold T when there is no pose, the position error is above T cm, or the
rotation error is above 5 degrees. Results are split into planarity bins (share of a frame's 3D points on one plane):
ordinary < 0.55, mid 0.55-0.85, flat >= 0.85, very flat >= 0.95. Each query also carries median depth, share of
candidates without depth, and map coverage (share of its pixels that some map keyframe sees).

## Page

`docs/` is a static page (GitHub Pages: https://kiwiaw.github.io/kpbench/). It loads every exported run and shows
fail rates per bin with bootstrap confidence intervals, per-flight tables, paired comparisons against a baseline run
(saves / breaks, exact binomial p), error curves, and a failure browser that lists every failed query with its
thumbnail, kept points and map keyframes. Open it locally with `cd docs && python -m http.server 8765`.

## Pipeline

```
python -m kpbench.extract --flights oldtown/P001 --dets raco,sp   # 1. candidates -> cache/ (GPU or CPU)
python -m kpbench.labels  --flights oldtown/P001                   # 2. planarity, depth, coverage -> results/labels/
python -m kpbench.thumbs  --flights oldtown/P001                   # 3. thumbnails -> docs/data/thumbs/
python -m kpbench.run --det raco --picker sd --K 100 --map same --flights all --save-picks   # 4. -> results/runs/<id>/
python -m kpbench.export                                           # 5. -> docs/data/ (index, labels, runs, picks)
```
Run id = `<det>_<picker>_K<K>_<map>_nn[_<tag>]`. Re-running a configuration skips flights already stored
(`--force` to redo). Pickers: `score`, `sd`, `cube<metres>`, `grid`, `grid10x8`, `fps`, `fps0`; add one in
`kpbench/pickers.py`. Map modes: `same` (keyframes keep the same picker's K points) and `dense` (keyframes keep
their strongest 300).

Environment variables: `KPBENCH_DATA` (data root; laptop `local_data`, Euler `/cluster/scratch/<user>`),
`KPBENCH_CACHE`, `KPBENCH_RESULTS`, `RACO_DIR`, `DEV`.

## Euler (all 31 flights)

```
bash euler/push.sh                       # code -> ~/kpbench
ssh euler 'cd kpbench && sbatch euler/extract.sbatch'                # candidates, labels, thumbnails (array over flights)
ssh euler 'cd kpbench && RUNS="raco:sd:100:same raco:grid:100:same" sbatch euler/run.sbatch'
bash euler/pull.sh                       # results + thumbnails -> laptop, then export
git add -A && git commit -m "..." && git push
```

## Checks

`python -m pytest tests` runs the data-free tests (pickers, plane fit, PnP on a synthetic scene). The pipeline
reproduces the 30 Sep 2026 reference numbers on old town P001 exactly (RaCo strongest-with-depth K=100, same map:
11.59 % fail at 25 cm). The solver is deterministic, so a re-run of a stored configuration gives identical rows.
