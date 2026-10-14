# kpbench: keypoint selection benchmark on TartanAir

One fixed task, one fixed protocol, every result stored and compared on one page.

**Task.** A drone flies through a TartanAir scene (old town, hospital, abandoned factory; 31 flights, about 22 000 frames).
Every 10th frame is a map keyframe. Every other frame is a query that must find its camera pose against the map built
from the keyframes within 20 frames (usually 4), using ground-truth depth and poses for the map. The thing under test is
the **picker**: which K of a detector's candidates are kept. Strongest-K is the baseline.

**Protocol (C23).** Candidates = the detector's strongest 2000 (RaCo ranker score or SuperPoint score), all with
SuperPoint descriptors. Matching = LightGlue (SuperPoint weights) per map keyframe, the headline setting (`lg`), or
mutual nearest neighbour (`nn`). Solver = `cv2.solvePnPRansac` with the AP3P minimal solver, 3 px, 2000 iterations,
at least 6 inliers, then Levenberg-Marquardt refinement on the inliers; position error = distance between estimated and
true camera centres. A query fails at a threshold pair when there is no pose, the position error is above T, or the
rotation error is above R: 25 cm / 2 deg (headline), 5 cm / 2 deg, 50 cm / 5 deg, 5 m / 10 deg. LightGlue output
depends slightly on the device (GPU model vs CPU), so pickers are compared only on runs made on the same device. Results are split into planarity bins (share of a frame's 3D points on one plane):
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

## Second dataset: Hypersim (preliminary)

A separate table at the bottom of the page (`docs/data/hypersim.json`, rendered by `docs/hypersim.js`); it never enters the
TartanAir numbers. Protocol: Apple Hypersim scenes ai_001_001 and ai_001_002, camera cam_00 (ai_003_001 dropped: its
renders are black in the released data); images 1024x768 resized to 640x480, fx = fy = 554.256, cx = 320, cy = 240;
planar depth from `depth_meters`. Frames are about 1.1 m apart, so a keyframe every 2nd frame and a query uses the
keyframes within +-3 frames (4 keyframes: q-3, q-1, q+1, q+3). RaCo, K = 100, same map, LightGlue, AP3P RANSAC + LM;
fail = no pose, > 25 cm or > 2 deg; 98 queries; laptop GPU. Preliminary: 2 scenes, wide intervals.

| run | fail % (98 q) | ai_001_001 | ai_001_002 | vs sd [95 % CI] |
|---|---|---|---|---|
| raco_dpp0.5_K100_same_lg | 6.1 | 10.2 | 2.0 | -4.1 [-9.2, +1.0] |
| raco_sd_K100_same_lg | 10.2 | 18.4 | 2.0 | baseline |
| raco_cube0.5_K100_same_lg | 13.3 | 18.4 | 8.2 | +3.1 [-3.0, +9.2] |

```
python tools/hypersim_fetch.py ai_001_001 ; python tools/hypersim_convert.py ai_001_001   # per scene: download + convert
FL=hypersim/ai_001_001,hypersim/ai_001_002 bash tools/run_hypersim.sh    # extract, labels, runs -> results_hypersim_w3/
python -m kpbench.export --hypersim                                      # -> docs/data/hypersim.json
PYTHONPATH=. KPBENCH_RESULTS=results_hypersim_w3 KPBENCH_INTR="554.256258 554.256258 320 240" python tools/hypersim_table.py
```
Data: Apple Hypersim, https://github.com/apple/ml-hypersim, CC BY-SA 3.0. M. Roberts, J. Ramapuram, A. Ranjan, A. Kumar,
M. A. Bautista, N. Paczan, R. Webb, J. M. Susskind, *Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor
Scene Understanding*, ICCV 2021.

## Frame viewer

`python -m kpbench.viewer --flight oldtown/P001 --frame 334 [--det raco] [--pickers sd,cube0.5,far2d,far3d] [--kmax 300]`
writes `docs/frames/oldtown_P001_334.html`: one frame's point cloud in 3D plus the image, with each picker's kept points
for any K (slider), optional 0.5 m cubes, and NN-distance / occupied-cube readouts. Standalone (plotly from the CDN).

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
