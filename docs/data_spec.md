# Site data contract (docs/data/)

Everything the page needs is static JSON and JPEG under `docs/data/`, written by `python -m kpbench.export`.
The page is plain HTML + JS, no build step. It must work when served from any sub-path (GitHub Pages
`https://<user>.github.io/kpbench/`) and from `python -m http.server` inside `docs/`. Use relative fetches (`data/...`).

## index.json
```json
{
  "generated": "2026-10-06T14:00:00",
  "dataset": "TartanAir (Hard), PnP relocalisation against a local map",
  "flights": [{"id": "oldtown/P001", "slug": "oldtown_P001", "env": "oldtown", "n_frames": 883, "n_queries": 794}],
  "bins": [
    {"id": "all", "label": "all frames"},
    {"id": "ordinary", "label": "ordinary (<0.55)", "lo": -1, "hi": 0.55},
    {"id": "mid", "label": "mid (0.55-0.85)", "lo": 0.55, "hi": 0.85},
    {"id": "flat", "label": "flat (>=0.85)", "lo": 0.85, "hi": 2},
    {"id": "very_flat", "label": "very flat (>=0.95)", "lo": 0.95, "hi": 2}
  ],
  "thresholds_cm": [5, 25, 50, 100],
  "rot_fail_deg": 5,
  "runs": [
    {"id": "raco_sd_K100_same_nn", "det": "raco", "picker": "sd", "K": 100, "map": "same", "matcher": "nn",
     "date": "2026-10-06T13:12:00", "commit": "a1b2c3d", "flights": ["oldtown/P001"], "n_queries": 794,
     "fail25": 12.5, "has_picks": true, "notes": ""}
  ]
}
```
Bins are on the per-query label `plane_frac` (share of the frame's 3D points on one plane, 30 cm tolerance).
A query belongs to a bin when `lo <= plane_frac < hi`. `all` has no bounds. `flat` and `very_flat` overlap on purpose.

## labels.json
Per flight, arrays aligned with the query list (queries = every frame that is not a keyframe; keyframes are every 10th frame).
```json
{"oldtown/P001": {"q": [1, 2, 3], "plane_frac": [0.91, 0.4, 0.7], "med_depth": [12.3, 8.0, 9.9],
                  "missing_depth": [0.0, 0.01, 0.0], "coverage": [0.97, 0.9, 0.99]}}
```
- `med_depth`: median depth (m) of the frame's strongest 300 candidates.
- `missing_depth`: share of the strongest 100 candidates without depth (sky).
- `coverage`: share of the query's pixels (with depth, 8 px grid) seen by at least one map keyframe within +-20 frames.
Values may be `null` when not computed.

## runs/<run_id>.json
```json
{"id": "raco_sd_K100_same_nn", "config": {"det": "raco", "picker": "sd", "K": 100, "map": "same", "matcher": "nn", "...": "..."},
 "cols": ["flight", "q", "err_cm", "rot_deg", "n_sel", "n_match", "n_inl", "rep1", "rep5", "rep10"],
 "rows": [[0, 1, 3.2, 0.4, 100, 61, 40, 0.8, 0.7, 0.6], [0, 2, null, null, 100, 12, 0, 0.5, 0.4, 0.3]]}
```
- `flight`: index into `index.json.flights` (NOT a string).
- `q`: frame number of the query.
- `err_cm`: camera position error in cm; `null` = solver gave no pose. `rot_deg`: rotation error; `null` with `err_cm`.
- A query FAILS at threshold T when `err_cm` is null, or `err_cm > T`, or `rot_deg > rot_fail_deg`.
- `n_sel`: number of points kept by the picker; `n_match`: mutual nearest-neighbour matches to the map; `n_inl`: RANSAC inliers (0 when no pose).
- `rep1/rep5/rep10`: repeatability of the kept points at +1/+5/+10 frames (share re-found within 3 px); may be `null`.
Runs can cover different flight subsets; compare runs only on the queries they share (same flight index and q).

## picks/<run_id>/<flight_slug>.json (optional)
```json
{"1": [[320.5, 100.2], [12.0, 400.1]], "2": [[...]]}
```
Pixel coordinates of the points kept at each query frame (640x480 image). Present only when `has_picks` is true. Fetch lazily; a 404 is fine.

## thumbs/<flight_slug>/<frame:06d>.jpg
160x120 JPEG of every frame (queries and keyframes). Original images are 640x480, so multiply pick coordinates by 0.25 to draw on a thumbnail.
Map keyframes of a query q are the keyframes k = 10*m with |k - q| <= 20.
