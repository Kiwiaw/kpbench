"""Fake dataset for testing the results page locally.

Writes index.json, labels.json, runs/*.json, picks/<run>/<slug>.json and thumbs/<slug>/<frame>.jpg
into docs/data/ following docs/data_spec.md. The real export (python -m kpbench.export)
overwrites this folder later.

Usage (from the repo root or anywhere):  python docs/dev/make_fake.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "data"
rng = np.random.default_rng(0)

# 334 frames -> keyframes 0,10,...,330 (34 of them) -> 300 queries per flight.
N_FRAMES = 334
FLIGHTS = [
    {"id": "oldtown/P001", "slug": "oldtown_P001", "env": "oldtown"},
    {"id": "abandonedfactory/P002", "slug": "abandonedfactory_P002", "env": "abandonedfactory"},
]
# skill: lower = fewer failures. The third run covers only the first flight (tests the subset warning).
RUNS = [
    dict(id="raco_sd_K100_same_nn", det="raco", picker="sd", K=100, map="same", matcher="nn",
         flights=[0, 1], skill=0.0, has_picks=True, date="2026-10-06T13:12:00", commit="a1b2c3d"),
    dict(id="raco_topk_K100_same_nn", det="raco", picker="topk", K=100, map="same", matcher="nn",
         flights=[0, 1], skill=0.35, has_picks=False, date="2026-10-06T13:40:00", commit="a1b2c3d"),
    dict(id="sp_sd_K200_same_nn", det="sp", picker="sd", K=200, map="same", matcher="nn",
         flights=[0], skill=-0.2, has_picks=False, date="2026-10-05T22:05:00", commit="9f8e7d6"),
]
BINS = [
    {"id": "all", "label": "all frames"},
    {"id": "ordinary", "label": "ordinary (<0.55)", "lo": -1, "hi": 0.55},
    {"id": "mid", "label": "mid (0.55-0.85)", "lo": 0.55, "hi": 0.85},
    {"id": "flat", "label": "flat (>=0.85)", "lo": 0.85, "hi": 2},
    {"id": "very_flat", "label": "very flat (>=0.95)", "lo": 0.95, "hi": 2},
]
ROT_FAIL = 5


def r(x, d=2):
    """Round for compact JSON; keeps None."""
    return None if x is None else round(float(x), d)


def make_thumb(path, slug, frame, seed):
    """Grey 160x120 image with a few shapes that drift with the frame number, and the frame number drawn."""
    g = np.random.default_rng(seed)
    img = Image.new("L", (160, 120), 110)
    d = ImageDraw.Draw(img)
    for _ in range(6):
        x0, y0 = g.integers(-40, 160), g.integers(0, 100)
        w, h = g.integers(20, 60), g.integers(10, 40)
        dx = int((frame * g.uniform(0.2, 0.8)) % 200) - 40
        d.rectangle([x0 + dx, y0, x0 + dx + w, y0 + h], fill=int(g.integers(40, 220)))
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    d.rectangle([0, 0, 160, 22], fill=30)
    d.text((4, 2), f"{frame:06d}", fill=255, font=font)
    img.save(path, "JPEG", quality=80)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "runs").mkdir(parents=True)

    # Per-flight labels plus a hidden "difficulty" per query shared by all runs (makes runs correlated).
    labels, hidden = {}, []
    for fl in FLIGHTS:
        q = [f for f in range(N_FRAMES) if f % 10 != 0]
        n = len(q)
        flat = rng.random(n) < 0.35
        pf = np.where(flat, rng.uniform(0.85, 1.0, n), rng.beta(2, 3, n))
        cov = np.clip(rng.normal(0.9, 0.08, n), 0.3, 1.0)
        labels[fl["id"]] = {
            "q": q,
            "plane_frac": [r(v, 3) for v in pf],
            "med_depth": [r(v) for v in rng.uniform(3, 30, n)],
            "missing_depth": [r(v, 3) for v in rng.beta(1, 12, n)],
            # a few nulls to exercise the null handling
            "coverage": [None if rng.random() < 0.03 else r(v, 3) for v in cov],
        }
        hidden.append(rng.normal(0, 1, n) + 2.2 * pf + 2.0 * (1 - cov))
        fl["n_frames"] = N_FRAMES
        fl["n_queries"] = n

    index_runs = []
    for run in RUNS:
        rows = []
        picks = {fi: {} for fi in run["flights"]}
        for fi in run["flights"]:
            q = labels[FLIGHTS[fi]["id"]]["q"]
            for j, qq in enumerate(q):
                z = hidden[fi][j] + run["skill"] + rng.normal(0, 0.6)
                n_sel = run["K"]
                n_match = int(np.clip(0.7 * n_sel - 12 * z + rng.normal(0, 6), 0, n_sel))
                if z > 3.0 or n_match < 8:
                    err = rot = None
                    n_inl = 0
                else:
                    err = float(np.exp(0.6 + 1.0 * z + rng.normal(0, 0.5)))
                    rot = err / 12 * rng.uniform(0.5, 1.5)
                    if rng.random() < 0.02:
                        rot = rng.uniform(5, 30)  # occasional rotation-only failure
                    n_inl = int(n_match * rng.uniform(0.3, 0.8))
                base_rep = float(np.clip(0.85 - 0.08 * z, 0.05, 1))
                rep1 = base_rep
                rep5 = base_rep * 0.85
                rep10 = None if qq + 10 >= N_FRAMES else base_rep * 0.7
                rows.append([fi, qq, r(err), r(rot), n_sel, n_match, n_inl, r(rep1, 3), r(rep5, 3), r(rep10, 3)])
                if run["has_picks"]:
                    pts = np.column_stack([rng.uniform(0, 640, n_sel), rng.uniform(0, 480, n_sel)])
                    picks[fi][str(qq)] = [[r(x, 1), r(y, 1)] for x, y in pts]
        cfg = {k: run[k] for k in ("det", "picker", "K", "map", "matcher")}
        cfg.update({"ransac_px": 4.0, "ransac_iters": 2000, "nms_px": 4, "depth_max_m": 80, "seed": 0})
        (OUT / "runs" / f"{run['id']}.json").write_text(json.dumps({
            "id": run["id"], "config": cfg,
            "cols": ["flight", "q", "err_cm", "rot_deg", "n_sel", "n_match", "n_inl", "rep1", "rep5", "rep10"],
            "rows": rows}, separators=(",", ":")))
        if run["has_picks"]:
            for fi, pk in picks.items():
                p = OUT / "picks" / run["id"]
                p.mkdir(parents=True, exist_ok=True)
                (p / f"{FLIGHTS[fi]['slug']}.json").write_text(json.dumps(pk, separators=(",", ":")))
        n_fail25 = sum(1 for row in rows if row[2] is None or row[2] > 25 or (row[3] or 0) > ROT_FAIL)
        index_runs.append({
            **cfg, "id": run["id"], "date": run["date"], "commit": run["commit"],
            "flights": [FLIGHTS[i]["id"] for i in run["flights"]], "n_queries": len(rows),
            "fail25": r(100 * n_fail25 / len(rows), 1), "has_picks": run["has_picks"], "notes": "fake data",
        })
        for k in ("ransac_px", "ransac_iters", "nms_px", "depth_max_m", "seed"):
            index_runs[-1].pop(k)

    (OUT / "index.json").write_text(json.dumps({
        "generated": "2026-10-06T14:00:00",
        "dataset": "TartanAir (Hard), PnP relocalisation against a local map [FAKE TEST DATA]",
        "flights": FLIGHTS, "bins": BINS, "thresholds_cm": [5, 25, 50, 100], "rot_fail_deg": ROT_FAIL,
        "runs": index_runs}, indent=1))
    (OUT / "labels.json").write_text(json.dumps(labels, separators=(",", ":")))

    for fi, fl in enumerate(FLIGHTS):
        d = OUT / "thumbs" / fl["slug"]
        d.mkdir(parents=True)
        for f in range(N_FRAMES):
            make_thumb(d / f"{f:06d}.jpg", fl["slug"], f, seed=fi * 1000 + 7)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
