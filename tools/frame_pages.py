"""Build the multi-dataset frame pages (kpbench.viewer, 7 pickers, K 0..400 of 3000 candidates) and docs/frames/index.html.

  python tools/frame_pages.py --root <folder with the tools/frame_export.py outputs>    # all pages + index
  python tools/frame_pages.py --root ... --only spring_0010                            # one page + index
  python tools/frame_pages.py --index                                                  # index only

Frame folders are made on Euler with tools/frame_export.py (frame f and its 3 predecessors, for dppmr's track bonus) and
copied here; detectors run locally (GPU if present). Each page runs in its own process because the camera intrinsics are
fixed per process (KPBENCH_INTR, read from the folder's intrinsics.txt).
"""
import argparse, html, os, subprocess, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(REPO, 'docs', 'frames')
PICKERS = 'dpp0.5,dppmr0.256a0.5b1.02t0.9,cube0.5,cubem0.1,raco:sd,sp:sd,aliked:sd'
# folder, dataset, scene (as shown), frame, where it comes from
PAGES = [
    ('tartanair_hospital_P042', 'TartanAir', 'hospital P042', 163, 'kpbench DEV flight'),
    ('tartanair_factory_P000', 'TartanAir', 'abandoned factory P000', 363, 'kpbench DEV flight'),
    ('tartanair_japanesealley_P001', 'TartanAir', 'Japanese alley Easy P001', 1003, 'TartanAir, outside the kpbench flights'),
    ('tartanair_office_P001', 'TartanAir', 'office Easy P001', 1323, 'TartanAir, outside the kpbench flights'),
    ('tartanair_seasidetown_P000', 'TartanAir', 'seaside town Easy P000', 403, 'TartanAir, outside the kpbench flights'),
    ('hypersim_ai_001_005', 'Hypersim', 'ai_001_005', 78, 'Hypersim, cam_00'),
    ('hypersim_ai_002_003', 'Hypersim', 'ai_002_003', 53, 'Hypersim, cam_00'),
    ('tum_fr3_long_office_household', 'TUM RGB-D', 'fr3 long office household', 353, 'TUM RGB-D (kpbench conversion, every 3rd frame)'),
    ('spring_0010', 'Spring', '0010', 78, 'Spring, left camera, centre crop 4:3'),
    ('pointodyssey_ani', 'Point Odyssey', 'ani (train)', 853, 'Point Odyssey, centre crop 4:3'),
]
# earlier pages, listed in the index too: file, dataset, scene, frame, what it shows
OLD = [
    ('oldtown_P001_334.html', 'TartanAir', 'old town P001', 334, 'strongest, cubes and farthest-point pickers, K 1..2000'),
    ('hospital_P041_200.html', 'TartanAir', 'hospital P041', 200, 'strongest, cubes and farthest-point pickers, K 1..2000'),
    ('dpp_tartanair_oldtown_P001_334.html', 'TartanAir', 'old town P001', 334, 'DPP vs strongest vs cubes, K 1..2000 of 3000 RaCo points'),
    ('dpp_hypersim_ai_001_001_87.html', 'Hypersim', 'ai_001_001', 87, 'DPP vs strongest vs cubes, K 1..2000 of 3000 RaCo points'),
]
NEW_WHAT = '7 pickers (DPP, scaled DPP, cubes, scaled cubes, RaCo / SuperPoint / ALIKED strongest), K 0..400 of 3000 points'
NOTE = ("Each picker puts its detector's 3000 strongest points with known depth in an order; the slider shows the first k. "
        "The first four use RaCo's points. DPP 0.5 m prefers strong points but avoids points within about half a metre (in 3D) "
        "of a point it already picked. The scaled DPP sets that spacing to 0.256 times the frame's median depth and favours "
        "points that were found again in the previous 3 frames. Cubes take the strongest point of each cube in turn, cubes of "
        "0.5 m or of 0.1 times the median depth. The last three simply take the strongest points of RaCo, SuperPoint and ALIKED. "
        "Picked points are coloured, unpicked candidates are faint.")


def page_name(folder, frame):
    return 'dpp_%s_%d.html' % (folder, frame)


def build(root, folder, dataset, scene, frame, src):
    d = os.path.join(root, folder)
    fx, fy, cx, cy = np.loadtxt(os.path.join(d, 'intrinsics.txt'))
    env = dict(os.environ, KPBENCH_INTR='%.6f %.6f %.6f %.6f' % (fx, fy, cx, cy))
    title = 'Keypoint pickers, %s %s frame %d (RaCo, SuperPoint, ALIKED; 0\u2013400 of 3000 points)' % (dataset, scene, frame)
    cmd = [sys.executable, '-m', 'kpbench.viewer', '--dir', d, '--frame', str(frame), '--ncand', '3000', '--kmin', '0',
           '--kmax', '400', '--pickers', PICKERS, '--default', 'dpp0.5', '--out', page_name(folder, frame),
           '--title', title, '--note', NOTE + ' Source: %s.' % src]
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def index():
    rows = [(f, ds, sc, fr, what) for f, ds, sc, fr, what in OLD]
    rows += [(page_name(folder, fr), ds, sc, fr, NEW_WHAT) for folder, ds, sc, fr, _ in PAGES]
    rows = [r for r in rows if os.path.isfile(os.path.join(FRAMES, r[0]))]
    tr = '\n'.join('<tr><td>%s</td><td>%s</td><td>%d</td><td><a href="%s">%s</a></td></tr>'
                   % (html.escape(ds), html.escape(sc), fr, f, html.escape(what)) for f, ds, sc, fr, what in rows)
    page = INDEX.replace('__ROWS__', tr)
    with open(os.path.join(FRAMES, 'index.html'), 'w', encoding='utf-8', newline='\r\n') as fh:
        fh.write(page)
    print('index: %d pages' % len(rows))


INDEX = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frame pages</title>
<style>
  :root { --bg: #fafafa; --fg: #222; --muted: #666; --line: #ddd; --link: #0b62c4; }
  @media (prefers-color-scheme: dark) { :root { --bg: #16181b; --fg: #e6e6e6; --muted: #9a9a9a; --line: #333; --link: #6fb0ff; } }
  body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg);
         font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  p { color: var(--muted); max-width: 900px; margin: 0 0 12px; }
  .wrap { overflow-x: auto; }
  table { border-collapse: collapse; min-width: 560px; }
  th, td { text-align: left; padding: 5px 12px 5px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { font-weight: 600; }
  a { color: var(--link); }
</style>
</head>
<body>
<h1>Frame pages</h1>
<p>One frame per page: the frame in 3D and as an image, with the keypoints each picker keeps as you move the K slider.
Back to the <a href="../index.html">kpbench page</a>.</p>
<div class="wrap"><table>
<tr><th>dataset</th><th>scene</th><th>frame</th><th>page</th></tr>
__ROWS__
</table></div>
</body>
</html>
'''


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=''); ap.add_argument('--only', default=''); ap.add_argument('--index', action='store_true')
    a = ap.parse_args()
    if not a.index:
        for p in PAGES:
            if not a.only or p[0] in a.only.split(','):
                build(a.root, *p)
    index()
