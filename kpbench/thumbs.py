"""160 x 120 JPEG thumbnails of every frame -> docs/data/thumbs/<flight_slug>/<frame>.jpg (used by the failure browser).

  python -m kpbench.thumbs --flights all [--out <dir>]
"""
import os
from PIL import Image
from .data import Flight, parse_flights
from .cache import REPO

OUT = os.path.join(REPO, 'docs', 'data', 'thumbs')


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--flights', default='all'); ap.add_argument('--out', default=OUT); a = ap.parse_args()
    for fid in parse_flights(a.flights):
        F = Flight(fid); d = os.path.join(a.out, F.slug); os.makedirs(d, exist_ok=True); n = 0
        for f in range(F.N):
            fn = os.path.join(d, '%06d.jpg' % f)
            if os.path.isfile(fn):
                continue
            Image.open(F.img_path(f)).convert('RGB').resize((160, 120), Image.BILINEAR).save(fn, quality=70, optimize=True); n += 1
        print(fid, 'thumbs written', n, 'of', F.N, flush=True)


if __name__ == '__main__':
    main()
