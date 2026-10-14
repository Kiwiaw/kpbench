"""Fetch only the files kpbench needs from a Hypersim scene zip (HTTP range requests, no full download).

  python tools/hypersim_fetch.py ai_001_001 [--cam cam_00] [--out local_data/hypersim_raw] [--list]

Per scene and camera: final_preview tonemap jpgs, geometry depth_meters hdf5, camera keyframe positions + orientations.
Source: Apple's Hypersim v1 scene zips (https://github.com/apple/ml-hypersim, CC BY-SA 3.0).
"""
import argparse, io, os, urllib.request, zipfile

URL = 'https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/%s.zip'


class RangeFile(io.RawIOBase):
    """read-only, seekable file over HTTP range requests"""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers['Content-Length'])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b''
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={'Range': 'bytes=%d-%d' % (self.pos, end)})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


def wanted(name, cam):
    return (('/images/scene_%s_final_preview/' % cam in name and name.endswith('.tonemap.jpg'))
            or ('/images/scene_%s_geometry_hdf5/' % cam in name and name.endswith('.depth_meters.hdf5'))
            or ('/_detail/%s/camera_keyframe_' % cam in name and name.endswith('.hdf5'))
            or name.endswith('/_detail/metadata_scene.csv'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('scene'); ap.add_argument('--cam', default='cam_00')
    ap.add_argument('--out', default=os.path.join('..', 'local_data', 'hypersim_raw')); ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    f = RangeFile(URL % a.scene)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    names = [n for n in z.namelist() if wanted(n, a.cam)]
    print('%s: zip %.1f GB, %d files wanted (%.0f MB)' % (a.scene, f.size / 1e9, len(names),
          sum(z.getinfo(n).compress_size for n in names) / 1e6), flush=True)
    if a.list:
        cams = sorted({n.split('/')[-2] for n in z.namelist() if '_final_preview/' in n})
        print('cameras:', cams); raise SystemExit
    for i, n in enumerate(names):
        p = os.path.join(a.out, n)
        if os.path.isfile(p) and os.path.getsize(p) == z.getinfo(n).file_size:
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as o:
            o.write(z.read(n))
        if i % 25 == 0:
            print(i, len(names), n, flush=True)
    print('done', a.scene)
