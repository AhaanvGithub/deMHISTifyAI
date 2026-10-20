#!/usr/bin/env python3
"""Download only UniToPatho's hyperplastic-polyp patches (800 um, 1812 px at 0.4415 um/px) from the 275 GB
UNITOPatho.zip, without downloading the whole archive.

A zip's file list sits at its end, so with HTTP range requests we read that list, then fetch just the wanted
entries. Needs a direct download URL that supports range requests (e.g. the signed link IEEE DataPort gives a
logged-in user). Usage:

    python research/fetch_unitopatho_hp.py "<direct download URL>" [--list] [--match 800/HP] [--out data/unitopatho]
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


class HttpFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (just enough for zipfile)."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                print("warning: server did not advertise range support; trying anyway", file=sys.stderr)
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            if r.status != 206:
                raise RuntimeError(f"server ignored the range request (HTTP {r.status}); cannot fetch partially")
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--match", default="800/HP", help="substring an entry's path must contain")
    ap.add_argument("--list", action="store_true", help="only list the archive's top-level folders and matches")
    ap.add_argument("--out", default="data/unitopatho")
    a = ap.parse_args()

    f = HttpFile(a.url)
    print(f"archive size: {f.size / 1e9:.1f} GB")
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    names = z.namelist()
    tops = sorted({"/".join(n.split("/")[:3]) for n in names if n.count("/") >= 2})[:40]
    want = [i for i in z.infolist() if a.match in i.filename and not i.is_dir()]
    print(f"{len(names)} entries; {len(want)} match '{a.match}' ({sum(i.file_size for i in want) / 1e9:.2f} GB)")
    if a.list:
        print("folders:", *tops, sep="\n  ")
        print("first matches:", *[i.filename for i in want[:10]], sep="\n  ")
        return
    out = Path(a.out)
    for k, info in enumerate(want, 1):
        dest = out / info.filename
        if dest.exists() and dest.stat().st_size == info.file_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(info))
        if k % 25 == 0 or k == len(want):
            print(f"  {k}/{len(want)}", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
