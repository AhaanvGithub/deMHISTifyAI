#!/usr/bin/env python3
"""Download UniToPatho's 800-um hyperplastic-polyp patches (plus the label CSVs) from IEEE DataPort's S3 bucket.

Credentials are read from the environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN if given)
and are never printed or saved. Get them from the dataset page: "Access on AWS" -> "View AWS Security Credentials".

    python research/fetch_unitopatho_s3.py --list      # show how many files and how big, download nothing
    python research/fetch_unitopatho_s3.py             # download into data/unitopatho/800/
"""

import argparse
import os
from pathlib import Path

import boto3

BUCKET = "ieee-dataport"
PREFIX = "open/49617/3592/UNITOPatho_zip/unitopath-public/800/"
ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--classes", nargs="+", default=["HP"])
    a = ap.parse_args()
    missing = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"set {', '.join(missing)} in this terminal first (from 'View AWS Security Credentials')")
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    keys = []
    for pre in [PREFIX + c + "/" for c in a.classes] + [PREFIX + "train.csv", PREFIX + "test.csv"]:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=pre):
            keys += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    total = sum(s for _, s in keys)
    print(f"{len(keys)} files, {total / 1e9:.2f} GB")
    if a.list:
        for k, s in keys[:5]:
            print(f"  {k[len(PREFIX):]}  {s / 1e6:.1f} MB")
        return
    out = ROOT / "data/unitopatho/800"
    for i, (k, s) in enumerate(keys, 1):
        dest = out / k[len(PREFIX):]
        if dest.exists() and dest.stat().st_size == s:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, k, str(dest))
        if i % 25 == 0 or i == len(keys):
            print(f"  {i}/{len(keys)}", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
