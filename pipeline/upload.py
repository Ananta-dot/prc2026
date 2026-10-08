#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Talk to the OpenSky object store with the keys in ../credentials.json (needs: pip install minio).

  python3 upload.py pull                      # download every challenge file into ../data/raw
  python3 upload.py list                      # what is in the team bucket
  python3 upload.py put ../outputs/courageous-onion_v1.parquet
  python3 upload.py list prc-2026-datasets    # the shared data bucket
  python3 upload.py get prc-2026-datasets final_ranking.parquet ../data/raw/final_ranking.parquet
"""
import json, os, sys
from minio import Minio

ENDPOINT = "s3.opensky-network.org"
TEAM_BUCKET = "prc-2026-courageous-onion"
DATA_BUCKET = "prc-2026-datasets"

here = os.path.dirname(os.path.abspath(__file__))
cred = json.load(open(os.path.join(here, "..", "credentials.json")))
s3 = Minio(ENDPOINT, access_key=cred["accessKey"], secret_key=cred["secretKey"], secure=True)


def listing(bucket):
    for o in s3.list_objects(bucket, recursive=True):
        print(f"{o.last_modified}  {o.size:>12,}  {o.object_name}")


cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
if cmd == "list":
    listing(sys.argv[2] if len(sys.argv) > 2 else TEAM_BUCKET)
elif cmd == "put":
    path = sys.argv[2]
    name = os.path.basename(path)
    if name in {o.object_name for o in s3.list_objects(TEAM_BUCKET)}:
        sys.exit(f"{name} is already in {TEAM_BUCKET}; use the next version number instead")
    s3.fput_object(TEAM_BUCKET, name, path)
    print(f"uploaded {name}; bucket now holds:")
    listing(TEAM_BUCKET)
elif cmd == "pull":
    dest = os.path.join(here, "..", "data", "raw")
    os.makedirs(dest, exist_ok=True)
    for o in s3.list_objects(DATA_BUCKET, recursive=True):
        target = os.path.join(dest, o.object_name)
        if os.path.exists(target) and os.path.getsize(target) == o.size:
            print("have   ", o.object_name)
            continue
        s3.fget_object(DATA_BUCKET, o.object_name, target)
        print("fetched", o.object_name, f"{o.size:,}")
elif cmd == "get":
    s3.fget_object(sys.argv[2], sys.argv[3], sys.argv[4])
    print("saved", sys.argv[4])
else:
    sys.exit(__doc__)
