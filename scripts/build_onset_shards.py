"""One-shot: build per-dyad shards of data/processed/features/onset_windows.parquet
for remote upload, mirroring scripts/build_observer_shards.py's pattern (per-dyad
parquet + manifest with sha256, instead of one 341 MB single-file upload).

Run from the repo root. Reads onset_windows.parquet, writes
onset_shard_<pair_id>.parquet x 12 (all dyads present, though exp8's own
modeling code excludes sub01_sub02 downstream -- shipping its shard too costs
nothing and keeps this script a straight mirror of the dyad set) plus
onset_shard_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
SRC = FEATURES_DIR / "onset_windows.parquet"
MANIFEST = FEATURES_DIR / "onset_shard_manifest.json"


def main() -> None:
    df = pd.read_parquet(SRC)
    df = df.reset_index(drop=True)
    df["_orig_row"] = range(len(df))
    column_list = df.columns.tolist()

    pair_ids = sorted(df["pair_id"].unique())
    print("pair_ids:", pair_ids, flush=True)
    shard_entries = {}
    for pid in pair_ids:
        out = FEATURES_DIR / ("onset_shard_" + pid + ".parquet")
        sub = df.loc[df["pair_id"] == pid]
        sub.to_parquet(out, index=False)
        shard_hash = hashlib.sha256(out.read_bytes()).hexdigest()
        shard_entries[pid] = {
            "n_rows": int(len(sub)),
            "sha256": shard_hash,
            "size_bytes": out.stat().st_size,
        }
        print("wrote %s: %d rows, %.1f MB, sha256=%s" % (
            out.name, len(sub), out.stat().st_size / 1e6, shard_hash), flush=True)

    manifest = {
        "n_rows_total": len(df),
        "n_cols": len(column_list),
        "column_list": column_list,
        "shards": shard_entries,
        "source": "data/processed/features/onset_windows.parquet",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print("wrote", MANIFEST.name, "n_rows_total=", manifest["n_rows_total"])


if __name__ == "__main__":
    main()
