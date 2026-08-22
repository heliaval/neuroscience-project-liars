"""
scripts/build_dyadic_shards.py -- Experiment 7 (S17) data prep, Task 2.

Throwaway prep script: splits data/processed/features/dyadic.parquet into
eleven per-dyad shards (twelve including sub01_sub02, per exp4's precedent --
the *driver* does the dyad exclusion, not the shard builder) so the 167 MB
dyadic table does not need to be uploaded to the remote whole. Mirrors exp4's
proven shard scheme (shard_<pair_id>.parquet + shard_manifest.json) exactly:
_orig_row preserves original row order for exact reassembly, and
total_frame_sha256 is computed over the assembled, _orig_row-sorted frame
using the same recipe exp5_history.load_deceiver_frame() asserts against, so
src/experiments/exp7_input_sets.py's load_dyadic_frame() can consume this
manifest with zero special-casing.

Column selection: reliable+marginal dyadic features only (1,560 of 1,991
columns) -- the 420 unreliable dy_* columns are dropped, matching exp7's
modeling column set exactly (Verified facts table in the plan). This takes
the payload from 167 MB to a measured (not estimated) size, printed below.

Deterministic, re-runnable, no network access. Python 3.8 safe: no dict
union (`{...} | {...}`), no functools.cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
DYADIC_PARQUET = FEATURES_DIR / "dyadic.parquet"
FEATURE_DICT_CSV = FEATURES_DIR / "feature_dictionary.csv"
MANIFEST_PATH = FEATURES_DIR / "shard_dy_manifest.json"

KEY_COLS = ["pair_id", "session_id", "dyad_trial_seq", "condition",
            "dm_excluded_reason", "fb_excluded_reason"]


def dyadic_feature_columns() -> list:
    fd = pd.read_csv(FEATURE_DICT_CSV)
    d = fd[fd["table"] == "dyadic"]
    cols = sorted(d.loc[d["reliability"].isin(["reliable", "marginal"]), "feature_name"].tolist())
    assert len(cols) == 1560, f"expected 1560 reliable+marginal dyadic columns, got {len(cols)}"
    return cols


def main():
    print("Step 1: selecting dyadic modeling columns from feature_dictionary.csv", flush=True)
    dy_cols = dyadic_feature_columns()
    print(f"  {len(dy_cols)} columns selected (reliable+marginal)", flush=True)

    print("Step 2: reading dyadic.parquet (column-projected)", flush=True)
    read_cols = KEY_COLS + dy_cols
    df = pd.read_parquet(DYADIC_PARQUET, columns=read_cols)
    print(f"  read {len(df)} rows x {len(df.columns)} cols", flush=True)

    print("Step 3: adding _orig_row before any filtering", flush=True)
    df = df.reset_index(drop=True)
    df["_orig_row"] = np.arange(len(df))

    print("Step 4: computing total_frame_sha256 over the assembled, "
          "_orig_row-sorted frame", flush=True)
    full_sorted = df.sort_values("_orig_row", kind="mergesort").reset_index(drop=True)
    total_hash = hashlib.sha256(
        pd.util.hash_pandas_object(full_sorted, index=False).values.tobytes()
    ).hexdigest()
    print(f"  total_frame_sha256 = {total_hash}", flush=True)

    print("Step 5: writing one parquet per pair_id (all twelve, including "
          "sub01_sub02 -- driver excludes it, not this script)", flush=True)
    pair_ids = sorted(df["pair_id"].unique().tolist())
    assert len(pair_ids) == 12, f"expected 12 pair_ids, got {len(pair_ids)}: {pair_ids}"
    total_bytes = 0
    shard_sizes = {}
    for pid in pair_ids:
        shard = df[df["pair_id"] == pid].reset_index(drop=True)
        out_path = FEATURES_DIR / f"shard_dy_{pid}.parquet"
        shard.to_parquet(out_path, index=False)
        sz = out_path.stat().st_size
        total_bytes += sz
        shard_sizes[pid] = sz
        print(f"  {pid}: {len(shard)} rows, {sz:,} bytes -> {out_path.name}", flush=True)

    print("Step 6: writing manifest", flush=True)
    manifest = {
        "shards": pair_ids,
        "n_rows_total": int(len(df)),
        "n_cols": int(len(df.columns)),
        "total_frame_sha256": total_hash,
        "columns": list(df.columns),
        "dy_feature_columns": dy_cols,
        "key_columns": KEY_COLS,
        "shard_sizes_bytes": shard_sizes,
        "total_bytes": total_bytes,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest written to {MANIFEST_PATH}", flush=True)
    print(f"\nTOTAL: {len(df)} rows across {len(pair_ids)} shards, "
          f"{total_bytes:,} bytes ({total_bytes / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
