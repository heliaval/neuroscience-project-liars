"""One-shot: build per-dyad observer-row shards mirroring the deceiver shard set.

Run from the repo root. Reads data/processed/features/single_brain.parquet,
keeps role == "observer" rows and the columns the modeling code needs, writes
shard_obs_<pair_id>.parquet x 12 plus shard_obs_manifest.json.

The manifest's key set is copied from the existing deceiver shard_manifest.json
so exp6's loader can be a line-for-line clone of exp4's load_deceiver_frame().

Divergence from the plan's literal Step 1/2 (build on the remote from the
single_brain.parquet already there): confirmed via a live kernel that the
remote's data/processed/features/ directory holds the deceiver shards +
shard_manifest.json + feature_dictionary.csv only -- no single_brain.parquet
anywhere under reveriehacks26/ (149M total tree size, no file found by
`find . -iname '*single_brain*'`). Unlike exp4/exp5, single_brain.parquet was
never uploaded to this remote in any prior task, so the "already there, no
re-upload" precondition does not hold. Falling back to Step 6 (build locally,
chunked-upload) per the plan's own instruction, logged in PROGRESS.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
SB = FEATURES_DIR / "single_brain.parquet"
# Deceiver manifest was fetched from the remote (not present on the laptop --
# see PROGRESS.md's exp5 entry) and saved to this local copy for shape reference.
DEC_MANIFEST = FEATURES_DIR / "shard_manifest.json"
OBS_MANIFEST = FEATURES_DIR / "shard_obs_manifest.json"


def main() -> None:
    dec = json.loads(DEC_MANIFEST.read_text())
    dec_keys = sorted(dec.keys())
    print(f"deceiver manifest keys: {dec_keys}", flush=True)
    column_list = dec["column_list"]  # 20 non-feature + 1770 feature + "_orig_row"
    assert column_list[-1] == "_orig_row", column_list[-5:]
    data_cols = column_list[:-1]  # the real frame columns, excluding _orig_row

    sb = pd.read_parquet(SB, columns=data_cols)
    obs = sb[sb["role"] == "observer"].copy()
    assert len(obs) == 10648, len(obs)

    # _orig_row preserves the canonical row order so the reassembled frame is
    # byte-reproducible regardless of shard read order -- same device exp4 used.
    obs = obs.reset_index(drop=True)
    obs["_orig_row"] = range(len(obs))
    obs = obs[column_list]  # exact column order match to the deceiver shard shape

    pair_ids = sorted(obs["pair_id"].unique())
    assert len(pair_ids) == 12, pair_ids
    shard_entries = {}
    for pid in pair_ids:
        out = FEATURES_DIR / f"shard_obs_{pid}.parquet"
        sub = obs.loc[obs["pair_id"] == pid]
        sub.to_parquet(out, index=False)
        shard_hash = hashlib.sha256(out.read_bytes()).hexdigest()
        shard_entries[pid] = {
            "n_rows": int(len(sub)),
            "sha256": shard_hash,
            "size_bytes": out.stat().st_size,
        }
        print(f"wrote {out.name}: {len(sub)} rows, sha256={shard_hash}", flush=True)

    full = pd.concat(
        [pd.read_parquet(FEATURES_DIR / f"shard_obs_{pid}.parquet") for pid in pair_ids],
        ignore_index=True,
    )
    full = full.sort_values("_orig_row", kind="mergesort").reset_index(drop=True)
    total_hash = hashlib.sha256(
        pd.util.hash_pandas_object(full, index=False).values.tobytes()
    ).hexdigest()

    manifest = {
        "n_rows_total": len(full),
        "n_cols": dec["n_cols"],
        "feature_set": dec["feature_set"],
        "n_feature_cols": dec["n_feature_cols"],
        "non_feature_cols": dec["non_feature_cols"],
        "column_list": column_list,
        "shards": shard_entries,
        "total_frame_sha256": total_hash,
        "role": "observer",
        "source": "data/processed/features/single_brain.parquet",
    }
    OBS_MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {OBS_MANIFEST.name}: n_rows_total={manifest['n_rows_total']} "
          f"total_frame_sha256={total_hash}", flush=True)


if __name__ == "__main__":
    main()
