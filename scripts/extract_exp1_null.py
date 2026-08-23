"""One-time extraction of exp1's permutation-null distribution.

The real exp1 run computed 200 shuffled-label AUROCs and checkpointed them to
exp1_checkpoints/permutation_null_partial.pkl (see src/experiments/exp1_baseline.py),
but only their mean, sd, and 5th/50th/95th percentiles were ever written into a
results file. The walkthrough renders the actual distribution, so the array itself
needs to live in a committed file: exp1_checkpoints/ and *.pkl are both gitignored,
which would otherwise make scripts/emit_results.py unreproducible from a clone.

This script computes no statistic. It transcribes the array and re-asserts that it
reproduces the already-published summary exactly, so a drifted or truncated
checkpoint fails loudly instead of silently changing the picture.
"""
import json
import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "exp1_checkpoints" / "permutation_null_partial.pkl"
OUT = REPO_ROOT / "results" / "exp1_permutation_null.json"
PUBLISHED = REPO_ROOT / "results" / "results.v1.json"


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(
            f"missing {CHECKPOINT}. It is gitignored and was not regenerated. "
            "Do not fabricate values -- report this instead."
        )

    with open(CHECKPOINT, "rb") as f:
        values = [float(v) for v in pickle.load(f)]

    pub = json.load(open(PUBLISHED, encoding="utf8"))["experiments"]["exp1"]["permutation_null"]
    arr = np.asarray(values, dtype=float)

    assert len(values) == pub["n_permutations"], (len(values), pub["n_permutations"])
    assert float(arr.mean()) == pub["null"]["mean"]
    assert float(arr.std(ddof=1)) == pub["null"]["sd"]
    for q in ("5", "50", "95"):
        assert float(np.percentile(arr, int(q))) == pub["null"]["percentiles"][q], q

    payload = {
        "source": "exp1_checkpoints/permutation_null_partial.pkl",
        "produced_by": "src/experiments/exp1_baseline.py",
        "n_permutations": len(values),
        "note": (
            "Per-permutation AUROC under within-dataset label shuffling, primary model "
            "(logistic regression), primary feature set. Transcribed verbatim from the run "
            "checkpoint; reproduces the published mean, sd, and 5th/50th/95th percentiles exactly."
        ),
        "distribution": values,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
    print(f"wrote {OUT} ({len(values)} values, {OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
