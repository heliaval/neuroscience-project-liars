#!/usr/bin/env python3
"""Validate a results.v1 candidate file against results/schema/results.v1.schema.json.

Usage:
    python scripts/validate_results.py <path-to-results-json>

Exits 0 on valid, non-zero with every JSON-pointer error printed on invalid.
This is the shape-drift gate: both the fixture and the real
`results/results.v1.json` run through this before either is trusted.

Note: `results/fixtures/results.v1.fixture.json` no longer validates against
this schema -- it still carries an `experiments.exp8` block from before
Experiment 8's run was halted, and the schema has since dropped exp8 from
its contract. This is expected, not a defect; do not "fix" it by
regenerating the fixture. See `results/results.v1.json` for the current,
exp8-free, real emitted file.
"""
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema is not installed. Run: pip install jsonschema", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "results" / "schema" / "results.v1.schema.json"


def _path_str(error: "jsonschema.exceptions.ValidationError") -> str:
    if not error.absolute_path:
        return "$"
    return "$." + ".".join(str(p) for p in error.absolute_path)



# Constraint §3.2 (plans/results-v1-fixture.md) bounds *invented* AUROC values to
# [0.46, 0.58] / [0.40, 0.65] sanity range. That bound applies to placeholder
# sections only -- real transcribed per-participant/per-dyad scores (exp1, exp2,
# exp3, exp4, exp5, exp6, exp7, gate, frozen) are copied verbatim from the
# pipeline's real output and can legitimately fall outside it (e.g. a single
# participant's personalized-model AUROC on a 161-trial test fold, or exp6's
# real per-participant minimum of 0.3994252873563218). See PROGRESS.md for the
# divergence note this resolved. "delta"-keyed subtrees hold paired differences,
# not AUROC values, and are exempt from the AUROC bound regardless of section.
#
# Experiment 8 is gone from the contract entirely (its run was halted before
# writing results); it no longer appears here or anywhere else in the schema.
# interpretability/interbrain/trials/failures are also gone entirely (dropped
# once real replacements existed but nothing in the app ever rendered them --
# see PROGRESS.md) -- no placeholder sections remain in the contract.
PLACEHOLDER_PATH_PREFIXES = ()


def _in_placeholder_section(path: str) -> bool:
    if any(path.startswith(p) for p in PLACEHOLDER_PATH_PREFIXES):
        return True
    # dyads[i].fingerprint.* is placeholder except personalization_gain (checked
    # separately, and personalization_gain is not an "auroc"-keyed leaf anyway).
    if path.startswith("$.dyads[") and ".fingerprint" in path:
        return True
    return False


def check_invariants(data: dict) -> list[str]:
    """Invariants JSON Schema cannot express, checked explicitly."""
    errors: list[str] = []

    def walk(node, path):
        if isinstance(node, dict):
            keys = set(node.keys())
            if {"n", "n_positive", "n_negative", "n_ties_excluded"} <= keys:
                total = node.get("n_positive", 0) + node.get("n_negative", 0) + node.get("n_ties_excluded", 0)
                if total != node.get("n"):
                    errors.append(
                        f"{path}: n_positive + n_negative + n_ties_excluded "
                        f"({node.get('n_positive')}+{node.get('n_negative')}+{node.get('n_ties_excluded')}"
                        f"={total}) != n ({node.get('n')})"
                    )
            for k, v in node.items():
                if k == "auroc" and isinstance(v, (int, float)) and "delta" not in path:
                    if _in_placeholder_section(f"{path}.{k}") and not (0.40 <= v <= 0.65):
                        errors.append(f"{path}.{k}: AUROC {v} outside [0.40, 0.65] invented-value regime (§3.2)")
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(data, "$")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    candidate_path = Path(sys.argv[1])
    if not candidate_path.exists():
        print(f"ERROR: no such file: {candidate_path}", file=sys.stderr)
        return 2

    schema = json.loads(SCHEMA_PATH.read_text())
    data = json.loads(candidate_path.read_text())

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))

    ok = True
    if errors:
        ok = False
        print(f"SCHEMA VALIDATION FAILED for {candidate_path} ({len(errors)} error(s)):")
        for e in errors:
            print(f"  {_path_str(e)}: {e.message}")

    invariant_errors = check_invariants(data)
    if invariant_errors:
        ok = False
        print(f"INVARIANT CHECKS FAILED ({len(invariant_errors)}):")
        for msg in invariant_errors:
            print(f"  {msg}")

    if ok:
        print(f"OK: {candidate_path} is a valid results.v1 file.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
