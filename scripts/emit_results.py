"""Assembly script for results/results.v1.json (plans/wire-real-results-and-drop-exp8.md).

Reads results/exp2_universal.json, results/exp5_history.json,
results/exp6_observer.json, and results/exp7_input_sets.json -- the four
experiments that have finished running since results/fixtures/results.v1.fixture.json
was built -- and copies their real numbers programmatically into the emitted file.
No new statistic is computed anywhere in this script; every value is either a
verbatim copy of a real result, a verbatim copy of an already-validated
placeholder section from the fixture, or a formatted string built from real
values.

experiments.exp1, exp3, exp4 (and gate, frozen, dyads) are lifted from the
fixture rather than re-derived from their own source files -- those blocks are
already validated real transcriptions and copying them is strictly safer than
duplicating that transcription here.

interpretability, interbrain, trials, and failures -- the fixture's four
invented placeholder sections -- are dropped from the contract entirely (not
emitted at all), rather than emitted with real or placeholder data. Nothing in
the app ever rendered them (verified: no component references them outside
ProvenanceBanner's disclosure copy), so continuing to carry invented numbers
for sections nobody displays served no purpose once noticed. See PROGRESS.md
for the investigation.

Experiment 8 is deliberately absent. Its run was halted before writing any
results (an explicit user decision, not a bug or an oversight) -- there is no
results/exp8_onset.json to read, and results/schema/results.v1.schema.json no
longer has an exp8 branch to satisfy. Nothing about exp8's halted run is
touched anywhere else in the repo.

Writes results/results.v1.json.
"""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

fixture = json.load(open(RESULTS / "fixtures" / "results.v1.fixture.json"))
exp2_raw = json.load(open(RESULTS / "exp2_universal.json"))
exp5_raw = json.load(open(RESULTS / "exp5_history.json"))
exp6_raw = json.load(open(RESULTS / "exp6_observer.json"))
exp7_raw = json.load(open(RESULTS / "exp7_input_sets.json"))  # flat -- this file IS the exp7 block

e2 = exp2_raw["experiments"]["exp2"]
e5 = exp5_raw["experiments"]["exp5"]
e6 = exp6_raw["experiments"]["exp6"]
e7 = exp7_raw

lr = e2["models"]["logistic_regression"]

EXP1_POOLED_AUROC = fixture["experiments"]["exp1"]["headline"]["value"]


def keep(d, *keys):
    """Select a fixed set of keys from a dict -- never a `dict(**d)` splat."""
    return {k: d[k] for k in keys if k in d}


# ---------------------------------------------------------------------------
# Sanity: verify against sources before assembling anything (mirrors
# build_fixture.py's own Step-2 pattern)
# ---------------------------------------------------------------------------
assert lr["mean"]["auroc"] == 0.5130592192193698
assert lr["n_dyads_above_0.5_auroc"] == 9
assert len(lr["per_dyad"]) == 12
assert e2["permutation_null"]["p_value"] == 0.05472636815920398

assert len(e5["per_dyad"]) == 10
hg = e5["tests"]["history_gain"]["result"]
assert hg["median_delta"] == 0.01362457573036091
assert hg["n"] == 10 and hg["n_positive"] == 8
assert hg["sign_test_p"] == 0.109375
assert hg["permutation_p"] == 0.17178282171782822

assert len(e6["per_dyad"]) == 10
oel = e6["tests"]["observer_positional_late_minus_early"]["result"]
assert oel["median_delta"] == 0.02380320510543457
assert oel["permutation_p"] == 0.4042595740425957

assert len(e7["per_dyad"]) == 11
assert e7["input_sets"] == ["deceiver_eeg", "observer_eeg", "both_brains", "interbrain", "eeg_plus_behavioral"]
bbvd = e7["tests"]["both_brains_vs_deceiver_eeg"]
assert bbvd["result"]["median_delta"] == 0.007267871130115511
assert e7["per_input_set"]["both_brains"]["median_auroc"] == 0.5273944909932384
assert e7["per_input_set"]["interbrain"]["median_auroc"] == 0.4879862221458744

# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
meta = {
    "schema_version": "results.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "pipeline_git_describe": None,
    "dataset_checksum": None,
    "is_fixture": False,
    "provenance": {
        "gate": "real", "frozen": "real", "dyads": "mixed",
        "exp1": "real", "exp2": "real", "exp3": "real", "exp4": "real",
        "exp5": "real", "exp6": "real", "exp7": "real",
        "tests": "real",
    },
    "sources": {
        "exp1": fixture["meta"]["sources"]["exp1"],
        "exp3": fixture["meta"]["sources"]["exp3"],
        "exp4": fixture["meta"]["sources"]["exp4"],
        "exp2": {"file": "results/exp2_universal.json", "generated_at": exp2_raw["meta"]["generated_at"], "seed": 0},
        "exp5": {"file": "results/exp5_history.json", "generated_at": exp5_raw["meta"]["generated_at"], "seed": 0},
        "exp6": {"file": "results/exp6_observer.json", "generated_at": exp6_raw["meta"]["generated_at"], "seed": 0},
        "exp7": {"file": "results/exp7_input_sets.json", "generated_at": None, "seed": 0},
        "placeholder_sections": {
            "file": "results/fixtures/results.v1.fixture.json",
            "sections": ["dyads[].fingerprint"],
        },
    },
}

# ---------------------------------------------------------------------------
# gate, frozen, dyads -- verbatim from the fixture, unchanged by this plan.
# interpretability/interbrain/trials/failures are intentionally NOT lifted --
# see the module docstring.
# ---------------------------------------------------------------------------
gate = fixture["gate"]
frozen = fixture["frozen"]
dyads = fixture["dyads"]

# ---------------------------------------------------------------------------
# experiments.exp1, exp3, exp4 -- verbatim from the fixture, already real
# ---------------------------------------------------------------------------
exp1 = fixture["experiments"]["exp1"]

# exp1's permutation null: the fixture carries only the summary (mean, sd, three
# percentiles). The real run's 200 per-permutation AUROCs live in
# results/exp1_permutation_null.json (transcribed by scripts/extract_exp1_null.py from
# the run checkpoint). Inject the array so the frontend can render the actual
# distribution instead of a reconstruction. No statistic is computed here -- this is a
# verbatim copy, re-checked against the summary the fixture already published.
_exp1_null = json.load(open(RESULTS / "exp1_permutation_null.json", encoding="utf8"))
_dist = _exp1_null["distribution"]
assert len(_dist) == exp1["permutation_null"]["n_permutations"] == _exp1_null["n_permutations"]
assert min(_dist) > 0.0 and max(_dist) < 1.0
exp1 = copy.deepcopy(exp1)
exp1["permutation_null"]["null"]["distribution"] = _dist

exp3 = fixture["experiments"]["exp3"]
exp4 = fixture["experiments"]["exp4"]

# ---------------------------------------------------------------------------
# experiments.exp2 -- from results/exp2_universal.json (§5.1)
# ---------------------------------------------------------------------------
exp2_models = {}
for name, m in e2["models"].items():
    stripped = keep(
        m, "primary", "mean", "ci95", "median", "min", "max",
        "n_dyads_above_0.5_auroc", "inner_splitter_class", "convergence_warnings", "per_fold",
    )
    exp2_models[name] = stripped

exp2_per_dyad = [
    keep(
        d, "pair_id", "fold", "n_test", "n_lie", "n_truth", "class_balance_pos",
        "auroc", "balanced_accuracy", "f1", "precision", "recall",
    )
    for d in lr["per_dyad"]
]
assert len(exp2_per_dyad) == 12

exp2 = {
    "status": "complete",
    "provenance": "real",
    "gate_verdict": "CONFIRMATORY",
    "design": e2["design"],
    "headline": {
        "metric": "auroc", "model": "logistic_regression",
        "value": lr["mean"]["auroc"], "ci95": lr["ci95"]["auroc"],
    },
    "models": exp2_models,
    "per_dyad": exp2_per_dyad,
    "per_dyad_scores": lr["per_dyad_scores"],
    "permutation_null": e2["permutation_null"],
    "comparison_to_exp1": e2["comparison_to_exp1"],
}

# ---------------------------------------------------------------------------
# experiments.exp5 -- from results/exp5_history.json (§5.3)
# ---------------------------------------------------------------------------
def strip_curve_entry(c):
    return {k: v for k, v in c.items() if k != "best_params"}

exp5_per_dyad = []
for d in e5["per_dyad"]:
    entry = keep(
        d, "pair_id", "tested_participant", "partner", "held_out_seq_min",
        "held_out_seq_max", "n_test", "curve", "control_curve_other_dyad",
        "positional_bins", "majority_reference",
    )
    entry["curve"] = [strip_curve_entry(c) for c in entry["curve"]]
    entry["control_curve_other_dyad"] = [strip_curve_entry(c) for c in entry["control_curve_other_dyad"]]
    exp5_per_dyad.append(entry)
assert len(exp5_per_dyad) == 10

exp5 = {
    "status": "complete",
    "provenance": "real",
    "gate_verdict": e5["gate_recheck"],
    "design": e5["design"],
    "test": e5["tests"]["history_gain"]["result"],
    "tests": {k: v for k, v in e5["tests"].items()},
    "curve_slopes": e5["curve_slopes"],
    "aggregate": e5["aggregate"],
    "cross_experiment": e5["cross_experiment"],
    "per_dyad": exp5_per_dyad,
}

# ---------------------------------------------------------------------------
# experiments.exp6 -- from results/exp6_observer.json (§5.3)
# ---------------------------------------------------------------------------
exp6_per_dyad = [
    keep(
        d, "pair_id", "observers", "positional_bins", "early", "middle", "late",
        "delta_late_minus_early", "majority_reference", "n_train", "inner_splitter_class",
    )
    for d in e6["per_dyad"]
]
assert len(exp6_per_dyad) == 10

exp6 = {
    "status": e6["status"],
    "provenance": e6["provenance"],
    "gate_verdict": e6["gate_verdict"],
    "design": e6["design"],
    "gate_recheck": e6["gate_recheck"],
    "aggregate": e6["aggregate"],
    "interpretation": e6["interpretation"],
    "cross_experiment": e6["cross_experiment"],
    "test": e6["tests"]["observer_positional_late_minus_early"]["result"],
    "tests": {k: v for k, v in e6["tests"].items()},
    "per_dyad": exp6_per_dyad,
}

BANNED_PHRASES = [p.lower() for p in e6["interpretation"]["banned_phrases"]]

# ---------------------------------------------------------------------------
# experiments.exp7 -- from results/exp7_input_sets.json, flat (§5.3)
# ---------------------------------------------------------------------------
exp7 = keep(
    e7, "status", "provenance", "gate_verdict", "design", "input_sets",
    "input_set_labels", "input_set_widths", "per_dyad", "per_input_set",
    "tests", "tests_exploratory_bh",
)
assert len(exp7["per_dyad"]) == 11

experiments = {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4, "exp5": exp5, "exp6": exp6, "exp7": exp7}

# ---------------------------------------------------------------------------
# tests -- nine entries: five copied verbatim from the fixture (already real),
# four rebuilt from the newly-real experiments. No exp8_* key is emitted.
# ---------------------------------------------------------------------------
fixture_tests = fixture["tests"]

exp5_median = hg["median_delta"]
exp5_sign_p = hg["sign_test_p"]
exp5_perm_p = hg["permutation_p"]

exp6_median = oel["median_delta"]
exp6_sign_p = oel["sign_test_p"]
exp6_perm_p = oel["permutation_p"]

exp7_median = bbvd["result"]["median_delta"]
exp7_sign_p = bbvd["result"]["sign_test_p"]
exp7_perm_p = bbvd["result"]["permutation_p"]

tests = {
    "exp1_above_chance": fixture_tests["exp1_above_chance"],
    "h1_lodo_generalization": {
        **{k: fixture_tests["h1_lodo_generalization"][k] for k in ("claim", "hypothesis", "experiment", "metric", "primary")},
        "kind": "permutation",
        "designation": "confirmatory",
        "supported": False,
        "result": e2["permutation_null"],
        "plain_language": "A model trained on eleven pairs didn't reliably predict deception in the twelfth pair, the one it had never seen.",
        "technical": (
            f"LODO mean AUROC = {lr['mean']['auroc']:.4f} across 12 held-out dyads, "
            f"{lr['n_dyads_above_0.5_auroc']} above 0.5; within-dyad label-permutation "
            f"p = {e2['permutation_null']['p_value']:.4f} (200 permutations). exp1's pooled "
            f"{EXP1_POOLED_AUROC:.4f} is an optimistically biased upper bound, not a matched comparison."
        ),
    },
    "h2_person_gain": fixture_tests["h2_person_gain"],
    "exp3_personalized_vs_population": fixture_tests["exp3_personalized_vs_population"],
    "h3_dyad_gain": fixture_tests["h3_dyad_gain"],
    "nfi_distribution": fixture_tests["nfi_distribution"],
    "exp5_learning_curve": {
        **{k: fixture_tests["exp5_learning_curve"][k] for k in ("claim", "hypothesis", "experiment", "metric", "kind", "designation", "primary")},
        "supported": bool(exp5_median > 0 and exp5_perm_p < 0.05),
        "result": hg,
        "plain_language": "Training on more of a pair's own history exhibited a slight increase, but not a reliable one.",
        "technical": (
            f"Median ΔAUROC (k=646 vs k=322, same-dyad) = {exp5_median:.4f}; "
            f"8 of 10 dyads positive; sign test p = {exp5_sign_p:.4f}; "
            f"permutation p = {exp5_perm_p:.4f}, n = 10 dyads."
        ),
    },
    "exp6_observer_early_late": {
        **{k: fixture_tests["exp6_observer_early_late"][k] for k in ("claim", "hypothesis", "experiment", "metric", "kind", "designation", "primary")},
        "supported": e6["tests"]["observer_positional_late_minus_early"]["supported"],
        "result": oel,
        "plain_language": e6["tests"]["observer_positional_late_minus_early"]["plain_language"],
        "technical": (
            f"Median ΔAUROC (late minus early positional bin) = {exp6_median:.4f}; "
            f"6 of 10 dyads positive; sign test p = {exp6_sign_p:.4f}; "
            f"permutation p = {exp6_perm_p:.4f}, n = 10 dyads."
        ),
    },
    "exp7_observer_vs_deceiver": {
        **{k: fixture_tests["exp7_observer_vs_deceiver"][k] for k in ("claim", "hypothesis", "experiment", "metric", "kind", "designation", "primary")},
        "supported": bbvd["supported"],
        "result": bbvd["result"],
        "plain_language": "Adding the observer's brain to the deceiver's own boosted prediction in 7 of 11 pairs, but not reliably enough to trust.",
        "technical": (
            f"Median ΔAUROC (both brains minus deceiver only) = {exp7_median:.4f}; "
            f"sign test p = {exp7_sign_p:.4f}; permutation p = {exp7_perm_p:.4f}, n = 11 dyads. "
            f"Median AUROC by input set: deceiver {e7['per_input_set']['deceiver_eeg']['median_auroc']:.4f}, "
            f"observer {e7['per_input_set']['observer_eeg']['median_auroc']:.4f}, "
            f"both {e7['per_input_set']['both_brains']['median_auroc']:.4f}, "
            f"inter-brain {e7['per_input_set']['interbrain']['median_auroc']:.4f}, "
            f"EEG+behavioral {e7['per_input_set']['eeg_plus_behavioral']['median_auroc']:.4f}."
        ),
    },
}

# Check every changed claim's derived-`supported` rule against the real flags
# where the real file already carries one (§5.5's rule must reproduce these).
assert tests["exp6_observer_early_late"]["supported"] is False
assert tests["exp7_observer_vs_deceiver"]["supported"] is False

# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
results = {
    "meta": meta, "gate": gate, "frozen": frozen, "dyads": dyads,
    "experiments": experiments, "tests": tests,
}

# ---------------------------------------------------------------------------
# Assert the contract before writing (Task 4's final checkboxes)
# ---------------------------------------------------------------------------
SCHEMA_TOP_KEYS = {"meta", "gate", "frozen", "dyads", "experiments", "tests"}
assert set(results.keys()) == SCHEMA_TOP_KEYS
assert "exp8" not in results["experiments"]
assert "exp8" not in results["meta"]["provenance"]
assert not [k for k in results["tests"] if k.startswith("exp8")]
assert results["meta"]["is_fixture"] is False
for _e in ("exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"):
    assert results["meta"]["provenance"][_e] == "real", _e


def _walk_for_forbidden_keys(node, forbidden, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in forbidden:
                raise AssertionError(f"forbidden key {k!r} present at {path}.{k}")
            _walk_for_forbidden_keys(v, forbidden, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_for_forbidden_keys(v, forbidden, f"{path}[{i}]")


_walk_for_forbidden_keys(
    results,
    {"coefs_per_fold", "feature_names", "best_params_per_fold", "coefficients", "validations"},
)


def _walk_for_banned_phrases(node, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("plain_language", "technical") and isinstance(v, str):
                low = v.lower()
                for phrase in BANNED_PHRASES:
                    if phrase in low:
                        raise AssertionError(f"banned phrase {phrase!r} found in {path}.{k}")
            _walk_for_banned_phrases(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_for_banned_phrases(v, f"{path}[{i}]")


_walk_for_banned_phrases(results)

# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
out_path = RESULTS / "results.v1.json"
payload = json.dumps(results, indent=2)
out_path.write_text(payload)
print(f"Wrote {out_path} ({len(payload.encode('utf-8'))} bytes)")
