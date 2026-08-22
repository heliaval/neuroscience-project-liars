"""Transcription script for results/fixtures/results.v1.fixture.json (throwaway helper,
kept in scripts/ per plans/results-v1-fixture.md Task 4 Step 1).

Loads the real result files and copies real numbers programmatically -- never retypes
them by hand -- then adds clearly-flagged invented placeholder numbers for the sections
that have no real experiment yet (exp2, exp5-exp8, interpretability, interbrain, trials,
failures), all constrained to the real regime per plans/results-v1-fixture.md Global
Constraints.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

gate_raw = json.load(open(RESULTS / "gate.json"))
exp1_raw = json.load(open(RESULTS / "exp1_baseline.json"))
exp3_raw = json.load(open(RESULTS / "exp3_personalized.json"))
exp4_raw = json.load(open(RESULTS / "exp4_dyadic.json"))

rng = random.Random(20260822)  # fixed seed -- placeholder numbers are reproducible

# ---------------------------------------------------------------------------
# Sanity: verify against source before doing anything else (plan Task 4 Step 2)
# ---------------------------------------------------------------------------
_dg = exp4_raw["experiments"]["exp4"]["tests"]["dyad_gain"]["auroc"]
assert _dg["median_delta"] == -0.018482431899607993
assert _dg["sign_test_p"] == 0.109375
assert _dg["permutation_p"] == 0.051794820517948204
assert exp1_raw["experiments"]["exp1"]["models"]["logistic_regression"]["mean"]["auroc"] == 0.533789809045845

EXP1_POOLED_AUROC = 0.533789809045845

# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
meta = {
    "schema_version": "results.v1",
    "generated_at": "2026-08-22T00:00:00+00:00",
    "pipeline_git_describe": None,
    "dataset_checksum": None,
    "is_fixture": True,
    "provenance": {
        "gate": "real", "frozen": "real", "dyads": "mixed",
        "exp1": "real", "exp2": "placeholder", "exp3": "real", "exp4": "real",
        "exp5": "placeholder", "exp6": "placeholder", "exp7": "placeholder", "exp8": "placeholder",
        "tests": "mixed", "interpretability": "placeholder",
        "interbrain": "placeholder", "trials": "placeholder", "failures": "placeholder",
    },
    "sources": {
        "exp1": {"file": "results/exp1_baseline.json", "generated_at": exp1_raw["meta"]["generated_at"], "seed": 0},
        "exp3": {"file": "results/exp3_personalized.json", "generated_at": exp3_raw["meta"]["generated_at"], "seed": 0},
        "exp4": {"file": "results/exp4_dyadic.json", "generated_at": exp4_raw["meta"]["generated_at"], "seed": 0},
        "exp2": None, "exp5": None, "exp6": None, "exp7": None, "exp8": None,
    },
}

# ---------------------------------------------------------------------------
# gate -- verbatim copy of gate.json -> .gate
# ---------------------------------------------------------------------------
gate = gate_raw["gate"]

# ---------------------------------------------------------------------------
# frozen -- verbatim copy of gate.json -> .frozen, plus amendments (not in gate.json)
# ---------------------------------------------------------------------------
frozen = dict(gate_raw["frozen"])
frozen["amendments"] = [
    {
        "id": "Amendment 1",
        "date": "2026-08-22",
        "applies_to": "exp4",
        "change": "unit-of-analysis count reduced from n = 11 to n = 10",
        "excluded_unit": "sub19_sub22",
        "rationale": (
            "sub19_sub22 is excluded from exp4 as a unit of analysis because it is "
            "structurally untestable at dyad grain: its only participant with "
            "deceiver-role data (sub19; sub22 was lost in §9 preprocessing) is the "
            "session-1 deceiver, so the dyad's late tercile of dyad_trial_seq -- exp4's "
            "held-out block per results/gate.json's frozen exp4 assumption -- contains no "
            "deceiver trials at all. This is a data-availability exclusion of the same kind "
            "as sub01_sub02's, determined from data structure before any exp4 model was "
            "fit; no exp4 result had been computed when this amendment was written. "
            "sub19_sub22's deceiver rows remain in exp4's Universal training pool. This "
            "amendment applies to exp4 only; exp5-exp8 are untouched and remain at n = 11 "
            "pending their own analyses."
        ),
        "post_hoc": False,
    }
]

# ---------------------------------------------------------------------------
# dyads -- 12 entries
# ---------------------------------------------------------------------------
trials_per_dyad = gate["counts"]["trials_per_dyad"]
seq_range_per_dyad = gate["counts"]["seq_range_per_dyad"]
sessions_per_dyad = gate["counts"]["sessions_per_dyad"]

DYAD_IDS = sorted(trials_per_dyad.keys())
assert len(DYAD_IDS) == 12

exp3_delta_by_dyad = dict(zip(
    exp3_raw["experiments"]["exp3"]["paired_test"]["primary"]["dyad_ids"],
    exp3_raw["experiments"]["exp3"]["paired_test"]["primary"]["deltas"],
))
exp4_pg_deltas = dict(zip(
    exp4_raw["experiments"]["exp4"]["tests"]["dyad_gain"]["auroc"]["dyad_ids"],
    exp4_raw["experiments"]["exp4"]["tests"]["dyad_gain"]["auroc"]["deltas"],
))

dyads = []
for pid in DYAD_IDS:
    participants = pid.split("_")
    n_trials = trials_per_dyad[pid]
    # real personalization_gain where exp3/exp4 measured it; else None (excluded dyad)
    personalization_gain = exp3_delta_by_dyad.get(pid)
    included_in = ["exp1", "exp2"]
    exclusion_note = None
    if pid == "sub01_sub02":
        exclusion_note = (
            "Excluded from exp3-8 by frozen_hypotheses.md: sub01_sub02 has no bidirectional "
            "deceiver-role history (sub02 was never recorded as deceiver -- an archive gap, "
            "not a property of the participant), so it stays in exp1/exp2 (pooled/LODO, "
            "role-agnostic) and is dropped everywhere the design compares role-swapped history."
        )
    else:
        included_in += ["exp3", "exp4", "exp5", "exp6", "exp7", "exp8"]
        if pid == "sub19_sub22":
            exclusion_note = (
                "Excluded from exp4 as a unit of analysis (frozen_hypotheses.md Amendment 1): "
                "its only deceiver-role participant, sub19, is the session-1 deceiver, so this "
                "dyad's late-tercile held-out block (exp4's design) contains zero deceiver rows. "
                "The dyad's rows remain in exp4's Universal training pool, and it is untouched "
                "in exp1, exp2, exp3, exp5-8."
            )
    # class balance: real from gate.json if present, else invented near pooled ~0.475
    minority_fraction = round(rng.uniform(0.44, 0.51), 3)
    lie = round(n_trials * minority_fraction)
    dyads.append({
        "pair_id": pid,
        "participants": participants,
        "n_sessions": sessions_per_dyad[pid],
        "n_trials": n_trials,
        "seq_range": seq_range_per_dyad[pid],
        "class_balance": {"lie": lie, "truth": n_trials - lie, "minority_fraction": minority_fraction},
        "included_in": included_in,
        "exclusion_note": exclusion_note,
        "fingerprint": {
            "frontal_coupling": round(rng.uniform(0.15, 0.55), 3),
            "temporal_lag_ms": round(rng.uniform(-120, 120)),
            "alpha_synchrony": round(rng.uniform(0.10, 0.45), 3),
            "observer_response": round(rng.uniform(0.05, 0.40), 3),
            "personalization_gain": personalization_gain if personalization_gain is not None else round(rng.uniform(-0.05, 0.05), 4),
        },
    })
assert len(dyads) == 12

# ---------------------------------------------------------------------------
# experiments.exp1 -- real
# ---------------------------------------------------------------------------
e1 = exp1_raw["experiments"]["exp1"]

def strip_model(m):
    return {
        "primary": m.get("primary", False),
        "mean": m["mean"],
        "ci95": m["ci95"],
        "per_fold": [
            {k: v for k, v in f.items()} for f in m["per_fold"]
        ],
    }

exp1 = {
    "status": "complete",
    "provenance": "real",
    "gate_verdict": "N/A (not gated)",
    "design": dict(e1["design"], n_features=e1["feature_sets"]["reliable_plus_marginal"]["n_features"]),
    "headline": {
        "metric": "auroc",
        "model": "logistic_regression",
        "value": e1["models"]["logistic_regression"]["mean"]["auroc"],
        "ci95": e1["models"]["logistic_regression"]["ci95"]["auroc"],
    },
    "models": {name: strip_model(m) for name, m in e1["models"].items()},
    "permutation_null": e1["permutation_null"],
    "cnn_sanity_check": {
        "purpose": e1["cnn_sanity_check"]["purpose"],
        "n_matched_trials": e1["cnn_sanity_check"]["n_matched_trials"],
        "overlap_fraction": e1["cnn_sanity_check"]["overlap_fraction"],
        "observed_auroc_mean": sum(f["auroc"] for f in e1["cnn_sanity_check"]["per_fold"]) / len(e1["cnn_sanity_check"]["per_fold"]),
        "delta_vs_lr": (sum(f["auroc"] for f in e1["cnn_sanity_check"]["per_fold"]) / len(e1["cnn_sanity_check"]["per_fold"])) - e1["models"]["logistic_regression"]["mean"]["auroc"],
        "verdict": "features adequate; no §10 revisit indicated",
    },
}

# ---------------------------------------------------------------------------
# experiments.exp2 -- placeholder, shape locked by experiment2-universal-model.md
# ---------------------------------------------------------------------------
exp2_dyad_ids = DYAD_IDS  # all 12
per_dyad_auroc = {}
for pid in exp2_dyad_ids:
    per_dyad_auroc[pid] = round(rng.uniform(0.47, 0.56), 4)
lodo_mean = round(sum(per_dyad_auroc.values()) / len(per_dyad_auroc), 4)
assert lodo_mean < EXP1_POOLED_AUROC

exp2_per_dyad = []
for pid in exp2_dyad_ids:
    n_test = trials_per_dyad[pid]
    n_lie = round(n_test * rng.uniform(0.44, 0.51))
    exp2_per_dyad.append({
        "pair_id": pid,
        "n_test": n_test,
        "n_lie": n_lie,
        "auroc": per_dyad_auroc[pid],
        "balanced_accuracy": round(per_dyad_auroc[pid] - rng.uniform(0.0, 0.02), 4),
        "f1": round(rng.uniform(0.40, 0.55), 4),
        "precision": round(rng.uniform(0.45, 0.55), 4),
        "recall": round(rng.uniform(0.40, 0.55), 4),
    })
assert len(exp2_per_dyad) == 12

exp2 = {
    "status": "running",
    "provenance": "placeholder",
    "gate_verdict": "CONFIRMATORY",
    "design": {
        "research_question": "S12/H1: does a model trained on all-but-one dyad generalize to the held-out dyad?",
        "cv": "LeaveOneGroupOut(groups=pair_id)",
        "n_folds": 12,
        "inner_cv": "StratifiedGroupKFold(3) on pair_id (S19)",
        "relation_to_exp1": "exp1 pooled = upper bound; exp2 LODO = the generalization claim",
    },
    "headline": {"metric": "auroc", "model": "logistic_regression", "value": lodo_mean, "ci95": {}},
    "models": {
        "logistic_regression": {
            "primary": True, "mean": {}, "ci95": {},
            "median": round(sorted(per_dyad_auroc.values())[len(per_dyad_auroc) // 2], 4),
            "min": round(min(per_dyad_auroc.values()), 4),
            "max": round(max(per_dyad_auroc.values()), 4),
            "n_dyads_above_0.5": sum(1 for v in per_dyad_auroc.values() if v > 0.5),
        }
    },
    "per_dyad": exp2_per_dyad,
    "per_dyad_scores": {"auroc": per_dyad_auroc},
    "permutation_null": {
        "n_permutations": 200, "shuffle_scheme": "within-dyad",
        "observed_auroc": lodo_mean,
        "null": {"mean": 0.500, "sd": 0.009},
        "p_value": round(rng.uniform(0.05, 0.4), 3),
    },
    "comparison_to_exp1": {
        "per_family": {"logistic_regression": {"exp1_auroc": EXP1_POOLED_AUROC, "exp2_auroc": lodo_mean, "delta": round(lodo_mean - EXP1_POOLED_AUROC, 4)}},
        "n_dyads_above_chance": sum(1 for v in per_dyad_auroc.values() if v > 0.5),
    },
}

# ---------------------------------------------------------------------------
# experiments.exp3 -- real
# ---------------------------------------------------------------------------
e3 = exp3_raw["experiments"]["exp3"]
_dg3 = e3["paired_test"]["primary"]
exp3_pairedtest = {
    "n": _dg3["n"], "dyad_ids": _dg3["dyad_ids"], "deltas": _dg3["deltas"],
    "median_delta": _dg3["median_delta"], "n_positive": _dg3["n_positive"],
    "n_negative": _dg3["n_negative"], "n_ties_excluded": _dg3["n"] - _dg3["n_positive"] - _dg3["n_negative"],
    "sign_test_p": _dg3["sign_test_p"], "permutation_p": _dg3["permutation_p"],
    "n_signflip": _dg3["n_signflip"], "ci95": _dg3["ci95"],
}

per_participant = []
for p in e3["per_participant"]:
    per_participant.append({
        "participant_id": p["participant_id"], "pair_id": p["pair_id"], "partner_id": p["partner_id"],
        "n_train_personalized": p["n_train_personalized"], "n_train_population": p["n_train_population"],
        "n_test": p["n_test"], "test_n_lie": p["test_n_lie"], "test_n_truth": p["test_n_truth"],
        "test_seq_min": p["test_seq_min"], "test_seq_max": p["test_seq_max"],
        "population": p["population"], "personalized": p["personalized"], "delta": p["delta"],
    })

per_dyad = []
pd_scores = e3["per_dyad_scores"]
for pid in e3["participant_reconciliation"]["resulting_dyads"]:
    idx = pd_scores["delta"]["auroc"] if isinstance(pd_scores["delta"], dict) else None
    per_dyad.append({
        "pair_id": pid,
        "population_auroc": pd_scores["population"]["auroc"][pid] if isinstance(pd_scores.get("population"), dict) and isinstance(pd_scores["population"].get("auroc"), dict) else None,
        "personalized_auroc": pd_scores["personalized"]["auroc"][pid] if isinstance(pd_scores.get("personalized"), dict) and isinstance(pd_scores["personalized"].get("auroc"), dict) else None,
        "delta": pd_scores["delta"]["auroc"][pid] if isinstance(pd_scores.get("delta"), dict) and isinstance(pd_scores["delta"].get("auroc"), dict) else None,
    })

exp3 = {
    "status": "complete",
    "provenance": "real",
    "gate_verdict": "CONFIRMATORY",
    "design": e3["design"],
    "unit_of_analysis": {"n_dyads": e3["unit_of_analysis_resolution"]["n_dyads"]},
    "per_participant": per_participant,
    "per_dyad": per_dyad,
    "test": exp3_pairedtest,
    "confidence_intervals": {"between_dyad_ci95_auroc": e3["confidence_intervals"]["between_dyad_ci95_auroc"]},
    "verdict": "H2 not supported",
    "exclusions": e3["participant_reconciliation"]["excluded_participants"],
}

# ---------------------------------------------------------------------------
# experiments.exp4 -- real, the central experiment
# ---------------------------------------------------------------------------
e4 = exp4_raw["experiments"]["exp4"]

def to_paired_test(block):
    return {
        "n": block["n"], "dyad_ids": block["dyad_ids"], "deltas": block["deltas"],
        "median_delta": block["median_delta"], "n_positive": block["n_positive"],
        "n_negative": block["n_negative"], "n_ties_excluded": block.get("n_ties_excluded", 0),
        "sign_test_p": block["sign_test_p"], "permutation_p": block["permutation_p"],
        "n_signflip": block["n_signflip"], "ci95": block["ci95"],
    }

exp4_per_dyad = []
for d in e4["per_dyad"]:
    entry = {"pair_id": d["pair_id"], "tested_participant": d["tested_participant"], "partner": d["partner"],
              "held_out_seq_min": d["held_out_seq_min"], "held_out_seq_max": d["held_out_seq_max"], "n_test": d["n_test"]}
    for cond in ["universal", "person_specific", "dyad_specific", "n_matched", "person_other_dyad"]:
        c = d[cond]
        entry[cond] = {"auroc": c["auroc"], "balanced_accuracy": c["balanced_accuracy"], "f1": c["f1"],
                        "precision": c["precision"], "recall": c["recall"], "n_train": c["n_train"]}
    m = d["majority"]
    entry["majority"] = {"auroc": m["auroc"], "balanced_accuracy": m["balanced_accuracy"], "f1": m["f1"],
                          "precision": m["precision"], "recall": m["recall"]}
    exp4_per_dyad.append(entry)
assert len(exp4_per_dyad) == 10

exp4_tests = {k: to_paired_test(v["auroc"]) for k, v in e4["tests"].items() if isinstance(v, dict) and "auroc" in v}
exp4 = {
    "status": "complete",
    "provenance": "real",
    "gate_verdict": "CONFIRMATORY",
    "design": e4["design"],
    "per_dyad": exp4_per_dyad,
    "gains": {k: v["auroc"] for k, v in e4["gains"].items()},
    "tests": dict(exp4_tests, primary=e4["tests"]["primary"], secondary=e4["tests"]["secondary"],
                  tertiary=e4["tests"]["tertiary"], multiple_comparisons_note=e4["tests"]["multiple_comparisons_note"]),
    "nfi": e4["nfi"],
    "verdict": "H3 not supported",
    "cross_experiment": e4["cross_experiment"],
}

# ---------------------------------------------------------------------------
# experiments.exp5-exp8 -- placeholder, shapes designed in the plan
# ---------------------------------------------------------------------------
exp5_pids = e3["participant_reconciliation"]["resulting_dyads"]  # n=11
train_sizes = [60, 120, 240, 360]
exp5_per_dyad = []
exp5_deltas = []
for pid in exp5_pids:
    curve = [{"n_train_trials": n, "auroc": round(rng.uniform(0.47, 0.56), 4), "ci95": {}} for n in train_sizes]
    control = [{"n_train_trials": n, "auroc": round(rng.uniform(0.47, 0.55), 4), "ci95": {}} for n in train_sizes]
    delta = round(curve[-1]["auroc"] - curve[0]["auroc"], 4)
    exp5_deltas.append(delta)
    exp5_per_dyad.append({"pair_id": pid, "curve": curve, "control_curve_other_dyad": control})

def build_paired_test(pids, deltas, seed_note=""):
    n = len(pids)
    n_pos = sum(1 for d in deltas if d > 0)
    n_neg = sum(1 for d in deltas if d < 0)
    sorted_d = sorted(deltas)
    median = sorted_d[n // 2] if n % 2 else (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2
    return {
        "n": n, "dyad_ids": pids, "deltas": [round(d, 4) for d in deltas],
        "median_delta": round(median, 4), "n_positive": n_pos, "n_negative": n_neg,
        "n_ties_excluded": n - n_pos - n_neg,
        "sign_test_p": round(rng.uniform(0.08, 0.55), 4),
        "permutation_p": round(rng.uniform(0.08, 0.55), 4),
        "n_signflip": 10000,
        "ci95": {"mean": round(sum(deltas) / n, 4), "lower": round(min(deltas), 4), "upper": round(max(deltas), 4), "sd": round((sum((d - sum(deltas)/n)**2 for d in deltas) / n) ** 0.5, 4)},
    }

exp5 = {
    "status": "not_started", "provenance": "placeholder", "gate_verdict": "CONFIRMATORY",
    "design": {"research_question": "§15: does within-dyad prediction improve as more of that dyad's own history accumulates in training (a learning curve), beyond what the same amount of a different dyad's history would give?"},
    "per_dyad": exp5_per_dyad,
    "train_sizes": train_sizes,
    "test": build_paired_test(exp5_pids, exp5_deltas),
    "control_note": "same-amount same-dyad history vs other-dyad history (§15's required control)",
}

exp6_per_dyad = []
exp6_deltas = []
for pid in exp5_pids:
    early = {"auroc": round(rng.uniform(0.47, 0.55), 4)}
    late = {"auroc": round(early["auroc"] + rng.uniform(-0.03, 0.03), 4)}
    delta = round(late["auroc"] - early["auroc"], 4)
    exp6_deltas.append(delta)
    exp6_per_dyad.append({
        "pair_id": pid,
        "early": {**early, "balanced_accuracy": round(early["auroc"] - 0.01, 4), "f1": round(rng.uniform(0.4, 0.5), 4), "precision": round(rng.uniform(0.45, 0.55), 4), "recall": round(rng.uniform(0.4, 0.5), 4)},
        "middle": {"auroc": round(rng.uniform(0.47, 0.55), 4), "balanced_accuracy": round(rng.uniform(0.47, 0.53), 4), "f1": round(rng.uniform(0.4, 0.5), 4), "precision": round(rng.uniform(0.45, 0.55), 4), "recall": round(rng.uniform(0.4, 0.5), 4)},
        "late": {**late, "balanced_accuracy": round(late["auroc"] - 0.01, 4), "f1": round(rng.uniform(0.4, 0.5), 4), "precision": round(rng.uniform(0.45, 0.55), 4), "recall": round(rng.uniform(0.4, 0.5), 4)},
        "delta_late_minus_early": delta,
    })
exp6 = {
    "status": "not_started", "provenance": "placeholder", "gate_verdict": "CONFIRMATORY",
    "design": {"research_question": "§16: does the observer's own neural response to deception become more informative from early to late in the interaction?"},
    "per_dyad": exp6_per_dyad,
    "test": build_paired_test(exp5_pids, exp6_deltas),
}

input_sets = ["deceiver_eeg", "observer_eeg", "both_brains", "interbrain", "eeg_plus_behavioral"]
input_set_labels = {"deceiver_eeg": "Deceiver's brain", "observer_eeg": "Observer's brain",
                     "both_brains": "Both brains", "interbrain": "Inter-brain features",
                     "eeg_plus_behavioral": "EEG + behavioral history"}
exp7_per_dyad = []
per_input_scores = {s: [] for s in input_sets}
for pid in exp5_pids:
    scores = {}
    for s in input_sets:
        v = round(rng.uniform(0.47, 0.56), 4)
        scores[s] = {"auroc": v, "balanced_accuracy": round(v - 0.01, 4), "f1": round(rng.uniform(0.4, 0.5), 4),
                     "precision": round(rng.uniform(0.45, 0.55), 4), "recall": round(rng.uniform(0.4, 0.5), 4)}
        per_input_scores[s].append(v)
    exp7_per_dyad.append({"pair_id": pid, "scores": scores})

def paired_from_pair(a_key, b_key):
    deltas = [d["scores"][a_key]["auroc"] - d["scores"][b_key]["auroc"] for d in exp7_per_dyad]
    return build_paired_test(exp5_pids, deltas)

exp7 = {
    "status": "not_started", "provenance": "placeholder", "gate_verdict": "CONFIRMATORY",
    "design": {"research_question": "§17: does adding the observer's brain, or inter-brain features, improve prediction over the deceiver's brain alone?"},
    "input_sets": input_sets,
    "input_set_labels": input_set_labels,
    "per_dyad": exp7_per_dyad,
    "per_input_set": {
        s: {"median_auroc": round(sorted(per_input_scores[s])[len(per_input_scores[s]) // 2], 4),
            "ci95": {}, "n_dyads_above_chance": sum(1 for v in per_input_scores[s] if v > 0.5)}
        for s in input_sets
    },
    "tests": {
        "observer_eeg_vs_deceiver_eeg": paired_from_pair("observer_eeg", "deceiver_eeg"),
        "both_brains_vs_deceiver_eeg": paired_from_pair("both_brains", "deceiver_eeg"),
        "interbrain_vs_deceiver_eeg": paired_from_pair("interbrain", "deceiver_eeg"),
        "eeg_plus_behavioral_vs_deceiver_eeg": paired_from_pair("eeg_plus_behavioral", "deceiver_eeg"),
    },
}

WINDOW_EDGES = [(-1500, -1250), (-1250, -1000), (-1000, -750), (-750, -500), (-500, -250), (-250, 0)]
windows = [{"id": f"w_{a}_{b}".replace("-", "m"), "label": f"{a} to {b} ms", "start_ms": a, "end_ms": b} for a, b in WINDOW_EDGES]
def build_role_windows():
    out = []
    for w in windows:
        p = round(rng.uniform(0.05, 0.6), 3)
        out.append({"window": w["id"], "auroc": round(rng.uniform(0.47, 0.55), 4), "permutation_p": p,
                     "p_corrected": round(min(p * len(windows), 1.0), 3), "exceeds_null": p < 0.05})
    return out
exp8 = {
    "status": "not_started", "provenance": "placeholder", "gate_verdict": "CONFIRMATORY",
    "design": {"research_question": "§18: at what point before the decision does deception-related information first become decodable from EEG, for the deceiver and for the observer?"},
    "windows": windows,
    "per_role": {"deceiver": build_role_windows(), "observer": build_role_windows()},
    "onset": {"deceiver_ms": -750, "observer_ms": -500, "lag_ms": 250, "note": "Onset lag is descriptive; no causal interpretation is claimed (§18)."},
    "multiple_comparison_correction": "Benjamini-Hochberg FDR, q = 0.05",
}

experiments = {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4, "exp5": exp5, "exp6": exp6, "exp7": exp7, "exp8": exp8}

# ---------------------------------------------------------------------------
# tests -- one entry per claim
# ---------------------------------------------------------------------------
tests = {
    "exp1_above_chance": {
        "claim": "Pooled deception classification exceeds chance.",
        "hypothesis": "N/A", "experiment": "exp1", "metric": "auroc", "kind": "permutation",
        "designation": "exploratory", "primary": False, "supported": True,
        "result": e1["permutation_null"],
        "plain_language": "Pooled across all dyads, the model predicted deception better than random guessing, but only slightly.",
        "technical": f"Observed AUROC = {e1['models']['logistic_regression']['mean']['auroc']:.4f} vs permutation null mean {e1['permutation_null']['null']['mean']:.4f} (sd {e1['permutation_null']['null']['sd']:.4f}); p = {e1['permutation_null']['p_value']:.4f}.",
    },
    "h1_lodo_generalization": {
        "claim": "A model trained on other dyads generalizes to a held-out dyad (universal signature).",
        "hypothesis": "H1", "experiment": "exp2", "metric": "auroc", "kind": "paired_per_dyad",
        "designation": "confirmatory", "primary": False, "supported": False,
        "result": build_paired_test(exp2_dyad_ids, [round(v - EXP1_POOLED_AUROC, 4) for v in per_dyad_auroc.values()]),
        "plain_language": "A model trained on other pairs did not reliably predict deception in a new, unseen pair.",
        "technical": f"LODO mean AUROC {lodo_mean:.3f}, below exp1's pooled {EXP1_POOLED_AUROC:.3f} (placeholder pending exp2 completion).",
    },
    "h2_person_gain": {
        "claim": "Person-specific history improves prediction beyond a universal model (PersonGain).",
        "hypothesis": "H2", "experiment": "exp4", "metric": "auroc", "kind": "paired_per_dyad",
        "designation": "exploratory", "primary": False, "supported": False,
        "result": exp4_tests["person_gain"],
        "plain_language": "Knowing the person's own prior trials did not reliably improve prediction over a general model. Results were split evenly, 5 pairs better and 5 worse.",
        "technical": f"Median ΔAUROC = {e4['tests']['person_gain']['auroc']['median_delta']:.4f}; sign test p = {e4['tests']['person_gain']['auroc']['sign_test_p']:.4f}; permutation p = {e4['tests']['person_gain']['auroc']['permutation_p']:.4f}, n = 10 dyads.",
    },
    "exp3_personalized_vs_population": {
        "claim": "Personalized (own-history) models outperform population models at the participant level.",
        "hypothesis": "H2", "experiment": "exp3", "metric": "auroc", "kind": "paired_per_dyad",
        "designation": "confirmatory", "primary": False, "supported": False,
        "result": exp3_pairedtest,
        "plain_language": "Personalizing the model to an individual's own prior trials showed a small positive trend, but it was not statistically reliable.",
        "technical": f"Median ΔAUROC = {_dg3['median_delta']:.4f} (95% CI {_dg3['ci95']['lower']:.4f} to {_dg3['ci95']['upper']:.4f}); sign test p = {_dg3['sign_test_p']:.4f}; permutation p = {_dg3['permutation_p']:.4f}, n = 11 dyads.",
    },
    "h3_dyad_gain": {
        "claim": "Dyad-specific history improves deception prediction beyond person-specific history.",
        "hypothesis": "H3", "experiment": "exp4", "metric": "auroc", "kind": "paired_per_dyad",
        "designation": "confirmatory", "primary": True, "supported": False,
        "result": exp4_tests["dyad_gain"],
        "plain_language": "Knowing the specific pair did not make deception easier to predict. In 8 of 10 pairs it made prediction slightly worse.",
        "technical": f"Median ΔAUROC = {_dg['median_delta']:.4f} (95% CI {e4['tests']['dyad_gain']['auroc']['ci95']['lower']:.4f} to {e4['tests']['dyad_gain']['auroc']['ci95']['upper']:.4f}); sign test p = {_dg['sign_test_p']:.4f}; permutation p = {_dg['permutation_p']:.4f}, n = 10 dyads.",
    },
    "nfi_distribution": {
        "claim": "Net Familiarity Index (dyad-specific minus population) is positive across dyads.",
        "hypothesis": "H3", "experiment": "exp4", "metric": "auroc", "kind": "paired_per_dyad",
        "designation": "exploratory", "primary": False, "supported": False,
        "result": {**exp4_tests.get("nfi", {}), **e4["nfi"]},
        "plain_language": "Combining relationship-specific and general information did not consistently beat the general model alone. Only 3 of 10 pairs showed a positive combined benefit.",
        "technical": f"Median NFI = {e4['nfi']['median']:.4f}; sign test p = {e4['nfi']['sign_test_p']:.4f}; n = {e4['nfi']['n']} dyads.",
    },
    "exp5_learning_curve": {
        "claim": "Prediction improves as more of a dyad's own history accumulates in training.",
        "hypothesis": "H3", "experiment": "exp5", "metric": "auroc", "kind": "paired_per_dyad",
        "designation": "confirmatory", "primary": False, "supported": None,
        "result": exp5["test"],
        "plain_language": "Not yet run.",
        "technical": "Placeholder pending exp5.",
    },
    "exp6_observer_early_late": {
        "claim": "The observer's neural response to deception becomes more informative over the course of the interaction.",
        "hypothesis": "H3", "experiment": "exp6", "metric": "auroc", "kind": "paired_per_dyad",
        "designation": "confirmatory", "primary": False, "supported": None,
        "result": exp6["test"],
        "plain_language": "Not yet run.",
        "technical": "Placeholder pending exp6.",
    },
    "exp7_observer_vs_deceiver": {
        "claim": "Adding the observer's brain (or inter-brain features) improves prediction over the deceiver's brain alone.",
        "hypothesis": "H1", "experiment": "exp7", "metric": "auroc", "kind": "paired_per_dyad",
        "designation": "confirmatory", "primary": False, "supported": None,
        "result": exp7["tests"]["both_brains_vs_deceiver_eeg"],
        "plain_language": "Not yet run.",
        "technical": "Placeholder pending exp7.",
    },
    "exp8_onset_lag": {
        "claim": "Deception-related information becomes decodable at a specific, consistent point before the decision.",
        "hypothesis": "N/A", "experiment": "exp8", "metric": "auroc", "kind": "permutation",
        "designation": "confirmatory", "primary": False, "supported": None,
        "result": {"windows": exp8["windows"], "onset": exp8["onset"]},
        "plain_language": "Not yet run.",
        "technical": "Placeholder pending exp8.",
    },
}

# ---------------------------------------------------------------------------
# interpretability -- real electrode geometry, placeholder values (S25/S34)
# ---------------------------------------------------------------------------
# 30-channel montage (src/features.py REGIONS, verified as the archive's exact 30
# channels by build_channel_order()'s own assertion). Coordinates are an approximate
# 10-20 layout normalized to a 0-1 unit square, nose at y=0 (top).
ELECTRODE_COORDS = {
    "Fp2": (0.60, 0.06), "F9": (0.06, 0.30), "F10": (0.94, 0.30),
    "F7": (0.18, 0.24), "F3": (0.36, 0.20), "Fz": (0.50, 0.18), "F4": (0.64, 0.20), "F8": (0.82, 0.24),
    "FC5": (0.26, 0.36), "FC1": (0.42, 0.34), "FC2": (0.58, 0.34), "FC6": (0.74, 0.36),
    "C3": (0.32, 0.50), "Cz": (0.50, 0.50), "C4": (0.68, 0.50),
    "T7": (0.10, 0.50), "T8": (0.90, 0.50),
    "CP5": (0.26, 0.64), "CP6": (0.74, 0.64),
    "CP1": (0.42, 0.66), "CP2": (0.58, 0.66),
    "P7": (0.18, 0.76), "P3": (0.36, 0.80), "Pz": (0.50, 0.82), "P4": (0.64, 0.80), "P8": (0.82, 0.76),
    "PO3": (0.40, 0.90), "PO4": (0.60, 0.90),
    "O1": (0.42, 0.97), "O2": (0.58, 0.97),
}
assert len(ELECTRODE_COORDS) == 30
electrodes = [{"name": n, "x": round(x, 3), "y": round(y, 3)} for n, (x, y) in ELECTRODE_COORDS.items()]
conditions = ["truth", "deception", "early", "late", "deceiver", "observer"]
scalp_maps = {c: {n: round(rng.uniform(-0.20, 0.20), 3) for n in ELECTRODE_COORDS} for c in conditions}
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
WINDOWS_INTERP = ["pre", "onset", "post"]
top_features = {}
for c in conditions:
    feats = []
    for _ in range(8):
        ch = rng.choice(list(ELECTRODE_COORDS))
        band = rng.choice(BANDS)
        win = rng.choice(WINDOWS_INTERP)
        w = round(rng.uniform(-0.18, 0.18), 3)
        feats.append({"feature": f"F{ch}_{band}_{win}" if False else f"{band}_{ch}_{win}",
                      "channel": ch, "band": band, "window": win, "weight": w,
                      "direction": "positive" if w >= 0 else "negative"})
    top_features[c] = sorted(feats, key=lambda f: -abs(f["weight"]))

interpretability = {
    "provenance": "placeholder",
    "method": "logistic regression coefficients (L2), standardized features",
    "conditions": conditions,
    "electrodes": electrodes,
    "scalp_maps": scalp_maps,
    "top_features": top_features,
}

# ---------------------------------------------------------------------------
# interbrain -- placeholder, real caveat text (S36)
# ---------------------------------------------------------------------------
interbrain_nodes = []
for n, (x, y) in ELECTRODE_COORDS.items():
    interbrain_nodes.append({"id": f"d_{n}", "side": "deceiver", "label": n, "x": round(x * 0.42, 3), "y": round(y, 3)})
    interbrain_nodes.append({"id": f"o_{n}", "side": "observer", "label": n, "x": round(0.58 + x * 0.42, 3), "y": round(y, 3)})

def build_state_edges():
    edges = []
    core = ["Fz", "Cz", "Pz", "F3", "F4", "C3", "C4", "P3", "P4"]
    for a in core:
        for b in core:
            if rng.random() < 0.18:
                edges.append({"source": f"d_{a}", "target": f"o_{b}", "weight": round(rng.uniform(0.05, 0.35), 3)})
    return edges

interbrain = {
    "provenance": "placeholder",
    "caveat": (
        "These edges are statistical relationships between two simultaneously recorded "
        "EEG signals. They are not evidence of communication between brains. Inter-brain "
        "analyses in this project are descriptive."
    ),
    "gate_note": "Inter-brain comparisons have not been run through the §7 gate; treat this figure as underpowered and exploratory until §26's analysis completes.",
    "metric": "plv",
    "nodes": interbrain_nodes,
    "states": {
        "early_truth": {"edges": build_state_edges()},
        "early_deception": {"edges": build_state_edges()},
        "late_truth": {"edges": build_state_edges()},
        "late_deception": {"edges": build_state_edges()},
    },
}

# ---------------------------------------------------------------------------
# trials -- 12 dyads x 2 roles x 24 slider positions = 576, placeholder values
# ---------------------------------------------------------------------------
trials = []
model_levels = ["universal", "person_specific", "dyad_specific"]
for dyad in dyads:
    pid = dyad["pair_id"]
    a, b = dyad["participants"]
    lo, hi = dyad["seq_range"]["min"], dyad["seq_range"]["max"]
    for role, participant, partner in [("deceiver", a, b), ("observer", b, a)]:
        for pos in range(24):
            frac = pos / 23
            seq = round(lo + frac * (hi - lo))
            round_no = ((seq - 1) // 44) + 1 if seq >= 1 else 1
            round_no = min(round_no, 11)
            p_lie = round(rng.uniform(0.35, 0.65), 3)
            condition = "lie" if rng.random() < 0.475 else "truth"
            predicted = "lie" if p_lie >= 0.5 else "truth"
            state = "early" if pos < 12 else "late"
            cond2 = "deception" if condition == "lie" else "truth"
            trials.append({
                "trial_uid": f"{pid}:{role}:pos{pos:02d}",
                "pair_id": pid, "participant_id": participant, "partner_id": partner,
                "role": role, "slider_pos": pos, "session_order": 1 if seq <= (hi // 2 if hi > 484 else hi) else 2,
                "round": round_no, "dyad_trial_seq": seq, "condition": condition,
                "outcome": rng.choice(["success", "failure"]),
                "observer_guess": rng.choice(["lie", "truth"]),
                "reaction_time_sec": round(rng.uniform(0.8, 3.5), 2),
                "prediction": {"model_level": rng.choice(model_levels), "p_lie": p_lie, "predicted": predicted, "correct": predicted == condition},
                "top_features": [{"feature": f"theta_{rng.choice(list(ELECTRODE_COORDS))}_pre", "contribution": round(rng.uniform(0.02, 0.15), 3)} for _ in range(3)],
                "interbrain_state_ref": f"{state}_{cond2}",
            })
assert len(trials) == 576

# ---------------------------------------------------------------------------
# failures -- 6 entries, 3 FP + 3 FN, placeholder
# ---------------------------------------------------------------------------
failures = []
fp_pool = [t for t in trials if t["prediction"]["predicted"] == "lie" and t["condition"] == "truth"]
fn_pool = [t for t in trials if t["prediction"]["predicted"] == "truth" and t["condition"] == "lie"]
rng.shuffle(fp_pool)
rng.shuffle(fn_pool)
for t in fp_pool[:3]:
    failures.append({
        "trial_uid": t["trial_uid"], "pair_id": t["pair_id"], "actual": t["condition"],
        "predicted": t["prediction"]["predicted"], "p_lie": t["prediction"]["p_lie"], "kind": "false_positive",
        "attribution": t["top_features"], "note": "Model predicted deception on a truthful trial.",
    })
for t in fn_pool[:3]:
    failures.append({
        "trial_uid": t["trial_uid"], "pair_id": t["pair_id"], "actual": t["condition"],
        "predicted": t["prediction"]["predicted"], "p_lie": t["prediction"]["p_lie"], "kind": "false_negative",
        "attribution": t["top_features"], "note": "Model missed a deceptive trial, predicting truth.",
    })
assert len(failures) == 6
trial_uids = {t["trial_uid"] for t in trials}
for f in failures:
    assert f["trial_uid"] in trial_uids

# ---------------------------------------------------------------------------
# assemble & write
# ---------------------------------------------------------------------------
fixture = {
    "meta": meta, "gate": gate, "frozen": frozen, "dyads": dyads,
    "experiments": experiments, "tests": tests, "interpretability": interpretability,
    "interbrain": interbrain, "trials": trials, "failures": failures,
}

# validate a handful of interbrain_state_ref values resolve
state_keys = set(interbrain["states"].keys())
for t in trials:
    assert t["interbrain_state_ref"] in state_keys, t["interbrain_state_ref"]

out_path = RESULTS / "fixtures" / "results.v1.fixture.json"
out_path.write_text(json.dumps(fixture, indent=2))
print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
print(f"dyads={len(dyads)} exp4.per_dyad={len(exp4_per_dyad)} exp2.per_dyad={len(exp2_per_dyad)} trials={len(trials)} failures={len(failures)}")
