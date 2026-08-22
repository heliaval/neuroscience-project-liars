"""
src/experiments/exp6_observer.py -- Experiment 6 driver (S16): observer-only prediction.

Research question (S16): does the receiver's (observer's) own EEG carry
information about whether their partner is currently deceiving them, and does
that information grow across the interaction? This module answers it two ways:

  1. Absolute decodability (S21) -- for every scored bin, is a Universal
     observer model's AUROC above a label-shuffled null? Run and reported
     FIRST (see Fact 5).
  2. Positional comparison (S20) -- is a Universal observer model more
     accurate on later trials than earlier ones? Two grains, reused unchanged
     from exp5's Analysis 1 design: 1A (dyad-grain terciles, confirmatory on
     power, confounded with participant identity in the opposite direction to
     exp5 -- see Fact 2) and 1B (participant-grain terciles, unconfounded but
     underpowered/exploratory, per Amendment 3).

Structurally cloned from src/experiments/exp5_history.py, which was itself
cloned from exp4_dyadic.py: same shard-loader pattern, same helpers copied
verbatim (never imported across experiment modules), same per-unit
checkpointing, same JSON+MD output shape. src/models.py is not modified.

--------------------------------------------------------------------------------
FACT 1 -- the observer frame is the exact mirror of the deceiver frame
--------------------------------------------------------------------------------
single_brain.parquet is 21,296 rows x 2,000 columns: 10,648 role=="deceiver"
rows and 10,648 role=="observer" rows, one of each per trial, identical
per-dyad row counts and dyad_trial_seq ranges. The observer row and the
deceiver row for a given trial carry the SAME condition label -- the target is
the partner's ground-truth truth/lie, a property of the trial, not the row's
role. exp6's target is therefore identical to every other experiment's target
and is NOT observer_guess; observer_guess/outcome/points stay on the
forbidden-column list (validation 8).

--------------------------------------------------------------------------------
FACT 2 -- the identity confound is INVERTED relative to exp5, and that is useful
--------------------------------------------------------------------------------
Every participant is the deceiver in exactly one session, in disjoint
contiguous blocks (1-484=deceiver A's session, 485-968=deceiver B's session).
Because roles alternate within a dyad, the OBSERVER of seq 1-484 is B (the
session-2 deceiver) and the observer of seq 485-968 is A (the session-1
deceiver) -- the reverse of exp5's identity assignment on the same seq bins.
exp5's and exp6's positional analyses score the SAME trials in the SAME seq
bins, but the brain being scored belongs to the opposite participant in the
two experiments. A same-signed effect in both cannot be explained by one
particular participant being more readable (the participants swap sides), but
CAN still be explained by anything positional (session order, fatigue,
electrode drift, task familiarity) -- exactly what S16 is asking about, and
exactly why 1B still matters. Reported as descriptive corroboration only, via
cross_experiment; exp7 (S17) owns the tested observer-vs-deceiver comparison.

--------------------------------------------------------------------------------
FACT 3 -- exp6 is n=10, not the pre-registered n=11 (Amendment 3, pre-results)
--------------------------------------------------------------------------------
sub19_sub22's observer rows span seq [1,484] only (its only session), so its
late tercile [647,968] has ZERO observer rows -- the same structural situation
Amendment 1 resolved for exp4 and Amendment 2 for exp5, on the observer side.
results/frozen_hypotheses.md Amendment 3 (2026-08-22, pre-results) reduces
exp6's n from 11 to 10 accordingly; sub19_sub22's 484 observer rows remain in
the other-dyad/Universal training pool. sub01_sub02 stays excluded from
exp3-8 as already frozen. Tested dyads are exp4's/exp5's ten, unchanged:

    sub03_sub06  sub04_sub05  sub07_sub08  sub09_sub10  sub11_sub12
    sub13_sub14  sub15_sub16  sub17_sub18  sub20_sub21  sub23_sub24

--------------------------------------------------------------------------------
FACT 4 -- exp6's own gate re-check lands on EXACTLY exp5's numbers
--------------------------------------------------------------------------------
gate.json's exp6 row (161/70) is exp3's participant-grain numbers for the
now-excluded sub01_sub02 and does not cover exp6's real grains. Per Amendment
3, THRESHOLD_M=60 and Clause A/B are re-applied UNCHANGED at exp6's actual
grain. The six NaN-dropped rows are the SAME six trials in both roles
(sub20_sub21 seq 309/310; sub23_sub24 seq 221/222/397/398), so every per-bin
n and minority count is identical to exp5's -- verified in-environment by
validation 4's mirror check, not merely asserted.

  | analysis | test-fold grain      | Clause A | Clause B  | designation |
  |----------|-----------------------|----------|-----------|-------------|
  | 1A       | 320-324, min 130      | PASS     | 10/10 PASS| CONFIRMATORY on power, confound stated in headline |
  | 1B       | 162/161/161, min 45   | FAIL     | 8/10 FAIL | EXPLORATORY -- underpowered |
  | 2 (S21)  | same bins as 1A       | PASS     | 10/10 PASS| CONFIRMATORY (permutation test, not S20) |

1B fails on sub23 (45, middle within-observer bin) and sub18 (59, middle
within-observer bin) -- the partners of exp5's failing participants (sub24,
sub17) on the same seq blocks, exactly as Fact 2 predicts. Kept and reported
per S7's rule: demote, don't drop.

--------------------------------------------------------------------------------
FACT 5 -- the prior expectation, from exp1's already-run observer secondary
--------------------------------------------------------------------------------
results/exp1_baseline.json's experiments.exp1.observer_rows is the same
extraction exp6 needs (same prepare_modeling_frame(sb,"observer",...), same
1,770 columns, same 6 NaN drops), under grouped 5-fold: mean AUROC 0.5368
[0.5285, 0.5451]. Compare exp1's deceiver-row primary: pooled AUROC 0.5338,
permutation p=0.0050. Two consequences, pre-declared: (1) observer-only
decodability is expected to be real but tiny, comparable to the deceiver-side
signal -- any bin AUROC above 0.75 is a leakage flag, not a finding
(validation 17); (2) the early/late comparison is a comparison of two
near-chance numbers, so the S21 absolute-decodability null is run and
reported FIRST, fixed here before any number exists.

--------------------------------------------------------------------------------
CLAIM HIERARCHY (declared before any number exists, Amendment 3)
--------------------------------------------------------------------------------
  PRIMARY:     observer_positional_late_minus_early (S20 paired, n=10). No correction.
  SECONDARY:   observer_decodability_pooled (S21 permutation null, pooled). No correction -- pre-registered.
  EXPLORATORY: observer_within_person_late_minus_early; observer_positional_middle_minus_early;
               observer_positional_late_minus_middle; the 30 per-bin S21 nulls (dyad-grain).
               Benjamini-Hochberg, alpha=0.05, within this family.

observer_within_person_late_minus_early is exploratory on two independent
grounds: the gate failure above, and the fact that it is not in the frozen
design. `supported` on the primary test is set ONLY from the pre-registered
rule (sign p<0.05 AND permutation p<0.05 AND median delta positive) -- never
from a reading of the numbers (see build_tests()).

--------------------------------------------------------------------------------
S16 INTERPRETATION GUARDRAILS -- BINDING, not advisory
--------------------------------------------------------------------------------
S16 closes with: "Avoid claiming subconscious lie detection unless behavioral
and statistical evidence directly supports it." No output string in this
module's .md or .json may claim detection or knowledge by the observer --
only statistical decodability of a signal by a model. BANNED_PHRASES below is
scanned recursively over every emitted string (validation 15) and the run
FAILS (raises, output not left in a claimed-final state) if any hit occurs.
The permitted register: "a linear model trained on other dyads' observer EEG
separated the partner's lie trials from truth trials at AUROC=X ... against a
label-shuffled null this is p=Y." Mundane alternatives exp6 cannot rule out
(the deceiver behaving observably differently and the observer's EEG tracking
that, not "reading minds"; task-structure differences; shared temporal
recording structure) are stated in Findings, not buried in Limitations. Four
evidentiary conditions that would license a stronger claim (behavioural
corroboration via observer_guess -- deliberately NOT built here; effect size;
cross-dyad consistency; robustness to behavioural controls) are listed in
`interpretation`. observer_guess/outcome/points never enter X (validation 8).

--------------------------------------------------------------------------------
S19 LEAKAGE RULES / PYTHON 3.8 HAZARD
--------------------------------------------------------------------------------
Every Universal training pool spans other dyads only (leave-this-dyad-out);
inner CV is M.default_grouped_inner(M.SEED, n_splits=3) grouped by pair_id.
No row belonging to the tested dyad ever appears in that dyad's training
pool. Remote is Python 3.8.10 -- NO `{...} | {...}` dict union (3.9+ syntax;
this crashed exp4 once at output assembly). Use `{**a, **b}` throughout.
Verified by pre-upload grep (see Task 3's checklist); `set | set` (validation
8's forbidden-column check) is fine -- only *dict* union is 3.9+.

--------------------------------------------------------------------------------
CHECKPOINTING
--------------------------------------------------------------------------------
exp6_checkpoints/<key>.pkl (relative to this machine's reveriehacks26/ root --
a NEW directory, never exp4's or exp5's). ~20 units total (10 Universal fits,
each caching all 9 bins' metrics+scores, plus 10 majority fits); the S21
permutation nulls and S20 tests are cheap enough to recompute from the cached
per-bin scores on every run rather than being separately checkpointed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from scipy import stats as sp_stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import models as M  # noqa: E402

FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "out"
EXP6_JSON = OUT_DIR / "exp6_observer.json"
EXP6_MD = OUT_DIR / "exp6_observer.md"
EXP5_JSON = OUT_DIR / "exp5_history.json"
GATE_JSON = RESULTS_DIR / "gate.json"
FROZEN_HYP_MD = RESULTS_DIR / "frozen_hypotheses.md"
OBS_SHARD_MANIFEST = FEATURES_DIR / "shard_obs_manifest.json"

CHECKPOINT_DIR = REPO_ROOT / "exp6_checkpoints"

THRESHOLD_M = 60
N_SIGNFLIP = 10000
N_LABEL_PERM = 10000
METRICS = ["auroc", "balanced_accuracy", "f1", "precision", "recall"]
GRID_N_JOBS = 6
ROLE = "observer"

TESTED_DYADS = [
    "sub03_sub06", "sub04_sub05", "sub07_sub08", "sub09_sub10", "sub11_sub12",
    "sub13_sub14", "sub15_sub16", "sub17_sub18", "sub20_sub21", "sub23_sub24",
]
EXCLUDED_FROM_EVERYTHING = "sub01_sub02"
UNTESTABLE_UNIT_STILL_IN_POOL = "sub19_sub22"

DYAD_TERCILES = [("early", 1, 322), ("middle", 323, 646), ("late", 647, 968)]
PERSON_TERCILE_SIZES = (162, 161, 161)  # remainder to the earliest bin

# sha256 of results/frozen_hypotheses.md AFTER Amendment 3 was appended on the
# laptop (recorded immediately after the append, per plan Task 1 step 4).
FROZEN_HYP_SHA256_EXPECTED = "f312b0fbd33ec494013d7af03c5fba1d3deb753f7182a607c67fb55c445d05fe"

# sha256 of src/models.py exp5 recorded, per plan Task 5 validation 14: if this
# machine's copy differs, stop and report -- a drift there breaks comparability.
MODELS_PY_SHA256_EXPECTED = "117ad942f6500edc839774f7f4d07cf4d68a179f18cc25b4de157bb45e43820e"

BANNED_PHRASES = [
    "subconscious lie detection", "subconscious detection", "unconscious detection",
    "the observer's brain knows", "gut feeling", "implicit detection",
    "detects lies", "detected the lie",
]

LAPTOP_ENV = {
    "python_version": "3.13.14", "sklearn_version": "1.8.0", "numpy_version": "2.3.2",
    "scipy_version": "1.17.1", "pandas_version": "2.3.2", "pyarrow_version": "25.0.1",
    "xgboost_version": "3.4.1",
}


# ---------------------------------------------------------------------------
# Checkpointing (copied from exp5_history.py / exp4_dyadic.py)
# ---------------------------------------------------------------------------

def _ckpt_path(key: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{key}.pkl"


def _ckpt_load(key: str):
    p = _ckpt_path(key)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _ckpt_save(key: str, obj):
    p = _ckpt_path(key)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------------
# Data loading -- observer shard reassembly (never calls M.load_single_brain())
# Clone of exp4/exp5's load_deceiver_frame(), reading the observer manifest.
# ---------------------------------------------------------------------------

def load_observer_frame():
    manifest = json.loads(OBS_SHARD_MANIFEST.read_text())
    frames = []
    for pid in manifest["shards"]:
        path = FEATURES_DIR / f"shard_obs_{pid}.parquet"
        frames.append(pd.read_parquet(path))
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("_orig_row", kind="mergesort").reset_index(drop=True)
    assert len(full) == manifest["n_rows_total"], (len(full), manifest["n_rows_total"])
    th = hashlib.sha256(pd.util.hash_pandas_object(full, index=False).values.tobytes()).hexdigest()
    assert th == manifest["total_frame_sha256"], "assembled frame hash mismatch -- data integrity failed"
    full = full.drop(columns=["_orig_row"])
    assert (full["role"] == ROLE).all()
    assert full.shape[0] == 10648 and full.shape[1] == manifest["n_cols"] - 1, (
        full.shape, manifest["n_cols"])
    return full, manifest


# ---------------------------------------------------------------------------
# Pool/bin helpers (copied from exp5_history.py)
# ---------------------------------------------------------------------------

def universal_pool_idx(row_keys: pd.DataFrame, exclude_dyad: str) -> np.ndarray:
    pair_id_arr = row_keys["pair_id"].values
    exclude = {exclude_dyad, EXCLUDED_FROM_EVERYTHING}
    mask = ~np.isin(pair_id_arr, list(exclude))
    return np.where(mask)[0]


def sort_by_seq(idx: np.ndarray, seq_arr: np.ndarray) -> np.ndarray:
    order = np.argsort(seq_arr[idx], kind="mergesort")
    return idx[order]


def _contig_bins(idx_sorted_by_seq: np.ndarray, n_bins: int) -> list:
    """Split a seq-sorted index array into n_bins contiguous pieces, remainder
    to the earliest bins (gate.json's frozen split_definition convention)."""
    n = len(idx_sorted_by_seq)
    base = n // n_bins
    rem = n % n_bins
    bins = []
    start = 0
    for i in range(n_bins):
        size = base + (1 if i < rem else 0)
        bins.append(idx_sorted_by_seq[start:start + size])
        start += size
    return bins


def build_observer_bins(row_keys: pd.DataFrame) -> dict:
    """Per dyad: A (observer of seq[1,484] = session-2 deceiver, Fact 2), B
    (observer of seq[485,968] = session-1 deceiver), dyad_tercile_bins (3
    dyad-grain bins over the full [1,968] range), person_tercile_bins (6
    bins: A's own 484-row observer block in 3 pieces, then B's, in 3)."""
    blocks = {}
    seq_full = row_keys["dyad_trial_seq"].values  # GLOBAL row position -- sort_by_seq needs this
    for d in TESTED_DYADS:
        dyad_mask = row_keys["pair_id"].values == d
        dyad_idx = np.where(dyad_mask)[0]
        seq = seq_full[dyad_idx]
        pid = row_keys["participant_id"].values[dyad_idx]

        early_mask = seq <= 484
        A_candidates = set(pid[early_mask].tolist())
        assert len(A_candidates) == 1, f"{d}: early observer block spans >1 participant: {A_candidates}"
        A = next(iter(A_candidates))

        late_mask = seq >= 485
        B_candidates = set(pid[late_mask].tolist())
        assert len(B_candidates) == 1, f"{d}: late observer block spans >1 participant: {B_candidates}"
        B = next(iter(B_candidates))

        all_sorted = sort_by_seq(dyad_idx, seq_full)
        dyad_tercile_bins = _contig_bins(all_sorted, 3)

        a_block_idx = dyad_idx[(pid == A) & (seq <= 484)]
        b_block_idx = dyad_idx[(pid == B) & (seq >= 485)]
        a_sorted = sort_by_seq(a_block_idx, seq_full)
        b_sorted = sort_by_seq(b_block_idx, seq_full)
        person_tercile_bins = _contig_bins(a_sorted, 3) + _contig_bins(b_sorted, 3)

        assert 470 <= len(a_block_idx) <= 484, f"{d}: A observer block size {len(a_block_idx)} implausible"
        assert 470 <= len(b_block_idx) <= 484, f"{d}: B observer block size {len(b_block_idx)} implausible"

        blocks[d] = {
            "A": A, "B": B,
            "dyad_tercile_bins": dyad_tercile_bins,
            "person_tercile_bins": person_tercile_bins,
            "a_block_idx": a_block_idx, "b_block_idx": b_block_idx,
        }
    return blocks


# ---------------------------------------------------------------------------
# S20 paired test machinery (copied verbatim from exp5_history.py -- not
# imported across experiment modules; src/models.py not modified)
# ---------------------------------------------------------------------------

def paired_sign_test(deltas: list) -> dict:
    n_ties = sum(1 for d in deltas if d == 0.0)
    nz = [d for d in deltas if d != 0.0]
    n = len(nz)
    n_pos = sum(1 for d in nz if d > 0)
    n_neg = n - n_pos
    k = min(n_pos, n_neg)
    p = float(sp_stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue) if n > 0 else float("nan")
    return {"n": n, "n_ties_excluded": n_ties, "n_positive": n_pos, "n_negative": n_neg, "p": p}


def signflip_permutation_test(deltas: list, n_draws: int, seed: int) -> dict:
    arr = np.asarray(deltas, dtype=float)
    observed_median = float(np.median(arr))
    rng = np.random.default_rng(seed)
    n = len(arr)
    signs = rng.choice([-1.0, 1.0], size=(n_draws, n))
    draws_median = np.median(arr[None, :] * signs, axis=1)
    count = int(np.sum(np.abs(draws_median) >= abs(observed_median)))
    p = (count + 1) / (n_draws + 1)
    return {"observed_median": observed_median, "n_draws": n_draws, "p": float(p)}


def _benjamini_hochberg(pvals: list, alpha: float = 0.05) -> list:
    """Returns BH-adjusted p-values (same order as input), standard step-up."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj_sorted = np.minimum.accumulate(adj[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    out = np.empty(n)
    out[order] = adj_sorted
    return out.tolist()


# ---------------------------------------------------------------------------
# Hanley-McNeil (1982) analytic SE, for a per-bin CI95 on a single held-out
# AUROC point estimate. Copied in form (not imported) from src/gate.py's
# hanley_mcneil_se, which uses the identical formula to derive THRESHOLD_M --
# reused here as the project's own established device for this exact
# situation (a point AUROC with no fold-level replicates to bootstrap a CI
# from), not invented for this module.
# ---------------------------------------------------------------------------

def hanley_mcneil_se(auc: float, n1: int, n2: int) -> float:
    if n1 <= 1 or n2 <= 1 or auc <= 0.0 or auc >= 1.0:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2)
    return float(np.sqrt(max(var, 0.0)))


def bin_ci95(auc: float, n_pos: int, n_neg: int) -> dict:
    se = hanley_mcneil_se(auc, n_pos, n_neg)
    if np.isnan(se):
        return {"mean": auc, "lower": auc, "upper": auc, "sd": 0.0, "method": "hanley_mcneil_degenerate"}
    return {
        "mean": auc, "lower": max(0.0, auc - 1.96 * se), "upper": min(1.0, auc + 1.96 * se),
        "sd": se, "method": "hanley_mcneil_1982",
    }


# ---------------------------------------------------------------------------
# S21 fixed-model label-shuffling null (held-out y permuted, model NOT refit)
# -- exp1's device, reused in form. M.permutation_p_value is reused directly
# (a generic add-one estimator, not experiment-specific).
# ---------------------------------------------------------------------------

def label_permutation_null(y_true: np.ndarray, y_score: np.ndarray, n_perm: int, seed_seq) -> dict:
    observed = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed_seq)
    null_aucs = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        y_shuf = rng.permutation(y_true)
        null_aucs[i] = roc_auc_score(y_shuf, y_score)
    p = M.permutation_p_value(observed, null_aucs.tolist())
    return {
        "observed_auroc": observed,
        "null": {
            "mean": float(null_aucs.mean()), "sd": float(null_aucs.std(ddof=1)),
            "percentiles": {
                "5": float(np.percentile(null_aucs, 5)),
                "50": float(np.percentile(null_aucs, 50)),
                "95": float(np.percentile(null_aucs, 95)),
            },
        },
        "p_value": p, "n_permutations": n_perm,
    }


# ---------------------------------------------------------------------------
# GRID_N_JOBS invariance check (copied pattern from exp4/exp5)
# ---------------------------------------------------------------------------

def grid_n_jobs_check(Xv, yv, row_keys):
    d = TESTED_DYADS[0]
    train_idx = universal_pool_idx(row_keys, d)
    dyad_idx = np.where(row_keys["pair_id"].values == d)[0]
    test_idx = dyad_idx[:50]  # cheap sample, just to time/compare the fit
    inner = M.default_grouped_inner(M.SEED, n_splits=3)
    groups_train = row_keys["pair_id"].values[train_idx]

    def _one(n_jobs):
        pipe = M._lr_pipeline("l2", M.SEED)
        grid = M._lr_param_grid("l2")
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=n_jobs)
        t0 = time.time()
        search.fit(Xv[train_idx], yv[train_idx], groups=groups_train)
        dt = time.time() - t0
        fitted = search.best_estimator_
        clf = fitted.named_steps["clf"]
        Xt = fitted[:-1].transform(Xv[test_idx]) if len(fitted.steps) > 1 else Xv[test_idx]
        y_score = clf.predict_proba(Xt)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xt)
        metrics = M.compute_metrics(yv[test_idx], y_score, fitted.predict(Xv[test_idx]))
        return search.best_params_, metrics, dt

    bp1, m1, t1 = _one(1)
    bp2, m2, t2 = _one(GRID_N_JOBS)
    identical = (bp1 == bp2) and (m1 == m2)
    return {
        "dyad_sampled": d, "n_jobs_1_seconds": t1, "n_jobs_chosen_seconds": t2,
        "n_jobs_chosen": GRID_N_JOBS, "metrics_identical": identical,
        "best_params_n1": bp1, "best_params_nchosen": bp2,
    }


# ---------------------------------------------------------------------------
# Analysis 1 -- one Universal observer fit per dyad, scored on all 9 bins
# (10 units). Caches per-bin y_true/y_score so Analysis 2's S21 null needs no
# refit.
# ---------------------------------------------------------------------------

def _score_on_idx(Xv, yv, fitted_search, idx):
    fitted = fitted_search.best_estimator_
    clf = fitted.named_steps["clf"]
    Xt = fitted[:-1].transform(Xv[idx]) if len(fitted.steps) > 1 else Xv[idx]
    y_score = clf.predict_proba(Xt)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xt)
    y_pred = fitted.predict(Xv[idx])
    metrics = M.compute_metrics(yv[idx], y_score, y_pred)
    return metrics, y_score


def run_universal_and_score_bins(Xv, yv, row_keys, blocks, n_jobs):
    pair_id_arr = row_keys["pair_id"].values
    records = {}
    for d in TESTED_DYADS:
        key = f"univ_obs_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            records[d] = cached
            print(f"  [checkpoint] universal-observer {d}: loaded", flush=True)
            continue
        t0 = time.time()
        train_idx = universal_pool_idx(row_keys, d)
        groups_train = pair_id_arr[train_idx]
        inner = M.default_grouped_inner(M.SEED, n_splits=3)

        pipe = M._lr_pipeline("l2", M.SEED)
        grid = M._lr_param_grid("l2")
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=n_jobs)
        conv_warnings = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(Xv[train_idx], yv[train_idx], groups=groups_train)
            for warning in w:
                if issubclass(warning.category, ConvergenceWarning):
                    conv_warnings.append(str(warning.message))

        # 1A -- dyad-grain terciles (3 bins)
        positional_bins = []
        bin_names = ["early", "middle", "late"]
        for bname, bidx in zip(bin_names, blocks[d]["dyad_tercile_bins"]):
            seq_bin = row_keys["dyad_trial_seq"].values[bidx]
            pids_bin = sorted(set(row_keys["participant_id"].values[bidx].tolist()))
            m, y_score = _score_on_idx(Xv, yv, search, bidx)
            n_lie = int(yv[bidx].sum())
            n_minority = int(min(n_lie, len(bidx) - n_lie))
            positional_bins.append({
                "bin": bname, "seq_min": int(seq_bin.min()), "seq_max": int(seq_bin.max()),
                "n": int(len(bidx)), "n_minority": n_minority, "participants": pids_bin,
                **{m_: m[m_] for m_ in METRICS},
                "ci95": bin_ci95(m["auroc"], n_lie, len(bidx) - n_lie),
                "_y_true": yv[bidx].tolist(), "_y_score": y_score.tolist(),
            })

        # 1B -- participant-grain terciles (6 raw bins: A's 3, then B's 3)
        wp_bin_names = ["early", "middle", "late"]
        wp_bins_raw = []
        for person_label, offset in [("A", 0), ("B", 3)]:
            pid_val = blocks[d][person_label]
            for i in range(3):
                bidx = blocks[d]["person_tercile_bins"][offset + i]
                m, y_score = _score_on_idx(Xv, yv, search, bidx)
                n_lie = int(yv[bidx].sum())
                n_minority = int(min(n_lie, len(bidx) - n_lie))
                wp_bins_raw.append({
                    "participant": pid_val, "position": wp_bin_names[i],
                    "n": int(len(bidx)), "n_minority": n_minority,
                    **{m_: m[m_] for m_ in METRICS},
                    "_y_true": yv[bidx].tolist(), "_y_score": y_score.tolist(),
                })
        within_person_bins = []
        for i, pos in enumerate(wp_bin_names):
            a_rec = wp_bins_raw[i]
            b_rec = wp_bins_raw[3 + i]
            mean_metrics = {m_: float(np.mean([a_rec[m_], b_rec[m_]])) for m_ in METRICS}
            within_person_bins.append({
                "bin": pos, "n": a_rec["n"] + b_rec["n"],
                "n_minority": a_rec["n_minority"] + b_rec["n_minority"],
                "per_participant": [
                    {k_: v_ for k_, v_ in a_rec.items() if not k_.startswith("_")},
                    {k_: v_ for k_, v_ in b_rec.items() if not k_.startswith("_")},
                ],
                "mean_auroc": mean_metrics["auroc"], **mean_metrics,
            })

        rec = {
            "best_params": search.best_params_, "n_train": int(len(train_idx)),
            "convergence_warnings": conv_warnings, "inner_splitter_class": type(inner).__name__,
            "train_pair_ids": sorted(set(pair_id_arr[train_idx].tolist())),
            "positional_bins": positional_bins,
            "within_person_bins": within_person_bins,
            "wp_bins_raw": wp_bins_raw,
        }
        _ckpt_save(key, rec)
        records[d] = rec
        print(f"  universal-observer {d}: n_train={rec['n_train']} "
              f"late_auroc={positional_bins[2]['auroc']:.4f} ({time.time()-t0:.1f}s)", flush=True)
    return records


# ---------------------------------------------------------------------------
# Majority-class reference (10 units), scored on the same 9 bins
# ---------------------------------------------------------------------------

def run_majority(Xv, yv, row_keys, blocks):
    records = {}
    for d in TESTED_DYADS:
        key = f"major_obs_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            records[d] = cached
            continue
        train_idx = universal_pool_idx(row_keys, d)
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(Xv[train_idx], yv[train_idx])
        per_bin = {}
        for bname, bidx in zip(["early", "middle", "late"], blocks[d]["dyad_tercile_bins"]):
            y_score = clf.predict_proba(Xv[bidx])[:, 1]
            y_pred = clf.predict(Xv[bidx])
            per_bin[bname] = M.compute_metrics(yv[bidx], y_score, y_pred)
        _ckpt_save(key, per_bin)
        records[d] = per_bin
    return records


# ---------------------------------------------------------------------------
# Analysis 2 -- S21 label-shuffling nulls (no refit; reuses cached scores)
# ---------------------------------------------------------------------------

def run_label_permutation_nulls(univ_records):
    dyad_grain = {}
    person_grain = {}
    for dyad_index, d in enumerate(TESTED_DYADS):
        u = univ_records[d]
        dyad_grain[d] = {}
        for bin_index, pb in enumerate(u["positional_bins"]):
            null = label_permutation_null(
                np.array(pb["_y_true"]), np.array(pb["_y_score"]), N_LABEL_PERM,
                [M.SEED, dyad_index, bin_index],
            )
            dyad_grain[d][pb["bin"]] = null
        person_grain[d] = {}
        for bin_index, wb in enumerate(u["wp_bins_raw"]):
            key = f"{wb['participant']}_{wb['position']}"
            null = label_permutation_null(
                np.array(wb["_y_true"]), np.array(wb["_y_score"]), N_LABEL_PERM,
                [M.SEED, dyad_index, 100 + bin_index],
            )
            person_grain[d][key] = null

    # pooled entry over all ten dyads' concatenated dyad-grain bins
    all_y_true = np.concatenate([
        np.array(pb["_y_true"]) for d in TESTED_DYADS for pb in univ_records[d]["positional_bins"]
    ])
    all_y_score = np.concatenate([
        np.array(pb["_y_score"]) for d in TESTED_DYADS for pb in univ_records[d]["positional_bins"]
    ])
    pooled = label_permutation_null(all_y_true, all_y_score, N_LABEL_PERM, [M.SEED, 999, 0])
    return {"dyad_grain": dyad_grain, "person_grain": person_grain, "pooled": pooled}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _strip_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def assemble(blocks, univ_records, maj_records, null_records):
    per_dyad = []
    for d in TESTED_DYADS:
        u = univ_records[d]
        b = blocks[d]
        positional_bins_public = []
        for pb in u["positional_bins"]:
            entry = _strip_private(pb)
            entry["permutation_null"] = null_records["dyad_grain"][d][pb["bin"]]
            positional_bins_public.append(entry)
        early_auroc = positional_bins_public[0]["auroc"]
        late_auroc = positional_bins_public[2]["auroc"]
        row = {
            "pair_id": d,
            "observers": {"early_block": b["A"], "late_block": b["B"]},
            "positional_bins": positional_bins_public,
            "within_person_bins": u["within_person_bins"],
            "delta_late_minus_early": late_auroc - early_auroc,
            "majority_reference": maj_records[d],
            "best_params": u["best_params"], "inner_splitter_class": u["inner_splitter_class"],
            "n_train": u["n_train"],
            # fixture-compatibility aliases (results.v1.fixture.json's placeholder shape)
            "early": {m_: positional_bins_public[0][m_] for m_ in METRICS},
            "middle": {m_: positional_bins_public[1][m_] for m_ in METRICS},
            "late": {m_: positional_bins_public[2][m_] for m_ in METRICS},
        }
        per_dyad.append(row)
    return per_dyad


def build_tests(per_dyad, null_records):
    by_d = {row["pair_id"]: row for row in per_dyad}

    def _paired_test(deltas):
        sign = paired_sign_test(deltas)
        perm = signflip_permutation_test(deltas, N_SIGNFLIP, M.SEED)
        ci = M.ci95_from_folds(deltas)
        return {
            "n": len(deltas), "dyad_ids": TESTED_DYADS, "deltas": deltas,
            "median_delta": float(np.median(deltas)),
            "n_positive": sign["n_positive"], "n_negative": sign["n_negative"],
            "n_ties_excluded": sign["n_ties_excluded"], "sign_test_p": sign["p"],
            "permutation_p": perm["p"], "n_signflip": N_SIGNFLIP, "ci95": ci,
        }

    def _bin_auroc(d, bname):
        for pb in by_d[d]["positional_bins"]:
            if pb["bin"] == bname:
                return pb["auroc"]
        raise KeyError(bname)

    def _wp_auroc(d, bname):
        for wb in by_d[d]["within_person_bins"]:
            if wb["bin"] == bname:
                return wb["mean_auroc"]
        raise KeyError(bname)

    tests = {}

    # PRIMARY: observer_positional_late_minus_early (S20 paired, n=10)
    deltas = [_bin_auroc(d, "late") - _bin_auroc(d, "early") for d in TESTED_DYADS]
    primary_result = _paired_test(deltas)
    supported = bool(
        primary_result["sign_test_p"] < 0.05
        and primary_result["permutation_p"] < 0.05
        and primary_result["median_delta"] > 0
    )
    tests["observer_positional_late_minus_early"] = {
        "result": primary_result, "designation": "confirmatory", "primary": True, "supported": supported,
        "plain_language": (
            f"The observer's brain did {'not ' if not supported else ''}become measurably more "
            f"informative about the partner's deception across the interaction "
            f"(median delta = {primary_result['median_delta']:+.4f}, "
            f"{primary_result['n_positive']} of {primary_result['n']} dyads positive, "
            f"sign p = {primary_result['sign_test_p']:.4g}, "
            f"permutation p = {primary_result['permutation_p']:.4g})."
        ),
    }

    # SECONDARY: observer_decodability_pooled (S21 permutation null, pooled)
    pooled = null_records["pooled"]
    tests["observer_decodability_pooled"] = {
        "result": pooled, "designation": "confirmatory", "primary": False, "tier": "secondary",
        "kind": "permutation",
        "plain_language": (
            f"Pooled across all ten dyads' positional bins, the fitted observer model's AUROC was "
            f"{pooled['observed_auroc']:.4f} against a label-shuffled null with mean "
            f"{pooled['null']['mean']:.4f} (sd {pooled['null']['sd']:.4f}), p = {pooled['p_value']:.4g}."
        ),
    }

    # EXPLORATORY family (BH-corrected together): 3 paired tests + 30 per-bin S21 nulls (dyad-grain)
    exploratory_paired = {}

    deltas = [_wp_auroc(d, "late") - _wp_auroc(d, "early") for d in TESTED_DYADS]
    exploratory_paired["observer_within_person_late_minus_early"] = _paired_test(deltas)

    deltas = [_bin_auroc(d, "middle") - _bin_auroc(d, "early") for d in TESTED_DYADS]
    exploratory_paired["observer_positional_middle_minus_early"] = _paired_test(deltas)

    deltas = [_bin_auroc(d, "late") - _bin_auroc(d, "middle") for d in TESTED_DYADS]
    exploratory_paired["observer_positional_late_minus_middle"] = _paired_test(deltas)

    per_bin_nulls = {}
    for d in TESTED_DYADS:
        for bname in ["early", "middle", "late"]:
            per_bin_nulls[f"{d}_{bname}"] = null_records["dyad_grain"][d][bname]

    # BH correction over the combined exploratory family: 3 paired-test sign p-values
    # + 30 per-bin null p-values, ranked and adjusted together as one family.
    combined_names = list(exploratory_paired.keys()) + list(per_bin_nulls.keys())
    combined_pvals = (
        [exploratory_paired[n_]["sign_test_p"] for n_ in exploratory_paired]
        + [per_bin_nulls[n_]["p_value"] for n_ in per_bin_nulls]
    )
    adj = _benjamini_hochberg(combined_pvals, alpha=0.05)
    adj_by_name = dict(zip(combined_names, adj))

    for n_, result in exploratory_paired.items():
        tests[n_] = {
            "result": result, "designation": "exploratory", "primary": False,
            "bh_adjusted_p": adj_by_name[n_], "bh_family_size": len(combined_names),
        }

    tests["per_bin_nulls"] = {
        n_: {**per_bin_nulls[n_], "designation": "exploratory", "bh_adjusted_p": adj_by_name[n_],
             "bh_family_size": len(combined_names)}
        for n_ in per_bin_nulls
    }

    tests["primary"] = "observer_positional_late_minus_early"
    tests["secondary"] = ["observer_decodability_pooled"]
    tests["exploratory"] = list(exploratory_paired.keys())
    tests["multiple_comparisons_note"] = (
        "observer_positional_late_minus_early (S20 paired, dyad-grain terciles) is the single "
        "pre-registered primary claim, no correction. observer_decodability_pooled (S21 permutation "
        "null) is a pre-registered secondary, no correction. observer_within_person_late_minus_early, "
        "observer_positional_middle_minus_early, observer_positional_late_minus_middle, and the 30 "
        "per-bin S21 nulls (dyad-grain) are exploratory, Benjamini-Hochberg corrected together at "
        "alpha=0.05 within that family (family size 33). Declared before any number below was "
        "computed, per results/frozen_hypotheses.md Amendment 3."
    )
    return tests


def build_interpretation() -> dict:
    return {
        "s16_reporting_constraint": (
            "Per S16, this output may not claim that the observer unconsciously or subconsciously "
            "perceives, senses, or knows their partner is lying unless behavioral and statistical "
            "evidence directly supports it. This output never attributes detection or knowledge of "
            "deception to the observer -- only statistical decodability of a signal to a fitted model."
        ),
        # Not scanned by validation 15 (see _scan_banned_phrases's SCAN_EXCLUDE_PATHS) -- this is the
        # specification of which phrases are forbidden, not a claim being made about the findings.
        # Scanning a list of banned strings against itself would be circular by construction.
        "banned_phrases": list(BANNED_PHRASES),
        "permitted_register": (
            "A linear model trained on other dyads' observer EEG separated the partner's lie trials "
            "from truth trials at AUROC = X (95% CI [., .]) on held-out trials from this dyad; against "
            "a label-shuffled null this is p = Y. This is a statement about decodability of a signal, "
            "not about a person detecting anything."
        ),
        "what_decodability_is_not": (
            "Above-chance decoding of a partner's condition from an observer's EEG is consistent with "
            "the observer's brain responding differently to lies than to truths; it is NOT evidence "
            "the observer knows, senses, or acts on that difference. It is also consistent with mundane "
            "alternatives that are not detection at all: the deceiver behaving differently on lie trials "
            "(longer pauses, altered prosody, different button-press timing) and the observer's EEG "
            "tracking that observable behaviour; task-structure differences between lie and truth "
            "trials; or shared temporal structure between the two brains' recordings. exp6 cannot "
            "distinguish these."
        ),
        "evidentiary_conditions_for_a_stronger_claim": [
            "Behavioural corroboration -- a relationship between the model's trial-level confidence "
            "and the observer's own observer_guess on the same trial, over and above ground truth. "
            "This dataset has observer_guess, so this is testable, and exp6 deliberately does not "
            "test it (a different target, a different experiment). Absent that link, no claim about "
            "detection is available at all.",
            "Effect size -- decodability unambiguously above chance, not a couple of AUROC points; a "
            "confidence interval well clear of 0.5, not one that grazes it.",
            "Consistency across dyads -- a clear majority of the ten dyads in the same direction, not "
            "a 6/4 split, and a S20 p-value that survives the exploratory-family correction.",
            "Robustness to the mundane alternatives above -- at minimum, showing the effect survives "
            "controlling for the behavioural columns (outcome, points, response timing). exp1's "
            "references.behavioral_only_lr is the existing precedent for that kind of control and "
            "exp6 does not build one.",
        ],
        "observer_guess_link_out_of_scope": (
            "Relating the model's trial-level confidence to the observer's own observer_guess is the "
            "single piece of behavioural corroboration S16's warning names, and the dataset has the "
            "column. It is a different target and a different experiment; building it here would turn "
            "exp6 into the subconscious-detection study S16 explicitly warns against, without the "
            "design to support it. Deliberately out of scope; named here and in Limitations."
        ),
    }


# ---------------------------------------------------------------------------
# Validations (Task 5, 18 checks -- every one runs and prints real output)
# ---------------------------------------------------------------------------

def run_validations(Xv, yv, row_keys, blocks, univ_records, maj_records, null_records,
                     per_dyad, tests, n_jobs_check_result, observer_manifest):
    print("\n" + "=" * 70, flush=True)
    print("Validations", flush=True)
    print("=" * 70, flush=True)
    v = {}
    pair_id_arr = row_keys["pair_id"].values

    # 1. role purity
    ok1 = True  # asserted at load_observer_frame() and at run() call site
    v["1_role_purity"] = {"ok": ok1, "note": "asserted (full['role']==ROLE).all() in load_observer_frame()"}
    print(f"1. role-purity: ok={ok1}", flush=True)

    # 2. no self-dyad leakage
    ok2 = True
    detail2 = {}
    for d in TESTED_DYADS:
        pids = set(univ_records[d]["train_pair_ids"])
        clean = d not in pids and EXCLUDED_FROM_EVERYTHING not in pids and UNTESTABLE_UNIT_STILL_IN_POOL in pids
        ok2 = ok2 and clean
        detail2[d] = {"n_train": univ_records[d]["n_train"], "pids": sorted(pids), "clean": clean}
    v["2_no_self_dyad_leakage"] = {"ok": ok2, "per_dyad": detail2}
    print(f"2. no-self-dyad-leakage: ok={ok2}", flush=True)

    # 3. identical bins across arms (Universal, majority, S21 null all score same idx)
    ok3 = True
    v["3_identical_bins_across_arms"] = {
        "ok": ok3,
        "note": "True by construction -- run_universal_and_score_bins/run_majority both index "
                "blocks[d]['dyad_tercile_bins']/['person_tercile_bins'] directly; the S21 null reuses "
                "the Universal model's own cached y_true/y_score for the identical bin, never a "
                "separate scoring pass.",
    }
    print(f"3. identical-bins-across-arms: ok={ok3}", flush=True)

    # 4. bin integrity + mirror check against deceiver rows on the same seq bins
    ok4 = True
    detail4_dyad, detail4_person = {}, {}
    min_minority_dyad, min_minority_person = None, None
    for d in TESTED_DYADS:
        all_idx = np.concatenate(blocks[d]["dyad_tercile_bins"])
        full_dyad_idx = np.where(row_keys["pair_id"].values == d)[0]
        partitions_dyad = set(all_idx.tolist()) == set(full_dyad_idx.tolist()) and len(all_idx) == len(full_dyad_idx)
        ok4 = ok4 and partitions_dyad
        for pb in univ_records[d]["positional_bins"]:
            n_min = pb["n_minority"]
            detail4_dyad.setdefault(d, {})[pb["bin"]] = {"n": pb["n"], "n_minority": n_min}
            min_minority_dyad = n_min if min_minority_dyad is None else min(min_minority_dyad, n_min)

        a_all = np.concatenate(blocks[d]["person_tercile_bins"][0:3])
        b_all = np.concatenate(blocks[d]["person_tercile_bins"][3:6])
        partitions_a = set(a_all.tolist()) == set(blocks[d]["a_block_idx"].tolist())
        partitions_b = set(b_all.tolist()) == set(blocks[d]["b_block_idx"].tolist())
        ok4 = ok4 and partitions_a and partitions_b
        for wb in univ_records[d]["wp_bins_raw"]:
            n_min = wb["n_minority"]
            detail4_person.setdefault(d, {})[f"{wb['participant']}_{wb['position']}"] = {
                "n": wb["n"], "n_minority": n_min}
            min_minority_person = n_min if min_minority_person is None else min(min_minority_person, n_min)

    # mirror check: re-derive the identical seq bins from deceiver rows and confirm identical counts
    dec_manifest_path = FEATURES_DIR / "shard_manifest.json"
    dec_manifest = json.loads(dec_manifest_path.read_text())
    dec_frames = [pd.read_parquet(FEATURES_DIR / f"shard_{pid}.parquet") for pid in dec_manifest["shards"]]
    dec_full = pd.concat(dec_frames, ignore_index=True).sort_values("_orig_row", kind="mergesort").reset_index(drop=True)
    dec_full = dec_full.drop(columns=["_orig_row"])
    dec_needed_cols = ["pair_id", "dyad_trial_seq", "condition"]
    dec_y = (dec_full["condition"] == "lie").astype(int).values
    dec_pair = dec_full["pair_id"].values
    dec_seq = dec_full["dyad_trial_seq"].values
    mirror_ok = True
    mirror_detail = {}
    for d in TESTED_DYADS:
        for pb_ in univ_records[d]["positional_bins"]:
            bname = pb_["bin"]
            obs_bin = pb_
            # Use the REAL seq_min/seq_max each observer bin was actually built with
            # (via _contig_bins' equal-row-count split, matching exp5's real behavior --
            # NOT the nominal DYAD_TERCILES cutoffs, which describe the ~322/324/322
            # seq-value-range boundaries but are not what _contig_bins produces for a
            # 968-row dyad; see PROGRESS.md for the divergence this caught).
            lo, hi = obs_bin["seq_min"], obs_bin["seq_max"]
            dec_mask = (dec_pair == d) & (dec_seq >= lo) & (dec_seq <= hi)
            dec_n = int(dec_mask.sum())
            dec_n_lie = int(dec_y[dec_mask].sum())
            dec_n_minority = min(dec_n_lie, dec_n - dec_n_lie)
            same = (dec_n == obs_bin["n"]) and (dec_n_minority == obs_bin["n_minority"])
            mirror_ok = mirror_ok and same
            mirror_detail[f"{d}_{bname}"] = {
                "observer_n": obs_bin["n"], "observer_n_minority": obs_bin["n_minority"],
                "deceiver_n": dec_n, "deceiver_n_minority": dec_n_minority, "identical": same,
            }
    ok4 = ok4 and mirror_ok
    v["4_bin_integrity_and_mirror"] = {
        "ok": ok4, "dyad_grain": detail4_dyad, "person_grain": detail4_person,
        "min_minority_dyad_grain": min_minority_dyad, "min_minority_person_grain": min_minority_person,
        "reproduces_fact4_130": min_minority_dyad == 130 if min_minority_dyad is not None else None,
        "reproduces_fact4_45": min_minority_person == 45 if min_minority_person is not None else None,
        "mirror_check_vs_deceiver_rows_ok": mirror_ok, "mirror_detail_sample": dict(list(mirror_detail.items())[:6]),
    }
    print(f"4. bin-integrity-and-mirror: ok={ok4}, min_minority_dyad={min_minority_dyad}, "
          f"min_minority_person={min_minority_person}, mirror_ok={mirror_ok}", flush=True)

    # 5. target correctness -- y on observer row equals y on deceiver row for same (pair_id, seq)
    dec_key = pd.Series(list(zip(dec_pair.tolist(), dec_seq.tolist())))
    dec_y_by_key = dict(zip(dec_key, dec_y.tolist()))
    obs_pair = row_keys["pair_id"].values
    obs_seq = row_keys["dyad_trial_seq"].values
    rng = np.random.default_rng(M.SEED)
    sample_idx = rng.choice(len(row_keys), size=min(200, len(row_keys)), replace=False)
    matches = 0
    for i in sample_idx:
        key = (obs_pair[i], int(obs_seq[i]))
        if key in dec_y_by_key and dec_y_by_key[key] == int(yv[i]):
            matches += 1
    match_rate = matches / len(sample_idx)
    v["5_target_correctness"] = {"match_rate": match_rate, "n_sampled": len(sample_idx), "ok": match_rate == 1.0}
    print(f"5. target-correctness: match_rate={match_rate:.4f}", flush=True)

    # 6. NaN drop -- checked at run() call site, recorded here
    v["6_nan_drop"] = {"note": "checked and recorded at run() call site (n_dropped_nan block)"}
    print("6. nan-drop: recorded at run() call site", flush=True)

    # 7. identity-swap table (Fact 2)
    detail7 = {}
    ok7 = True
    for d in TESTED_DYADS:
        b = blocks[d]
        dec_early_mask = (dec_pair == d) & (dec_seq <= 484)
        dec_late_mask = (dec_pair == d) & (dec_seq >= 485)
        dec_full_pid = dec_full["participant_id"].values
        dec_early_deceiver = sorted(set(dec_full_pid[dec_early_mask].tolist()))
        dec_late_deceiver = sorted(set(dec_full_pid[dec_late_mask].tolist()))
        ok = (len(dec_early_deceiver) == 1 and len(dec_late_deceiver) == 1
              and dec_early_deceiver[0] != b["A"] and dec_late_deceiver[0] != b["B"]
              and dec_early_deceiver[0] == b["B"] and dec_late_deceiver[0] == b["A"])
        ok7 = ok7 and ok
        detail7[d] = {"observer_early": b["A"], "observer_late": b["B"],
                       "deceiver_early": dec_early_deceiver, "deceiver_late": dec_late_deceiver, "swapped": ok}
    v["7_identity_swap_table"] = {"ok": ok7, "per_dyad": detail7}
    print(f"7. identity-swap-table: ok={ok7}", flush=True)

    # 8. no identity/outcome columns in X
    forbidden = set(M.IDENTITY_COLUMNS) | {"outcome", "observer_guess", "points"}
    v["8_no_identity_columns"] = {"forbidden_set": sorted(forbidden), "checked_at": "run() before .values"}
    print("8. no-identity-columns: checked at run() call site", flush=True)

    # 9. majority-class sanity
    maj_aurocs = [m["auroc"] for d in TESTED_DYADS for m in maj_records[d].values()]
    maj_bal = [m["balanced_accuracy"] for d in TESTED_DYADS for m in maj_records[d].values()]
    ok9 = all(abs(a - 0.5) <= 0.02 for a in maj_aurocs) and all(abs(bacc - 0.5) <= 0.02 for bacc in maj_bal)
    v["9_majority_class_sanity"] = {"mean_auroc": float(np.mean(maj_aurocs)),
                                     "mean_balanced_accuracy": float(np.mean(maj_bal)), "ok": ok9}
    print(f"9. majority-class-sanity: {v['9_majority_class_sanity']}", flush=True)

    # 10. reproducibility -- refit 3 sampled units from scratch, bypass checkpoint cache
    sample_dyads = TESTED_DYADS[:3]
    diffs10 = {}
    for d in sample_dyads:
        train_idx = universal_pool_idx(row_keys, d)
        groups_train = pair_id_arr[train_idx]
        inner = M.default_grouped_inner(M.SEED, n_splits=3)
        test_idx = blocks[d]["dyad_tercile_bins"][2]

        def _fit_once():
            pipe = M._lr_pipeline("l2", M.SEED)
            grid = M._lr_param_grid("l2")
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1)
            search.fit(Xv[train_idx], yv[train_idx], groups=groups_train)
            m, _ = _score_on_idx(Xv, yv, search, test_idx)
            return m["auroc"]

        a1 = _fit_once()
        a2 = _fit_once()
        diffs10[d] = abs(a1 - a2)
    v["10_reproducibility"] = {"max_diff": max(diffs10.values()), "per_dyad": diffs10}
    print(f"10. reproducibility: max_diff={v['10_reproducibility']['max_diff']:.2e}", flush=True)

    # 11. GRID_N_JOBS invariance
    v["11_grid_n_jobs_invariance"] = n_jobs_check_result
    print(f"11. grid-n-jobs-invariance: identical={n_jobs_check_result['metrics_identical']}", flush=True)

    # 12. metric cross-check
    from scipy.stats import mannwhitneyu
    d0 = TESTED_DYADS[0]
    late_bin = next(pb for pb in univ_records[d0]["positional_bins"] if pb["bin"] == "late")
    y_test = np.array(late_bin["_y_true"]) if "_y_true" in late_bin else None
    # positional_bins in the assembled per_dyad output has private fields stripped;
    # recompute directly from the checkpointed record instead.
    raw_late = next(pb for pb in _ckpt_load(f"univ_obs_{d0}")["positional_bins"] if pb["bin"] == "late")
    y_test = np.array(raw_late["_y_true"])
    y_score = np.array(raw_late["_y_score"])
    sk_auc = float(roc_auc_score(y_test, y_score))
    pos = y_score[y_test == 1]
    neg = y_score[y_test == 0]
    u_stat, _ = mannwhitneyu(pos, neg)
    hand_auc = float(u_stat / (len(pos) * len(neg)))
    v["12_metric_crosscheck"] = {"dyad": d0, "sklearn_auroc": sk_auc, "hand_auroc": hand_auc,
                                  "match_to_1e6": abs(sk_auc - hand_auc) < 1e-6}
    print(f"12. metric-crosscheck: {v['12_metric_crosscheck']}", flush=True)

    # 13. upstream file hashes
    models_py_path = REPO_ROOT / "src" / "models.py"
    models_py_hash = hashlib.sha256(models_py_path.read_bytes()).hexdigest()
    models_py_ok = models_py_hash == MODELS_PY_SHA256_EXPECTED
    gate_hash = hashlib.sha256(GATE_JSON.read_bytes()).hexdigest()
    obs_manifest_hash = hashlib.sha256(OBS_SHARD_MANIFEST.read_bytes()).hexdigest()
    frozen_hash = hashlib.sha256(FROZEN_HYP_MD.read_bytes()).hexdigest()
    frozen_ok = frozen_hash == FROZEN_HYP_SHA256_EXPECTED
    v["13_upstream_files_hashed"] = {
        "hashes": {"models.py": models_py_hash, "gate.json": gate_hash,
                   "shard_obs_manifest.json": obs_manifest_hash, "frozen_hypotheses.md": frozen_hash},
        "models_py_matches_exp5": models_py_ok,
        "frozen_hypotheses_matches_amended_hash": frozen_ok,
    }
    if not models_py_ok:
        raise RuntimeError(
            f"src/models.py sha256 {models_py_hash} != exp5's recorded {MODELS_PY_SHA256_EXPECTED} -- "
            "stop and report per plan Task 5 validation 14.")
    print(f"13. upstream-files-hashed: models_py_ok={models_py_ok} frozen_ok={frozen_ok}", flush=True)

    # 14. data integrity
    v["14_data_integrity"] = {
        "row_count_10648_before_nan_drop": observer_manifest["n_rows_total"] == 10648,
        "total_frame_hash_verified": "asserted in load_observer_frame() -- would have raised if mismatched",
    }
    print(f"14. data-integrity: {v['14_data_integrity']}", flush=True)

    # 15. S16 language guard -- recursive banned-phrase scan, FAILS THE RUN on any hit
    scan_targets = []

    def _collect_strings(obj, path=""):
        if isinstance(obj, str):
            scan_targets.append((path, obj))
        elif isinstance(obj, dict):
            for k_, v_ in obj.items():
                _collect_strings(v_, f"{path}.{k_}")
        elif isinstance(obj, (list, tuple)):
            for i_, v_ in enumerate(obj):
                _collect_strings(v_, f"{path}[{i_}]")

    # Scanned by run() after full assembly (results dict + markdown) -- see write_outputs().
    v["15_s16_language_guard"] = {"note": "executed in write_outputs() against the final assembled "
                                           "results dict and markdown; see meta.language_guard_result "
                                           "for the actual scan outcome"}
    print("15. s16-language-guard: deferred to write_outputs() final-assembly scan", flush=True)

    # 16. amendment and designation recorded
    amend_text = FROZEN_HYP_MD.read_text()
    amend_present = "Amendment 3" in amend_text and "n = 10" in amend_text
    recomputed_supported = bool(
        tests["observer_positional_late_minus_early"]["result"]["sign_test_p"] < 0.05
        and tests["observer_positional_late_minus_early"]["result"]["permutation_p"] < 0.05
        and tests["observer_positional_late_minus_early"]["result"]["median_delta"] > 0
    )
    supported_matches = recomputed_supported == tests["observer_positional_late_minus_early"]["supported"]
    only_hierarchy_confirmatory = (
        tests["observer_positional_late_minus_early"]["designation"] == "confirmatory"
        and tests["observer_decodability_pooled"]["designation"] == "confirmatory"
        and all(tests[n_]["designation"] == "exploratory" for n_ in tests["exploratory"])
    )
    v["16_amendment_and_designation_recorded"] = {
        "amendment_3_present": amend_present, "n_dyads": len(TESTED_DYADS),
        "sub19_sub22_absent_from_tested_dyads": UNTESTABLE_UNIT_STILL_IN_POOL not in TESTED_DYADS,
        "only_hierarchy_labelled_confirmatory": only_hierarchy_confirmatory,
        "supported_flag_recomputed_matches": supported_matches,
    }
    print(f"16. amendment-and-designation-recorded: {v['16_amendment_and_designation_recorded']}", flush=True)

    # 17. plausibility / leakage
    all_aurocs = [pb["auroc"] for row in per_dyad for pb in row["positional_bins"]]
    flagged = []
    for row in per_dyad:
        d = row["pair_id"]
        if abs(row["delta_late_minus_early"]) > 0.20:
            flagged.append(d)
        if any(pb["auroc"] > 0.75 for pb in row["positional_bins"]):
            flagged.append(d)
    v["17_plausibility"] = {
        "max_auroc": float(max(all_aurocs)), "dyads_flagged": sorted(set(flagged)),
        "leakage_suspicion": bool(len(flagged) > 0),
        "note": "Given Fact 5 (prior expectation ~0.53-0.54), anything above 0.75 on observer rows is "
                "far more likely a leak than a finding.",
    }
    print(f"17. plausibility: {v['17_plausibility']}", flush=True)

    # 18. convergence
    n_conv_univ = sum(len(univ_records[d]["convergence_warnings"]) for d in TESTED_DYADS)
    v["18_convergence"] = {"n_convergence_warnings": {"universal": n_conv_univ}}
    print(f"18. convergence: {v['18_convergence']}", flush=True)

    return v


def _cross_experiment_check(per_dyad):
    """Descriptive comparison against exp5's deceiver-row positional bins on
    the IDENTICAL trials/seq bins -- Fact 2's identity-swap table. Never a
    test; exp7 (S17) owns the tested observer-vs-deceiver comparison."""
    if not EXP5_JSON.exists():
        return {"status": "exp5_not_available"}
    exp5 = json.loads(EXP5_JSON.read_text())
    exp5_per_dyad = {row["pair_id"]: row for row in exp5["experiments"]["exp5"]["per_dyad"]}
    comparison = {}
    for row in per_dyad:
        d = row["pair_id"]
        if d not in exp5_per_dyad:
            continue
        exp5_bins = {pb["bin"]: pb["auroc"] for pb in exp5_per_dyad[d]["positional_bins"]}
        exp6_bins = {pb["bin"]: pb["auroc"] for pb in row["positional_bins"]}
        exp5_delta = exp5_bins.get("late", float("nan")) - exp5_bins.get("early", float("nan"))
        exp6_delta = row["delta_late_minus_early"]
        comparison[d] = {
            "exp5_deceiver_bins": exp5_bins, "exp6_observer_bins": exp6_bins,
            "exp5_delta_late_minus_early": exp5_delta, "exp6_delta_late_minus_early": exp6_delta,
            "observer_of_early_block": row["observers"]["early_block"],
            "observer_of_late_block": row["observers"]["late_block"],
        }
    return {
        "per_dyad": comparison,
        "note": "descriptive corroboration, not a test -- exp7 (S17) owns the paired "
                "observer-vs-deceiver comparison. exp5 scores the deceiver's own brain on the same "
                "trials/seq bins; exp6 scores the observer's brain. The identity-swap table (Fact 2) "
                "shows the two experiments' 'early' and 'late' brains belong to opposite participants.",
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def build_markdown(results: dict) -> str:
    exp6 = results["experiments"]["exp6"]
    lines = []
    lines.append("# Experiment 6 -- Observer-Only Prediction (S16)\n")

    lines.append("## Findings\n")
    lines.append(exp6["interpretation"]["what_decodability_is_not"] + "\n")
    primary = exp6["tests"]["observer_positional_late_minus_early"]
    lines.append(f"\n{primary['plain_language']}\n")

    lines.append("\n## Design\n")
    d = exp6["design"]
    lines.append(f"- n = {d['n_dyads']} dyads (amended from pre-registered n=11; see "
                  "`results/frozen_hypotheses.md` Amendment 3).\n"
                  f"- Role: {d['role']}. Target: {d['target']}\n"
                  f"- Feature set: {d['feature_set']} ({d['n_features']} cols). Model: {d['model_family']}\n")
    lines.append(d.get("identity_confound_prose", "") + "\n")
    lines.append(d.get("n10_prose", "") + "\n")
    lines.append(d.get("prior_expectation", "") + "\n")

    lines.append("\n## Gate re-check (Amendment 3)\n")
    gr = exp6["gate_recheck"]
    lines.append(f"THRESHOLD_M = {gr['threshold']} (copied unchanged from the original gate).\n\n")
    lines.append("| analysis | test-fold grain | minority | Clause A | Clause B | designation |\n"
                  "|---|---|---|---|---|---|\n")
    for row in gr["rows"]:
        lines.append(f"| {row['analysis']} | {row['grain']} | {row['minority']} | {row['clause_a']} | "
                      f"{row['clause_b']} | {row['designation']} |\n")
    lines.append(f"\nFailing bins: {gr['failing_bins']}\n")

    lines.append("\n## Absolute decodability (S21) -- reported before the early/late comparison\n")
    pooled = exp6["tests"]["observer_decodability_pooled"]
    lines.append(f"{pooled['plain_language']}\n")

    lines.append("\n## Positional early/middle/late per dyad (dyad-grain, AUROC)\n")
    lines.append("| pair_id | early | middle | late | delta(late-early) |\n|---|---|---|---|---|\n")
    for row in exp6["per_dyad"]:
        e, m_, l = row["early"]["auroc"], row["middle"]["auroc"], row["late"]["auroc"]
        lines.append(f"| {row['pair_id']} | {e:.4f} | {m_:.4f} | {l:.4f} | {row['delta_late_minus_early']:+.4f} |\n")

    lines.append("\n## Primary claim (S20)\n")
    r = primary["result"]
    lines.append(f"Median = {r['median_delta']:+.4f}, {r['n_positive']}/{r['n']} dyads positive, "
                  f"sign-test p = {r['sign_test_p']:.4g}, permutation p = {r['permutation_p']:.4g}. "
                  f"95% CI: [{r['ci95']['lower']:.4f}, {r['ci95']['upper']:.4f}]. "
                  f"supported = {primary['supported']}.\n")

    lines.append("\n## Exploratory (Benjamini-Hochberg corrected)\n")
    for name in exp6["tests"]["exploratory"]:
        t = exp6["tests"][name]
        r = t["result"]
        lines.append(f"- **{name}**: median = {r['median_delta']:+.4f}, {r['n_positive']}/{r['n']} positive, "
                      f"sign-test p = {r['sign_test_p']:.4g} (BH-adjusted: {t['bh_adjusted_p']:.4g})\n")
    lines.append(f"\n{exp6['tests']['multiple_comparisons_note']}\n")

    lines.append("\n## Within-observer grain (1B, EXPLORATORY, UNDERPOWERED)\n")
    lines.append("Fails the exp6-specific gate re-check (Clause A on sub23=45, Clause B at 8/10). "
                  "Reported with CIs per the frozen file's own rule, never dropped.\n")
    r = exp6["tests"]["observer_within_person_late_minus_early"]["result"]
    lines.append(f"Median = {r['median_delta']:+.4f}, {r['n_positive']}/{r['n']} positive, "
                  f"sign-test p = {r['sign_test_p']:.4g}. 95% CI: [{r['ci95']['lower']:.4f}, "
                  f"{r['ci95']['upper']:.4f}].\n")

    lines.append("\n## Aggregate (NOT the test, S20)\n")
    ag = exp6["aggregate"]
    lines.append(f"- early: median AUROC = {ag['by_bin']['early']['median']:.4f}\n"
                  f"- middle: median AUROC = {ag['by_bin']['middle']['median']:.4f}\n"
                  f"- late: median AUROC = {ag['by_bin']['late']['median']:.4f}\n"
                  f"- pooled observer decodability (S21): observed = {pooled['result']['observed_auroc']:.4f}, "
                  f"p = {pooled['result']['p_value']:.4g}\n")

    lines.append("\n## Cross-experiment vs exp5 (descriptive, not a test)\n")
    ce = exp6["cross_experiment"]
    if ce.get("status") == "exp5_not_available":
        lines.append("exp5_history.json was not available when this run executed.\n")
    else:
        lines.append(ce["note"] + "\n")

    lines.append("\n## Environment\n")
    env = results["meta"]["environment"]
    lines.append(f"Remote: Python {env['remote']['python_version']}, sklearn {env['remote']['sklearn_version']}. "
                  f"Laptop: Python {env['laptop']['python_version']}, sklearn {env['laptop']['sklearn_version']}.\n")

    lines.append("\n## Validation summary\n")
    for k, val in exp6["validations"].items():
        lines.append(f"- **{k}**: {val}\n")

    lines.append("\n## Limitations\n")
    lines.append(
        "- n=10, not the pre-registered n=11 (Amendment 3; sub19_sub22 structurally untestable).\n"
        "- Analysis 1A (dyad-grain positional split) is confounded with participant identity, in the "
        "opposite direction to exp5: the early tercile's observer and the late tercile's observer are "
        "different people. This is published as data (per-bin participant ids), not smoothed over.\n"
        "- Analysis 1B (within-observer grain, unconfounded) is underpowered and exploratory -- "
        "reported with CIs, not dropped.\n"
        "- The S21 null holds the fitted model fixed and permutes only the scored bin's labels; it "
        "tests whether the model's ranking of these trials beats chance, not whether the tuning "
        "procedure itself would find signal in noise. This is the same class of null exp1 used.\n"
        "- The observer_guess behavioural-corroboration link is deliberately not built here -- see "
        "interpretation.observer_guess_link_out_of_scope. Absent it, exp6 alone cannot license any "
        "claim stronger than statistical decodability.\n"
        "- All conditions fit in one Python-3.8/sklearn-1.3.2 remote environment; not directly "
        "comparable point-to-point to laptop-fit experiments.\n"
    )

    return "\n".join(lines)


# Paths excluded from validation 15's scan: fields whose entire purpose is to
# NAME the forbidden phrases (the specification of the rule), not to assert
# anything about exp6's findings. Scanning the banned-phrase list against
# itself is circular by construction and would make the interpretation block
# (which validation 15 itself requires to exist) unwritable. Every other
# string in the output -- including the S16 prose describing the constraint --
# is still scanned; that prose was rewritten (see build_interpretation) to
# paraphrase rather than quote the banned phrases verbatim.
SCAN_EXCLUDE_PATHS = {"interpretation.banned_phrases"}


def _scan_banned_phrases(results: dict, markdown: str) -> dict:
    """Validation 15: recursive scan of every string value in `results` plus
    the full markdown text, case-insensitive, for BANNED_PHRASES. Any hit is
    a hard failure -- raises, does not leave a claimed-final output."""
    hits = []
    n_strings = 0

    def _walk(obj, path=""):
        nonlocal n_strings
        if any(path == excl or path.endswith("." + excl) for excl in SCAN_EXCLUDE_PATHS):
            return
        if isinstance(obj, str):
            n_strings += 1
            low = obj.lower()
            for phrase in BANNED_PHRASES:
                if phrase in low:
                    hits.append({"phrase": phrase, "snippet": obj[:200], "path": path})
        elif isinstance(obj, dict):
            for k_, v_ in obj.items():
                _walk(v_, f"{path}.{k_}" if path else k_)
        elif isinstance(obj, (list, tuple)):
            for v_ in obj:
                _walk(v_, path)

    _walk(results, "results")
    n_strings += 1
    low_md = markdown.lower()
    for phrase in BANNED_PHRASES:
        if phrase in low_md:
            hits.append({"phrase": phrase, "snippet": "(in markdown body)"})

    return {"n_strings_scanned": n_strings, "hits": hits, "clean": len(hits) == 0}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not serializable: {type(o)}")


def write_outputs(results: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    md = build_markdown(results)

    # Validation 15 -- run against the FINAL assembled results dict + markdown,
    # before either file is written. Fails the run (raises) on any hit.
    scan = _scan_banned_phrases(results, md)
    results["experiments"]["exp6"]["validations"]["15_s16_language_guard"] = {
        **results["experiments"]["exp6"]["validations"]["15_s16_language_guard"],
        "n_strings_scanned": scan["n_strings_scanned"], "hits": scan["hits"], "clean": scan["clean"],
        "evidentiary_conditions_present": len(
            results["experiments"]["exp6"]["interpretation"]["evidentiary_conditions_for_a_stronger_claim"]
        ) == 4,
    }
    print(f"15. s16-language-guard: n_strings_scanned={scan['n_strings_scanned']} clean={scan['clean']}",
          flush=True)
    if not scan["clean"]:
        raise RuntimeError(
            f"Validation 15 FAILED: banned phrase(s) found in output: {scan['hits']}. "
            "Refusing to write results/exp6_observer.json or .md.")

    # re-render markdown since the validations block above was mutated after the first render
    md = build_markdown(results)
    scan2 = _scan_banned_phrases(results, md)
    if not scan2["clean"]:
        raise RuntimeError(f"Validation 15 FAILED on re-render: {scan2['hits']}")

    with open(EXP6_JSON, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nWrote {EXP6_JSON}", flush=True)

    with open(EXP6_MD, "w") as f:
        f.write(md)
    print(f"Wrote {EXP6_MD}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run():
    global GRID_N_JOBS
    print("=" * 70, flush=True)
    print("Experiment 6: observer-only prediction (S16)", flush=True)
    print("=" * 70, flush=True)

    env = {"python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
           "numpy_version": np.__version__, "pandas_version": pd.__version__,
           "scipy_version": __import__("scipy").__version__, "pyarrow_version": __import__("pyarrow").__version__}
    print(f"Remote environment: {env}", flush=True)

    t_start = time.time()
    n_resume_launches = _ckpt_load("n_resume_launches") or 0
    n_resume_launches += 1
    _ckpt_save("n_resume_launches", n_resume_launches)
    print(f"Launch #{n_resume_launches}", flush=True)

    frozen_hash_now = hashlib.sha256(FROZEN_HYP_MD.read_bytes()).hexdigest()
    assert frozen_hash_now == FROZEN_HYP_SHA256_EXPECTED, (
        f"frozen_hypotheses.md hash mismatch: {frozen_hash_now} != {FROZEN_HYP_SHA256_EXPECTED}. "
        "Refusing to run against an un-amended or differently-amended frozen file.")
    print(f"frozen_hypotheses.md hash verified: {frozen_hash_now}", flush=True)

    fd = M.load_feature_dictionary()
    feature_sets = M.build_feature_sets(fd)
    feat_cols = feature_sets["reliable_plus_marginal"]
    print(f"Feature set reliable_plus_marginal: {len(feat_cols)} columns", flush=True)

    full, observer_manifest = load_observer_frame()
    print(f"Assembled observer frame: {full.shape}", flush=True)
    assert (full["role"] == ROLE).all()

    needed = [c for c in feat_cols if c in full.columns]
    missing = set(feat_cols) - set(needed)
    assert not missing, f"feature columns missing: {missing}"
    forbidden = set(M.IDENTITY_COLUMNS) | {"outcome", "observer_guess", "points"}

    nan_mask = full[needed].isna().any(axis=1)
    n_dropped_nan = int(nan_mask.sum())
    print(f"NaN-row drop: {n_dropped_nan} of {len(full)} observer rows dropped (mirrors exp4/exp5).", flush=True)
    full = full.loc[~nan_mask].reset_index(drop=True)

    X = full[needed].reset_index(drop=True)
    assert forbidden & set(X.columns) == set(), "identity/outcome/observer_guess/points columns leaked into X"
    y = (full["condition"] == "lie").astype(int).reset_index(drop=True)
    row_keys = full[["pair_id", "session_id", "round", "trial", "dyad_trial_seq", "participant_id",
                      "partner_id"]].reset_index(drop=True)
    Xv = X.values
    yv = y.values

    blocks = build_observer_bins(row_keys)
    print(f"Built observer bins for {len(blocks)} tested dyads (expected 10)", flush=True)

    print("\n--- GRID_N_JOBS invariance check ---", flush=True)
    njobs_check = _ckpt_load("njobs_check")
    if njobs_check is None:
        njobs_check = grid_n_jobs_check(Xv, yv, row_keys)
        _ckpt_save("njobs_check", njobs_check)
    print(f"  n_jobs=1: {njobs_check['n_jobs_1_seconds']:.1f}s, "
          f"n_jobs={GRID_N_JOBS}: {njobs_check['n_jobs_chosen_seconds']:.1f}s, "
          f"identical={njobs_check['metrics_identical']}", flush=True)
    if not njobs_check["metrics_identical"]:
        print("  WARNING: parallelism changed results -- falling back to n_jobs=1", flush=True)
        GRID_N_JOBS = 1
    chosen_n_jobs = GRID_N_JOBS

    print("\n--- Majority-class references (10 units) ---", flush=True)
    t0 = time.time()
    maj_records = run_majority(Xv, yv, row_keys, blocks)
    maj_seconds = time.time() - t0

    print("\n--- Universal observer fits + 9-bin scoring (10 units) ---", flush=True)
    t0 = time.time()
    univ_records = run_universal_and_score_bins(Xv, yv, row_keys, blocks, chosen_n_jobs)
    univ_seconds = time.time() - t0

    print("\n--- S21 label-shuffling nulls (no refit, cached scores) ---", flush=True)
    t0 = time.time()
    null_records = run_label_permutation_nulls(univ_records)
    null_seconds = time.time() - t0

    total_seconds = time.time() - t_start
    runtime = {
        "per_condition_seconds": {"majority": maj_seconds, "universal": univ_seconds,
                                   "label_permutation_nulls": null_seconds},
        "total_seconds": total_seconds, "n_resume_launches": n_resume_launches,
        "grid_n_jobs_used": chosen_n_jobs,
    }
    print(f"\nTotal runtime this launch: {total_seconds:.1f}s", flush=True)

    per_dyad = assemble(blocks, univ_records, maj_records, null_records)
    tests = build_tests(per_dyad, null_records)
    interpretation = build_interpretation()

    aggregate = {"note": "descriptive only -- NOT the test (S20)", "by_bin": {}}
    for bname in ["early", "middle", "late"]:
        vals = [row[bname]["auroc"] for row in per_dyad]
        aggregate["by_bin"][bname] = {"median": float(np.median(vals)),
                                       "spread": [float(min(vals)), float(max(vals))]}
    aggregate["pooled_decodability"] = null_records["pooled"]

    cross_experiment = _cross_experiment_check(per_dyad)

    validations = run_validations(Xv, yv, row_keys, blocks, univ_records, maj_records, null_records,
                                   per_dyad, tests, njobs_check, observer_manifest)

    gate_recheck = {
        "threshold": THRESHOLD_M, "threshold_copied_not_rederived": True,
        "clause_a": "smallest test-fold minority-class trial count >= THRESHOLD_M",
        "clause_b": "at least 10 of 12 dyads (or the equivalent fraction at reduced n; n=10 => >=9) individually clear Clause A",
        "rows": [
            {"analysis": "1A dyad-grain positional", "grain": "320-324", "minority": 130,
             "clause_a": "PASS", "clause_b": "10/10 PASS", "designation": "confirmatory_on_power_confounded"},
            {"analysis": "1B participant-grain positional", "grain": "162/161/161", "minority": 45,
             "clause_a": "FAIL", "clause_b": "8/10 FAIL", "designation": "exploratory_underpowered"},
            {"analysis": "2 S21 absolute-decodability null", "grain": "same bins as 1A", "minority": 130,
             "clause_a": "PASS", "clause_b": "10/10 PASS", "designation": "confirmatory"},
        ],
        "failing_bins": {"sub23": 45, "sub18": 59},
        "amendment_reference": "results/frozen_hypotheses.md Amendment 3",
    }

    design = {
        "research_question": "S16: does the observer's own EEG carry information about whether their "
                              "partner is currently deceiving them, and does that information grow "
                              "across the interaction?",
        "n_dyads": len(TESTED_DYADS), "tested_dyads": TESTED_DYADS,
        "excluded_from_test_set_but_in_universal_pool": {UNTESTABLE_UNIT_STILL_IN_POOL:
            "structurally untestable at dyad grain (Fact 3/Amendment 3) -- observer rows still used as "
            "other-dyad training data"},
        "excluded_from_everything": {EXCLUDED_FROM_EVERYTHING: "excluded from exp3-8 by frozen_hypotheses.md"},
        "role": ROLE, "target": "trial ground-truth condition; NOT observer_guess",
        "feature_set": "reliable_plus_marginal", "n_features": len(feat_cols),
        "model_family": "logistic_regression (L2) only",
        "bins": {"dyad_terciles": [t[0] for t in DYAD_TERCILES], "person_terciles": list(PERSON_TERCILE_SIZES)},
        "direction_convention": "positive delta = the observer's brain becomes more informative about "
                                 "the partner's deception later in the interaction",
        "identity_confound_prose": (
            "The observer of seq[1,484] is the session-2 deceiver and the observer of seq[485,968] is "
            "the session-1 deceiver -- the reverse of exp5's identity assignment on the same seq bins "
            "(Fact 2). This partially breaks the participant-identity confound across the two "
            "experiments but does not remove positional confounds (session order, fatigue, electrode "
            "drift, task familiarity) within exp6 alone. Published as data (per-bin participant ids), "
            "not smoothed over."
        ),
        "n10_prose": (
            "sub19_sub22's observer rows span seq [1,484] only (its only session), so its late tercile "
            "(exp6's positional 'late' bin) contains zero observer rows -- structurally untestable. "
            "results/frozen_hypotheses.md Amendment 3 reduces exp6's n from 11 to 10 accordingly; "
            "sub19_sub22's rows remain in the other-dyad training pool."
        ),
        "prior_expectation": (
            "exp1's already-run observer secondary put observer-only decodability at mean AUROC 0.5368 "
            "[0.5285, 0.5451] -- real but tiny, comparable to the deceiver-side signal (pooled AUROC "
            "0.5338). The early/late comparison is therefore a comparison of two near-chance numbers; "
            "the S21 absolute-decodability null is run and reported first for this reason."
        ),
    }

    results = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": M.SEED, "grid_n_jobs": chosen_n_jobs,
            "environment": {"remote": env, "laptop": LAPTOP_ENV},
            "runtime": runtime,
            "observer_shard_manifest_hash": observer_manifest["total_frame_sha256"],
            "frozen_hypotheses_amended_sha256": FROZEN_HYP_SHA256_EXPECTED,
        },
        "experiments": {
            "exp6": {
                "status": "complete", "provenance": "real",
                "gate_verdict": {"1A": "CONFIRMATORY", "1B": "EXPLORATORY_UNDERPOWERED", "2": "CONFIRMATORY"},
                "design": design,
                "gate_recheck": gate_recheck,
                "per_dyad": per_dyad,
                "aggregate": aggregate,
                "tests": tests,
                "interpretation": interpretation,
                "validations": validations,
                "cross_experiment": cross_experiment,
            }
        },
    }

    write_outputs(results)
    return results


if __name__ == "__main__":
    run()
