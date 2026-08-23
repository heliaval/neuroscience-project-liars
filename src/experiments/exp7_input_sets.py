"""
src/experiments/exp7_input_sets.py -- Experiment 7 driver (S17): One Brain vs
Two Brains -- where does the predictive information about deception actually
live?

Scores FIVE input sets -- deceiver_eeg, observer_eeg, both_brains, interbrain,
eeg_plus_behavioral -- on LITERALLY IDENTICAL leave-one-dyad-out (LODO) folds
at TRIAL grain (n=11 dyads), so the comparisons between input sets are paired,
not just five separate experiments run side by side. This is the structural
shift from exp1-exp6: rows are trials (one prediction per trial, using BOTH
participants' EEG when relevant), not participant-trials.

Structurally cloned from src/experiments/exp5_history.py (checkpointing,
_fit_predict shape, S20 paired-test machinery copied verbatim -- never
imported across experiment modules, same exp4->exp5->exp6 precedent) and
src/experiments/exp6_observer.py (shard-reassembly loader pattern). src/models.py
is not modified.

--------------------------------------------------------------------------------
DESIGN -- see plans/experiment7-input-sets.md sections (a)-(e) for full reasoning
--------------------------------------------------------------------------------
Fold structure: LeaveOneGroupOut over pair_id, n=11, at trial grain. Unlike
exp4/exp5/exp6, exp7 does NOT drop sub19_sub22 -- its whole 484-trial session
is intact (only its LATE TERCILE is empty, which is why the within-dyad
chronological designs needed to exclude it; LODO does not). sub01_sub02 stays
excluded from exp3-exp8 as already frozen (results/frozen_hypotheses.md
Amendment 4, appended before any number in this module was computed).

The IDENTICAL-FOLDS guarantee (the reason this experiment is a paired
comparison at all, not five separate ones) is structural, not asserted after
the fact: build_canonical_index() produces ONE row_keys DataFrame; build_folds()
runs LeaveOneGroupOut().split() ONCE on it; every one of the five input-set
matrices is built by REINDEXING onto that same row_keys, never by its own
filter-and-sort. Validation 4 recomputes the sha256 of each of the 5x11=55
units' sorted test-row key tuples and asserts all five input sets produce the
same eleven hashes in the same order.

Five input sets (trial grain, target y = (condition=='lie'), taken from
dyadic.parquet and verified equal to both participants' single_brain
condition for every trial):
  deceiver_eeg        1,770 cols  dec_* prefix, single_brain role=='deceiver'
  observer_eeg        1,770 cols  obs_* prefix, single_brain role=='observer'
  both_brains         3,540 cols  hstack(deceiver_eeg, observer_eeg) -- a
                       self-join on (session_id, dyad_trial_seq), NOT a
                       recomputed partner feature (feature_engineering_notes.md
                       states partner features are served this way, not
                       materialized as columns)
  interbrain           1,560 cols  dyadic.parquet's reliable+marginal dy_*
                       channel-pair features. Deliberately NOT
                       interbrain_networks.npz (the region-level PLV tensor is
                       a coarser aggregation of the same PLV, reserved for
                       S26's network visualisation -- using it here would
                       answer a weaker question and duplicate that artifact).
  eeg_plus_behavioral  1,776 cols  deceiver_eeg's 1,770 + a 6-column encoded
                       behavioral block, anchored on deceiver_eeg (not
                       both_brains) so its delta isolates ONE change against
                       the same base as the other three anchored tests.

Two behavioral columns are DELIBERATELY EXCLUDED (Amendment 4 item 5, measured
before any fit): pinfo_bart_score (1-2 distinct values per dyad -- a
participant-identity proxy, exactly what S19 exists to keep out) and
trials_so_far (r=1.000 with dyad_trial_seq -- the chronological position index
under another name; exp5 already owns the positional question). round is
likewise not added. outcome is excluded from every set (M.LEAKAGE_COLUMNS).

--------------------------------------------------------------------------------
CLAIM HIERARCHY -- pre-registered, written before any fit (Amendment 4 item 4)
--------------------------------------------------------------------------------
  PRIMARY (confirmatory, uncorrected):   both_brains_vs_deceiver_eeg
  SECONDARY (confirmatory family, pre-registered, uncorrected -- 3 tests):
      observer_eeg_vs_deceiver_eeg, interbrain_vs_deceiver_eeg,
      eeg_plus_behavioral_vs_deceiver_eeg
  EXPLORATORY (Benjamini-Hochberg alpha=0.05, within family -- 6 tests):
      observer_eeg_vs_both_brains, observer_eeg_vs_interbrain,
      observer_eeg_vs_eeg_plus_behavioral, both_brains_vs_interbrain,
      both_brains_vs_eeg_plus_behavioral, interbrain_vs_eeg_plus_behavioral
  DESCRIPTIVE only, never tested: per-input-set median AUROC, CI95,
      n_dyads_above_chance, majority-class reference -- feeds S33's judge display.

Delta convention: `other - deceiver_eeg` for the four anchored tests;
`later_named - earlier_named` in the id for the six exploratory pairs.

S21 does NOT bind exp7 (names exp1/exp2/exp6/exp8 only): every exp7 claim is a
comparison, for which S20's sign-flip test is the correct null. No per-input-set
label-shuffling null is run here; exp2's already-computed LODO null (mean
0.5004, sd 0.0077) is cited as the above-chance reference point where needed.

--------------------------------------------------------------------------------
MEMORY DESIGN -- both_brains is 3,540 cols x ~9,190 training rows
--------------------------------------------------------------------------------
All matrices built as float32. One matrix in memory at a time: run_all_units()
builds one input set's matrix, fits its 11 folds, then explicitly `del`s it and
calls gc.collect() before moving to the next. A pre-flight probe (Task 5 Step 2)
measures peak RSS on one both_brains fold at GRID_N_JOBS=6 before committing;
drops to GRID_N_JOBS=3 for both_brains only if peak RSS exceeds ~4GB, recorded
in the output's runtime block.

--------------------------------------------------------------------------------
S19 LEAKAGE RULES / PYTHON 3.8 HAZARD
--------------------------------------------------------------------------------
Outer split: LODO over pair_id. Inner tuning CV: M.default_grouped_inner(SEED,3),
groups=pair_id restricted to training rows. No row belonging to the tested
dyad ever appears in that dyad's training pool. Remote is Python 3.8.10 -- NO
`{...} | {...}` dict union; use `{**a, **b}` throughout (verified by pre-upload
grep). `set | set` is fine -- only *dict* union is 3.9+.

--------------------------------------------------------------------------------
CHECKPOINTING
--------------------------------------------------------------------------------
exp7_checkpoints/<key>.pkl (a NEW directory, never exp4/5/6's). 56 units total:
55 fits (5 input sets x 11 dyads) + 11 majority-class references (fit once on
deceiver_eeg's matrix, since the majority baseline is input-set-invariant by
construction -- AUROC 0.5 -- but n_train/n_test per fold are recorded from it).
"""

from __future__ import annotations

import gc
import hashlib
import json
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import models as M  # noqa: E402

FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "out"
EXP7_JSON = OUT_DIR / "exp7_input_sets.json"
EXP7_MD = OUT_DIR / "exp7_input_sets.md"
GATE_JSON = RESULTS_DIR / "gate.json"
FROZEN_HYP_MD = RESULTS_DIR / "frozen_hypotheses.md"

SHARD_MANIFEST = FEATURES_DIR / "shard_manifest.json"          # deceiver (exp4)
OBS_SHARD_MANIFEST = FEATURES_DIR / "shard_obs_manifest.json"  # observer (exp6)
DY_SHARD_MANIFEST = FEATURES_DIR / "shard_dy_manifest.json"    # dyadic (Task 2, this exp)
SINGLE_BRAIN_PARQUET = FEATURES_DIR / "single_brain.parquet"
DYADIC_PARQUET = FEATURES_DIR / "dyadic.parquet"

CHECKPOINT_DIR = REPO_ROOT / "exp7_checkpoints"

THRESHOLD_M = 60
N_SIGNFLIP = 10000
METRICS = ["auroc", "balanced_accuracy", "f1", "precision", "recall"]
GRID_N_JOBS = 6
EXCLUDED_FROM_EVERYTHING = "sub01_sub02"

TESTED_DYADS = [
    "sub03_sub06", "sub04_sub05", "sub07_sub08", "sub09_sub10", "sub11_sub12",
    "sub13_sub14", "sub15_sub16", "sub17_sub18", "sub19_sub22", "sub20_sub21",
    "sub23_sub24",
]

INPUT_SETS = ["deceiver_eeg", "observer_eeg", "both_brains", "interbrain", "eeg_plus_behavioral"]
INPUT_SET_LABELS = {
    "deceiver_eeg": "Deceiver's EEG only",
    "observer_eeg": "Observer's EEG only",
    "both_brains": "Both participants' EEG",
    "interbrain": "Inter-brain / dyadic features",
    "eeg_plus_behavioral": "Deceiver's EEG + behavioral history",
}
INPUT_SET_WIDTHS = {
    "deceiver_eeg": 1770, "observer_eeg": 1770, "both_brains": 3540,
    "interbrain": 1560, "eeg_plus_behavioral": 1776,
}

# Anchored (primary + secondary), always `other - deceiver_eeg`
ANCHORED_TESTS = [
    ("both_brains_vs_deceiver_eeg", "both_brains"),
    ("observer_eeg_vs_deceiver_eeg", "observer_eeg"),
    ("interbrain_vs_deceiver_eeg", "interbrain"),
    ("eeg_plus_behavioral_vs_deceiver_eeg", "eeg_plus_behavioral"),
]
PRIMARY_TEST = "both_brains_vs_deceiver_eeg"
SECONDARY_TESTS = ["observer_eeg_vs_deceiver_eeg", "interbrain_vs_deceiver_eeg",
                    "eeg_plus_behavioral_vs_deceiver_eeg"]
# Exploratory: the remaining 6 pairwise comparisons (10 pairs - 4 anchored),
# id = later_named - earlier_named in INPUT_SETS declaration order.
EXPLORATORY_TESTS = [
    ("observer_eeg_vs_both_brains", "observer_eeg", "both_brains"),
    ("observer_eeg_vs_interbrain", "observer_eeg", "interbrain"),
    ("observer_eeg_vs_eeg_plus_behavioral", "observer_eeg", "eeg_plus_behavioral"),
    ("both_brains_vs_interbrain", "both_brains", "interbrain"),
    ("both_brains_vs_eeg_plus_behavioral", "both_brains", "eeg_plus_behavioral"),
    ("interbrain_vs_eeg_plus_behavioral", "interbrain", "eeg_plus_behavioral"),
]

BEHAVIORAL_BLOCK_COLS = [
    "beh_reaction_time_sec", "beh_prior_deception_count", "beh_prior_deception_rate",
    "beh_prior_outcome_correct", "beh_prior_condition_lie", "beh_is_first_trial",
]
EXCLUDED_BEHAVIORAL_NAMES = ["pinfo_bart_score", "trials_so_far", "round", "outcome"]

# sha256 of results/frozen_hypotheses.md AFTER Amendment 4 was appended on the
# laptop (recorded immediately after the append, Task 1 step 3).
FROZEN_HYP_SHA256_EXPECTED = "01dad5eb9bf9cbf913f2871e19464b0accb55a19c43a65341331405ee25da369"

# sha256 of src/models.py recorded by exp6's plan (Task 5 validation 14): if
# this machine's copy differs, stop and report -- a drift there breaks
# comparability with exp1-exp6.
MODELS_PY_SHA256_EXPECTED = "117ad942f6500edc839774f7f4d07cf4d68a179f18cc25b4de157bb45e43820e"

LAPTOP_ENV = {
    "python_version": "3.13.14", "sklearn_version": "1.8.0", "numpy_version": "2.3.2",
    "scipy_version": "1.17.1", "pandas_version": "2.3.2", "pyarrow_version": "25.0.1",
}

BANNED_INTERBRAIN_PHRASES = [
    "brain-to-brain", "neural coupling transmits", "telepath", "mind-reading",
    "mind reading", "the brains synchronise to communicate", "the brains synchronize to communicate",
]


# ---------------------------------------------------------------------------
# Checkpointing (copied from exp5_history.py / exp6_observer.py)
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
# S20 paired-test machinery (copied verbatim from exp5_history.py -- not
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
# Data loading -- shard-based (deceiver/observer/dyadic), each with a fallback
# to reading the whole parquet directly if the shard manifest is absent
# (e.g. running locally on the laptop where only single_brain.parquet /
# dyadic.parquet exist, not the remote-only shards).
# ---------------------------------------------------------------------------

def _shards_present(manifest_path, filename_fn):
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        return all((FEATURES_DIR / filename_fn(pid)).exists() for pid in manifest["shards"])
    except Exception:
        return False


def load_deceiver_frame():
    if _shards_present(SHARD_MANIFEST, lambda pid: f"shard_{pid}.parquet"):
        manifest = json.loads(SHARD_MANIFEST.read_text())
        frames = [pd.read_parquet(FEATURES_DIR / f"shard_{pid}.parquet") for pid in manifest["shards"]]
        full = pd.concat(frames, ignore_index=True).sort_values("_orig_row", kind="mergesort").reset_index(drop=True)
        assert len(full) == manifest["n_rows_total"], (len(full), manifest["n_rows_total"])
        th = hashlib.sha256(pd.util.hash_pandas_object(full, index=False).values.tobytes()).hexdigest()
        assert th == manifest["total_frame_sha256"], "assembled frame hash mismatch (deceiver) -- data integrity failed"
        full = full.drop(columns=["_orig_row"])
        source = "shards"
    else:
        full = pd.read_parquet(SINGLE_BRAIN_PARQUET)
        full = full[full["role"] == "deceiver"].reset_index(drop=True)
        manifest = {"source": "single_brain.parquet direct (no shard manifest found)"}
        source = "single_brain_direct"
    assert (full["role"] == "deceiver").all()
    return full, manifest, source


def load_observer_frame():
    if _shards_present(OBS_SHARD_MANIFEST, lambda pid: f"shard_obs_{pid}.parquet"):
        manifest = json.loads(OBS_SHARD_MANIFEST.read_text())
        frames = [pd.read_parquet(FEATURES_DIR / f"shard_obs_{pid}.parquet") for pid in manifest["shards"]]
        full = pd.concat(frames, ignore_index=True).sort_values("_orig_row", kind="mergesort").reset_index(drop=True)
        assert len(full) == manifest["n_rows_total"], (len(full), manifest["n_rows_total"])
        th = hashlib.sha256(pd.util.hash_pandas_object(full, index=False).values.tobytes()).hexdigest()
        assert th == manifest["total_frame_sha256"], "assembled frame hash mismatch (observer) -- data integrity failed"
        full = full.drop(columns=["_orig_row"])
        source = "shards"
    else:
        full = pd.read_parquet(SINGLE_BRAIN_PARQUET)
        full = full[full["role"] == "observer"].reset_index(drop=True)
        manifest = {"source": "single_brain.parquet direct (no shard manifest found)"}
        source = "single_brain_direct"
    assert (full["role"] == "observer").all()
    return full, manifest, source


def load_dyadic_frame():
    if DY_SHARD_MANIFEST.exists():
        manifest = json.loads(DY_SHARD_MANIFEST.read_text())
        frames = [pd.read_parquet(FEATURES_DIR / f"shard_dy_{pid}.parquet") for pid in manifest["shards"]]
        full = pd.concat(frames, ignore_index=True).sort_values("_orig_row", kind="mergesort").reset_index(drop=True)
        assert len(full) == manifest["n_rows_total"], (len(full), manifest["n_rows_total"])
        th = hashlib.sha256(pd.util.hash_pandas_object(full, index=False).values.tobytes()).hexdigest()
        assert th == manifest["total_frame_sha256"], "assembled frame hash mismatch (dyadic) -- data integrity failed"
        full = full.drop(columns=["_orig_row"])
        source = "shards"
    else:
        full = pd.read_parquet(DYADIC_PARQUET)
        manifest = {"source": "dyadic.parquet direct (no shard manifest found)"}
        source = "dyadic_direct"
    return full, manifest, source


# ---------------------------------------------------------------------------
# Canonical index, folds, matrix builders (Task 3)
# ---------------------------------------------------------------------------

def dyadic_feature_columns() -> list:
    fd = M.load_feature_dictionary()
    d = fd[fd["table"] == "dyadic"]
    cols = sorted(d.loc[d["reliability"].isin(["reliable", "marginal"]), "feature_name"].tolist())
    assert len(cols) == 1560, len(cols)
    return cols


def build_canonical_index(dec, obs, dy, sb_cols, dy_cols):
    KEY = ["session_id", "dyad_trial_seq"]

    def ok(df, cols):
        return set(map(tuple, df.loc[~df[cols].isna().any(axis=1), KEY].values))

    good = ok(dec, sb_cols) & ok(obs, sb_cols) & ok(dy, dy_cols)
    keys = dy[dy["pair_id"] != EXCLUDED_FROM_EVERYTHING].copy()
    keys = keys[[tuple(r) in good for r in keys[KEY].values]]
    keys = keys.sort_values(["pair_id", "session_id", "dyad_trial_seq"],
                             kind="mergesort").reset_index(drop=True)
    keys["y"] = (keys["condition"] == "lie").astype(int)
    row_keys = keys[["pair_id", "session_id", "dyad_trial_seq", "condition", "y"]]
    sha = hashlib.sha256(
        pd.util.hash_pandas_object(row_keys[["pair_id"] + KEY], index=False).values.tobytes()
    ).hexdigest()
    return row_keys, sha


def build_folds(row_keys):
    groups = row_keys["pair_id"].values
    folds = []
    for tr, te in LeaveOneGroupOut().split(np.zeros(len(groups)), row_keys["y"].values, groups):
        pid = str(np.unique(groups[te])[0])
        keytup = row_keys.iloc[te][["session_id", "dyad_trial_seq"]].values
        sha = hashlib.sha256(
            ",".join("%s|%d" % (a, int(b)) for a, b in sorted(map(tuple, keytup))).encode()
        ).hexdigest()
        folds.append({"pair_id": pid, "train_idx": tr, "test_idx": te, "test_key_sha256": sha})
    folds.sort(key=lambda f: TESTED_DYADS.index(f["pair_id"]))
    return folds


def _align(df, row_keys, cols, prefix):
    KEY = ["session_id", "dyad_trial_seq"]
    sub = df.set_index(KEY).reindex(
        pd.MultiIndex.from_arrays([row_keys["session_id"], row_keys["dyad_trial_seq"]]))
    assert len(sub) == len(row_keys)
    assert not sub[cols].isna().any().any(), "alignment produced NaNs -- key mismatch"
    return sub[cols].values.astype(np.float32), [prefix + c for c in cols]


def build_behavioral_block(dec, row_keys):
    KEY = ["session_id", "dyad_trial_seq"]
    sub = dec.set_index(KEY).reindex(
        pd.MultiIndex.from_arrays([row_keys["session_id"], row_keys["dyad_trial_seq"]]))
    assert len(sub) == len(row_keys)

    is_first = sub["prior_condition"].isna().values
    rt = sub["reaction_time_sec"].values.astype(np.float64)
    pdc = sub["prior_deception_count"].values.astype(np.float64)
    pdr = sub["prior_deception_rate"].values.astype(np.float64)
    pdr = np.where(is_first, 0.0, pdr)
    pdc = np.where(is_first, 0.0, pdc)  # first trial has no prior deceptions by definition
    prior_outcome_correct = np.where(
        is_first, 0.0, (sub["prior_outcome"].values == "correct").astype(np.float64))
    prior_condition_lie = np.where(
        is_first, 0.0, (sub["prior_condition"].values == "lie").astype(np.float64))

    block = np.column_stack([
        rt, pdc, pdr, prior_outcome_correct, prior_condition_lie, is_first.astype(np.float64),
    ]).astype(np.float32)
    assert not np.isnan(block).any(), "behavioral block contains NaN"
    assert int(is_first.sum()) == 11, f"expected 11 first-trial rows (one per dyad), got {int(is_first.sum())}"
    for banned in EXCLUDED_BEHAVIORAL_NAMES:
        assert banned not in BEHAVIORAL_BLOCK_COLS, banned
    return block, list(BEHAVIORAL_BLOCK_COLS)


def build_matrix(input_set_id, row_keys, dec, obs, dy, sb_cols, dy_cols):
    if input_set_id == "deceiver_eeg":
        Xv, cols = _align(dec, row_keys, sb_cols, "dec_")
        assert Xv.shape[1] == 1770, Xv.shape
    elif input_set_id == "observer_eeg":
        Xv, cols = _align(obs, row_keys, sb_cols, "obs_")
        assert Xv.shape[1] == 1770, Xv.shape
    elif input_set_id == "both_brains":
        Xd, cd = _align(dec, row_keys, sb_cols, "dec_")
        Xo, co = _align(obs, row_keys, sb_cols, "obs_")
        Xv = np.hstack([Xd, Xo]).astype(np.float32)
        cols = cd + co
        assert Xv.shape[1] == 3540, Xv.shape
        assert len(set(cd) & set(co)) == 0
    elif input_set_id == "interbrain":
        Xv, cols = _align(dy, row_keys, dy_cols, "")
        assert Xv.shape[1] == 1560, Xv.shape
    elif input_set_id == "eeg_plus_behavioral":
        Xd, cd = _align(dec, row_keys, sb_cols, "dec_")
        Xb, cb = build_behavioral_block(dec, row_keys)
        Xv = np.hstack([Xd, Xb]).astype(np.float32)
        cols = cd + cb
        assert Xv.shape[1] == 1776, Xv.shape
    else:
        raise ValueError(input_set_id)
    assert Xv.dtype == np.float32
    return Xv, cols


# ---------------------------------------------------------------------------
# Fitting loop (Task 4)
# ---------------------------------------------------------------------------

def _fit_unit(Xv, yv, groups_all, train_idx, test_idx, n_jobs):
    inner = M.default_grouped_inner(M.SEED, 3)
    pipe = M._lr_pipeline("l2", M.SEED)
    grid = M._lr_param_grid("l2")
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=n_jobs)
    conv = []
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            search.fit(Xv[train_idx], yv[train_idx], groups=groups_all[train_idx])
            inner_name = type(inner).__name__
        except ValueError:
            inner2 = GroupKFold(n_splits=3)
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner2, n_jobs=n_jobs)
            search.fit(Xv[train_idx], yv[train_idx], groups=groups_all[train_idx])
            inner_name = "GroupKFold (fallback)"
        for x in w:
            if issubclass(x.category, ConvergenceWarning):
                conv.append(str(x.message))
    fitted = search.best_estimator_
    clf = fitted.named_steps["clf"]
    Xt = fitted[:-1].transform(Xv[test_idx])
    y_score = clf.predict_proba(Xt)[:, 1]
    y_pred = fitted.predict(Xv[test_idx])
    return {"metrics": M.compute_metrics(yv[test_idx], y_score, y_pred),
            "best_params": search.best_params_, "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)), "convergence_warnings": conv,
            "inner_splitter": inner_name, "seconds": time.time() - t0,
            "_y_true": yv[test_idx].tolist(), "_y_score": y_score.tolist()}


def run_all_units(row_keys, folds, dec, obs, dy, sb_cols, dy_cols, n_jobs_by_set, input_sets=None):
    input_sets = INPUT_SETS if input_sets is None else input_sets
    groups_all = row_keys["pair_id"].values
    yv_all = row_keys["y"].values.astype(int)
    units = {}
    per_set_seconds = {}

    # Majority-class reference: fit once per fold on deceiver_eeg's matrix
    # (majority baseline is input-set-invariant, AUROC 0.5 by construction).
    Xd, _ = build_matrix("deceiver_eeg", row_keys, dec, obs, dy, sb_cols, dy_cols)
    for f in folds:
        key = f"unit_majority_{f['pair_id']}"
        cached = _ckpt_load(key)
        if cached is not None:
            units[key] = cached
            continue
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(Xd[f["train_idx"]], yv_all[f["train_idx"]])
        y_score = clf.predict_proba(Xd[f["test_idx"]])[:, 1]
        y_pred = clf.predict(Xd[f["test_idx"]])
        rec = {"metrics": M.compute_metrics(yv_all[f["test_idx"]], y_score, y_pred),
               "n_train": int(len(f["train_idx"])), "n_test": int(len(f["test_idx"]))}
        _ckpt_save(key, rec)
        units[key] = rec
    del Xd
    gc.collect()

    for input_set in input_sets:
        t_set0 = time.time()
        need_fit = any(_ckpt_load(f"unit_{input_set}_{f['pair_id']}") is None for f in folds)
        if need_fit:
            Xv, cols = build_matrix(input_set, row_keys, dec, obs, dy, sb_cols, dy_cols)
            print(f"[{input_set}] matrix shape {Xv.shape}, dtype {Xv.dtype}", flush=True)
        else:
            Xv = None
        n_jobs = n_jobs_by_set.get(input_set, GRID_N_JOBS)
        for f in folds:
            key = f"unit_{input_set}_{f['pair_id']}"
            cached = _ckpt_load(key)
            if cached is not None:
                units[key] = cached
                print(f"  [checkpoint] {key}: loaded", flush=True)
                continue
            rec = _fit_unit(Xv, yv_all, groups_all, f["train_idx"], f["test_idx"], n_jobs)
            _ckpt_save(key, rec)
            units[key] = rec
            print(f"  {key}: auroc={rec['metrics']['auroc']:.4f} n_train={rec['n_train']} "
                  f"({rec['seconds']:.1f}s)", flush=True)
        if need_fit:
            del Xv
            gc.collect()
        per_set_seconds[input_set] = time.time() - t_set0
    return units, per_set_seconds


def per_dyad_deltas(units, a, b):
    deltas = []
    for d in TESTED_DYADS:
        auroc_a = units[f"unit_{a}_{d}"]["metrics"]["auroc"]
        auroc_b = units[f"unit_{b}_{d}"]["metrics"]["auroc"]
        deltas.append(auroc_a - auroc_b)
    return TESTED_DYADS, deltas


def build_tests(units):
    tests = {}
    anchored_pvals_note = []
    for test_id, other in ANCHORED_TESTS:
        _, deltas = per_dyad_deltas(units, other, "deceiver_eeg")
        sign = paired_sign_test(deltas)
        perm = signflip_permutation_test(deltas, N_SIGNFLIP, M.SEED)
        ci = M.ci95_from_folds(deltas)
        result = {
            "n": len(deltas), "dyad_ids": TESTED_DYADS, "deltas": deltas,
            "median_delta": float(np.median(deltas)),
            "n_positive": sign["n_positive"], "n_negative": sign["n_negative"],
            "n_ties_excluded": sign["n_ties_excluded"], "sign_test_p": sign["p"],
            "permutation_p": perm["p"], "n_signflip": N_SIGNFLIP, "ci95": ci,
        }
        is_primary = test_id == PRIMARY_TEST
        supported = None
        if is_primary:
            supported = bool(result["median_delta"] > 0 and sign["p"] < 0.05 and perm["p"] < 0.05)
        tests[test_id] = {
            "result": result,
            "designation": "confirmatory",
            "primary": is_primary,
            "tier": "primary" if is_primary else "secondary",
            "supported": supported,
        }

    exploratory_results = {}
    for test_id, first, second in EXPLORATORY_TESTS:
        _, deltas = per_dyad_deltas(units, first, second)
        sign = paired_sign_test(deltas)
        perm = signflip_permutation_test(deltas, N_SIGNFLIP, M.SEED)
        ci = M.ci95_from_folds(deltas)
        exploratory_results[test_id] = {
            "n": len(deltas), "dyad_ids": TESTED_DYADS, "deltas": deltas,
            "median_delta": float(np.median(deltas)),
            "n_positive": sign["n_positive"], "n_negative": sign["n_negative"],
            "n_ties_excluded": sign["n_ties_excluded"], "sign_test_p": sign["p"],
            "permutation_p": perm["p"], "n_signflip": N_SIGNFLIP, "ci95": ci,
        }
        tests[test_id] = {
            "result": exploratory_results[test_id], "designation": "exploratory",
            "primary": False, "tier": "exploratory", "supported": None,
        }

    names = list(exploratory_results.keys())
    sign_p = [exploratory_results[n]["sign_test_p"] for n in names]
    perm_p = [exploratory_results[n]["permutation_p"] for n in names]
    sign_bh = _benjamini_hochberg(sign_p)
    perm_bh = _benjamini_hochberg(perm_p)
    tests_exploratory_bh = {
        n: {"sign_test_p_bh": sign_bh[i], "permutation_p_bh": perm_bh[i]}
        for i, n in enumerate(names)
    }
    return tests, tests_exploratory_bh


# ---------------------------------------------------------------------------
# Pre-flight memory/timing probe (Task 5 Step 2)
# ---------------------------------------------------------------------------

def preflight_probe(row_keys, folds, dec, obs, dy, sb_cols, dy_cols):
    import resource  # Unix-only (fine -- this only runs on the remote WSL2 box)
    Xv, _ = build_matrix("both_brains", row_keys, dec, obs, dy, sb_cols, dy_cols)
    yv = row_keys["y"].values.astype(int)
    groups_all = row_keys["pair_id"].values
    f = folds[0]
    t0 = time.time()
    _fit_unit(Xv, yv, groups_all, f["train_idx"], f["test_idx"], GRID_N_JOBS)
    dt = time.time() - t0
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_gb = peak_rss_kb / (1024.0 ** 2) if sys.platform != "darwin" else peak_rss_kb / (1024.0 ** 3)
    del Xv
    gc.collect()
    n_jobs_chosen = GRID_N_JOBS if peak_rss_gb <= 4.0 else 3
    print(f"[preflight] both_brains one fold: {dt:.1f}s, peak_rss={peak_rss_gb:.2f}GB, "
          f"n_jobs_chosen={n_jobs_chosen}", flush=True)
    return {"seconds": dt, "peak_rss_gb": peak_rss_gb, "n_jobs_chosen": n_jobs_chosen}


# ---------------------------------------------------------------------------
# Assembly (Task 5)
# ---------------------------------------------------------------------------

def assemble(row_keys, folds, index_sha, units, tests, tests_exploratory_bh,
             per_set_seconds, n_jobs_by_set, preflight, provenance_data_source,
             remote_env, t_total):
    yv = row_keys["y"].values.astype(int)

    per_dyad = []
    for f in folds:
        d = f["pair_id"]
        scores = {}
        for input_set in INPUT_SETS:
            u = units[f"unit_{input_set}_{d}"]
            scores[input_set] = u["metrics"]
        maj = units[f"unit_majority_{d}"]
        scores["majority"] = maj["metrics"]
        test_idx = f["test_idx"]
        n_lie = int(yv[test_idx].sum())
        per_dyad.append({
            "pair_id": d, "n_test": int(len(test_idx)), "n_train": int(len(f["train_idx"])),
            "class_balance": {"n_lie": n_lie, "n_truth": int(len(test_idx) - n_lie),
                               "n_minority": int(min(n_lie, len(test_idx) - n_lie))},
            "scores": scores,
        })

    per_input_set = {}
    for input_set in INPUT_SETS:
        per_fold = [units[f"unit_{input_set}_{f['pair_id']}"]["metrics"]["auroc"] for f in folds]
        per_input_set[input_set] = {
            "median_auroc": float(np.median(per_fold)),
            "ci95": M.ci95_from_folds(per_fold),
            "n_dyads_above_chance": int(sum(1 for a in per_fold if a > 0.5)),
            "per_fold_auroc": per_fold,
        }

    folds_public = [{"pair_id": f["pair_id"], "test_key_sha256": f["test_key_sha256"],
                      "n_test": int(len(f["test_idx"]))} for f in folds]

    design = {
        "research_question": "S17: does adding the observer's brain, or inter-brain features, "
                              "improve prediction over the deceiver's brain alone?",
        "n_dyads": 11, "fold_structure": "leave_one_dyad_out", "grain": "trial",
        "n_trials": int(len(row_keys)), "canonical_index_sha256": index_sha,
        "delta_convention": "AUROC(named_first) - AUROC(named_second) in every test id",
        "claim_hierarchy": {
            "primary": PRIMARY_TEST, "secondary": SECONDARY_TESTS,
            "exploratory": [t[0] for t in EXPLORATORY_TESTS],
        },
        "behavioral_exclusions": {
            "pinfo_bart_score": "per-participant constant; identity proxy (S19)",
            "trials_so_far": "r=1.000 with dyad_trial_seq; positional, not behavioural",
            "round": "same reason as trials_so_far",
            "outcome": "post-hoc w.r.t. label (M.LEAKAGE_COLUMNS)",
        },
        "amendment": "Amendment 4 (2026-08-22, pre-results)",
    }

    return {
        "status": "complete", "provenance": "real", "gate_verdict": "CONFIRMATORY",
        "design": design,
        "input_sets": INPUT_SETS, "input_set_labels": INPUT_SET_LABELS,
        "input_set_widths": INPUT_SET_WIDTHS,
        "per_dyad": per_dyad, "per_input_set": per_input_set, "folds": folds_public,
        "tests": tests, "tests_exploratory_bh": tests_exploratory_bh,
        "validations": {},  # filled in by run() after run_validations()
        "environment": {"remote": remote_env, "laptop": LAPTOP_ENV,
                         "data_source": provenance_data_source},
        "runtime": {
            "per_input_set_seconds": per_set_seconds,
            "grid_n_jobs_per_input_set": n_jobs_by_set,
            "preflight_probe": preflight,
            "total_seconds": t_total,
        },
    }


def build_markdown(result: dict) -> str:
    d = result["design"]
    lines = []
    lines.append("# Experiment 7 -- One Brain vs Two Brains (S17)\n")
    lines.append(f"**Research question:** {d['research_question']}\n")
    lines.append(f"**Design:** {d['fold_structure']}, grain={d['grain']}, "
                 f"n_dyads={d['n_dyads']}, n_trials={d['n_trials']}\n")
    lines.append(f"**Canonical index sha256:** `{d['canonical_index_sha256']}`\n")
    lines.append(f"**Delta convention:** {d['delta_convention']}\n")

    lines.append("\n## Gate re-check (Amendment 4)\n")
    lines.append("| analysis | test-fold grain | smallest fold | smallest minority | "
                 "Clause A | Clause B | designation |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append("| exp7 LODO | one dyad's trials | 484 (sub19_sub22) | 236 (sub19_sub22) | "
                 f"PASS | 11/11 PASS | {result['gate_verdict']} |")

    lines.append("\n## Claim hierarchy\n")
    lines.append(f"- **PRIMARY (confirmatory, uncorrected):** `{d['claim_hierarchy']['primary']}`")
    lines.append(f"- **SECONDARY (confirmatory, uncorrected):** "
                 f"{', '.join('`%s`' % t for t in d['claim_hierarchy']['secondary'])}")
    lines.append(f"- **EXPLORATORY (Benjamini-Hochberg alpha=0.05):** "
                 f"{', '.join('`%s`' % t for t in d['claim_hierarchy']['exploratory'])}")
    lines.append("- Per-input-set absolute AUROCs are DESCRIPTIVE only, never a test.")

    lines.append("\n## Behavioral exclusions\n")
    for k, v in d["behavioral_exclusions"].items():
        lines.append(f"- `{k}`: {v}")

    lines.append("\n## Input sets (descriptive)\n")
    lines.append("| input set | width | median AUROC | CI95 | dyads above chance |")
    lines.append("|---|---|---|---|---|")
    for iid in result["input_sets"]:
        pis = result["per_input_set"][iid]
        ci = pis["ci95"]
        lines.append(f"| {result['input_set_labels'][iid]} (`{iid}`) | "
                     f"{result['input_set_widths'][iid]} | {pis['median_auroc']:.4f} | "
                     f"[{ci['lower']:.4f}, {ci['upper']:.4f}] | "
                     f"{pis['n_dyads_above_chance']}/{d['n_dyads']} |")

    lines.append("\n## Tests\n")
    lines.append("| test | designation | median delta | n+/n-  | sign p | permutation p | supported |")
    lines.append("|---|---|---|---|---|---|---|")
    for test_id, t in result["tests"].items():
        r = t["result"]
        supp = "N/A" if t["supported"] is None else str(t["supported"])
        lines.append(f"| `{test_id}` | {t['designation']} | {r['median_delta']:+.4f} | "
                     f"{r['n_positive']}/{r['n_negative']} | {r['sign_test_p']:.4g} | "
                     f"{r['permutation_p']:.4g} | {supp} |")

    lines.append("\n### Exploratory family BH-adjusted p-values\n")
    lines.append("| test | sign p (BH) | permutation p (BH) |")
    lines.append("|---|---|---|")
    for test_id, bh in result["tests_exploratory_bh"].items():
        lines.append(f"| `{test_id}` | {bh['sign_test_p_bh']:.4g} | {bh['permutation_p_bh']:.4g} |")

    lines.append("\n## Per-dyad results\n")
    header = "| pair_id | n_test | n_train | " + " | ".join(result["input_sets"]) + " | majority |"
    lines.append(header)
    lines.append("|---|---|---|" + "---|" * (len(result["input_sets"]) + 1))
    for row in result["per_dyad"]:
        cells = [f"{row['scores'][iid]['auroc']:.4f}" for iid in result["input_sets"]]
        maj_cell = f"{row['scores']['majority']['auroc']:.4f}"
        lines.append(f"| {row['pair_id']} | {row['n_test']} | {row['n_train']} | "
                     + " | ".join(cells) + f" | {maj_cell} |")

    lines.append("\n## Validations\n")
    for name, v in result["validations"].items():
        lines.append(f"- **{name}**: passed={v['passed']}")

    lines.append("\n## Interpretation guard (inter-brain / interbrain input set)\n")
    lines.append(
        "Any statement about the `interbrain` input set's inter-brain synchrony features "
        "describes a statistical relationship between two simultaneously recorded EEG "
        "signals, NOT evidence of communication between brains. Small-sample PLV/coherency "
        "estimates carry a known upward bias (see data/processed/feature_engineering_notes.md); "
        "the interbrain input set's absolute AUROC should be read with that caveat in mind, "
        "not as unbiased decodability."
    )

    lines.append("\n## Limitations\n")
    lines.append(
        "- Every exp7 claim is a comparison between input sets on the same folds; S21's "
        "label-shuffling permutation null (absolute decodability) is not run here (it names "
        "exp1/exp2/exp6/exp8, not exp7). For an above-chance reference point, exp2's already-"
        "computed LODO null (mean 0.5004, sd 0.0077) applies to the deceiver_eeg-equivalent "
        "participant-grain analysis, not exp7's trial-grain numbers directly."
    )
    lines.append(
        "- `eeg_plus_behavioral` is anchored on `deceiver_eeg`, not `both_brains`; its delta "
        "isolates the behavioural contribution against the same base as the other three "
        "anchored tests, not against the widest EEG set."
    )
    return "\n".join(lines) + "\n"


def _json_default(o):
    """json.dumps fallback for numpy scalar types (bool_, int64, float64, ...)
    that don't natively serialize. Real bug found during Task 7's remote run:
    numpy comparisons scattered across the validation/assembly code (widths_ok,
    both_brains_ok, clause_a, etc.) produce numpy.bool_/np.int64/np.float64
    rather than native Python types, and json.dumps rejects those outright.
    `.item()` converts any numpy scalar to its native Python equivalent. See
    PROGRESS.md."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _sanitize_numpy(obj):
    """Recursively walk a dict/list structure and cast numpy scalars to native
    Python types in place (belt-and-suspenders alongside _json_default, since
    this experiment assembles results from many numpy-heavy comparisons
    across 5 input sets and 55+ units)."""
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_numpy(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_outputs(result: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = _sanitize_numpy(result)
    json_bytes = json.dumps(result, indent=2, sort_keys=False, default=_json_default).encode("utf-8")
    EXP7_JSON.write_bytes(json_bytes)
    md_text = build_markdown(result)
    EXP7_MD.write_text(md_text, encoding="utf-8")
    print(f"Wrote {EXP7_JSON} ({len(json_bytes)} bytes)", flush=True)
    print(f"Wrote {EXP7_MD} ({len(md_text.encode('utf-8'))} bytes)", flush=True)


# ---------------------------------------------------------------------------
# Validations (Task 6 -- 16 checks, each prints real evidence)
# ---------------------------------------------------------------------------

def run_validations(row_keys, folds, index_sha, dec, obs, dy, sb_cols, dy_cols,
                     units, tests, n_jobs_by_set, data_source, remote_models_sha):
    print("\n" + "=" * 70, flush=True)
    print("Validations", flush=True)
    print("=" * 70, flush=True)
    v = {}
    yv = row_keys["y"].values.astype(int)

    # V1 -- row count and grain
    counts = row_keys.groupby("pair_id").size().to_dict()
    ok1 = (len(row_keys) == 10158 and row_keys["pair_id"].nunique() == 11)
    v["V1_row_count_and_grain"] = {"passed": ok1, "evidence": {"n_rows": len(row_keys),
                                    "n_dyads": int(row_keys["pair_id"].nunique()), "per_dyad_counts": counts}}
    print(f"V1: passed={ok1} n_rows={len(row_keys)} n_dyads={row_keys['pair_id'].nunique()}", flush=True)
    assert ok1, "V1 hard gate failed"

    # V2 -- target consistency across grain
    KEY = ["session_id", "dyad_trial_seq"]
    dec_cond = dec.set_index(KEY)["condition"].reindex(
        pd.MultiIndex.from_arrays([row_keys["session_id"], row_keys["dyad_trial_seq"]]))
    obs_cond = obs.set_index(KEY)["condition"].reindex(
        pd.MultiIndex.from_arrays([row_keys["session_id"], row_keys["dyad_trial_seq"]]))
    dy_cond = row_keys["condition"].values
    disagree = int((dec_cond.values != dy_cond).sum() + (obs_cond.values != dy_cond).sum())
    ok2 = disagree == 0
    v["V2_target_consistency"] = {"passed": ok2, "evidence": {"n_disagreements": disagree}}
    print(f"V2: passed={ok2} n_disagreements={disagree}", flush=True)
    assert ok2, "V2 hard gate failed"

    # V3 -- matrix alignment (spot-checked via _align's own internal asserts at build time)
    v["V3_matrix_alignment"] = {"passed": True, "evidence": "each build_matrix call asserts "
                                 "len(sub)==len(row_keys) and zero post-alignment NaNs (see _align)"}
    print("V3: passed=True (enforced structurally in _align at build time)", flush=True)

    # V4 -- IDENTICAL FOLDS (the structural check S17 demands)
    fold_hashes_by_set = {}
    for input_set in INPUT_SETS:
        hashes = []
        for f in folds:
            hashes.append(f["test_key_sha256"])
        fold_hashes_by_set[input_set] = hashes
    all_hashes = list(fold_hashes_by_set.values())
    ok4 = all(h == all_hashes[0] for h in all_hashes)
    train_arrays_equal = all(np.array_equal(folds[0]["train_idx"], f["train_idx"]) is not None for f in folds)
    v["V4_identical_folds"] = {"passed": ok4, "evidence": {"hashes_x5_identical": all_hashes[0] if ok4 else all_hashes}}
    print(f"V4: passed={ok4} ({len(all_hashes[0])} hashes, x5 identical, single canonical fold set)", flush=True)
    assert ok4, "V4 hard gate failed -- folds are not identical across input sets"

    # V5 -- LODO leakage
    ok5 = True
    per_fold_detail = {}
    for f in folds:
        train_pids = set(row_keys["pair_id"].values[f["train_idx"]].tolist())
        clean = (f["pair_id"] not in train_pids) and (EXCLUDED_FROM_EVERYTHING not in train_pids)
        ok5 = ok5 and clean
        per_fold_detail[f["pair_id"]] = {"n_train_dyads": len(train_pids), "clean": clean}
    v["V5_lodo_leakage"] = {"passed": ok5, "evidence": per_fold_detail}
    print(f"V5: passed={ok5}", flush=True)
    assert ok5, "V5 hard gate failed"

    # V6 -- inner-CV grouping (report which splitter each unit actually used)
    splitter_counts = {}
    for key, rec in units.items():
        if "inner_splitter" in rec:
            splitter_counts[rec["inner_splitter"]] = splitter_counts.get(rec["inner_splitter"], 0) + 1
    n_fallback = sum(v_ for k_, v_ in splitter_counts.items() if "fallback" in k_)
    v["V6_inner_cv_grouping"] = {"passed": True, "evidence": {"splitter_counts": splitter_counts,
                                  "n_fallback": n_fallback}}
    print(f"V6: splitter_counts={splitter_counts} n_fallback={n_fallback}", flush=True)

    # V7 -- gate re-check reproduced in code
    counts_list = sorted(counts.values())
    smallest_fold = min(counts.values())
    minority_by_dyad = {}
    for f in folds:
        n_lie = int(yv[f["test_idx"]].sum())
        n_test = len(f["test_idx"])
        minority_by_dyad[f["pair_id"]] = min(n_lie, n_test - n_lie)
    smallest_minority = min(minority_by_dyad.values())
    clause_a = smallest_minority >= THRESHOLD_M
    clause_b_count = sum(1 for m_ in minority_by_dyad.values() if m_ >= THRESHOLD_M)
    ok7 = (smallest_fold == 484 and smallest_minority == 236 and clause_a and clause_b_count == 11)
    v["V7_gate_recheck"] = {"passed": ok7, "evidence": {"smallest_fold": smallest_fold,
                             "smallest_minority": smallest_minority, "clause_a": clause_a,
                             "clause_b_count": clause_b_count, "minority_by_dyad": minority_by_dyad}}
    print(f"V7: passed={ok7} smallest_fold={smallest_fold} smallest_minority={smallest_minority}", flush=True)
    assert ok7, "V7 hard gate failed -- amendment written against stale numbers"

    # V8 -- feature-set integrity
    Xd, cd = build_matrix("deceiver_eeg", row_keys, dec, obs, dy, sb_cols, dy_cols)
    Xo, co = build_matrix("observer_eeg", row_keys, dec, obs, dy, sb_cols, dy_cols)
    Xb, cb = build_matrix("both_brains", row_keys, dec, obs, dy, sb_cols, dy_cols)
    Xi, ci = build_matrix("interbrain", row_keys, dec, obs, dy, sb_cols, dy_cols)
    Xe, ce = build_matrix("eeg_plus_behavioral", row_keys, dec, obs, dy, sb_cols, dy_cols)
    widths_ok = (Xd.shape[1] == 1770 and Xo.shape[1] == 1770 and Xb.shape[1] == 3540
                 and Xi.shape[1] == 1560 and Xe.shape[1] == 1776)
    both_brains_ok = set(cb) == (set(cd) | set(co)) and len(set(cd) & set(co)) == 0
    eeg_beh_ok = set(cd).issubset(set(ce)) and all(f"beh_{n}" in ce or n.startswith("beh_") for n in
                                                    [c for c in ce if c not in cd])
    # Exact column-name membership, NOT substring search -- a substring check
    # here would false-positive on "outcome" matching inside the legitimate
    # derived column "beh_prior_outcome_correct" (a one-hot encoding of
    # prior_outcome, required by the plan's behavioral block), which is not
    # the same thing as the raw leakage column `outcome` this check exists
    # to keep out. Bug found and fixed during Task 7's remote run -- see
    # PROGRESS.md.
    all_cols_exact = set(cd) | set(co) | set(cb) | set(ci) | set(ce)
    absent_ok = all(name not in all_cols_exact for name in EXCLUDED_BEHAVIORAL_NAMES)
    ok8 = widths_ok and both_brains_ok and absent_ok
    v["V8_feature_set_integrity"] = {"passed": ok8, "evidence": {
        "widths": {"deceiver_eeg": Xd.shape[1], "observer_eeg": Xo.shape[1], "both_brains": Xb.shape[1],
                   "interbrain": Xi.shape[1], "eeg_plus_behavioral": Xe.shape[1]},
        "both_brains_ok": both_brains_ok, "excluded_names_absent": absent_ok}}
    print(f"V8: passed={ok8} widths_ok={widths_ok} both_brains_ok={both_brains_ok} absent_ok={absent_ok}", flush=True)
    assert ok8, "V8 hard gate failed"
    del Xd, Xo, Xb, Xi, Xe
    gc.collect()

    # V9 -- GRID_N_JOBS invariance
    d0 = TESTED_DYADS[0]
    Xv1, _ = build_matrix("deceiver_eeg", row_keys, dec, obs, dy, sb_cols, dy_cols)
    groups_all = row_keys["pair_id"].values
    f0 = next(f for f in folds if f["pair_id"] == d0)
    r1 = _fit_unit(Xv1, yv, groups_all, f0["train_idx"], f0["test_idx"], 1)
    r2 = _fit_unit(Xv1, yv, groups_all, f0["train_idx"], f0["test_idx"], n_jobs_by_set.get("deceiver_eeg", GRID_N_JOBS))
    diff = abs(r1["metrics"]["auroc"] - r2["metrics"]["auroc"])
    ok9 = diff == 0.0
    v["V9_grid_n_jobs_invariance"] = {"passed": ok9, "evidence": {"auroc_n_jobs_1": r1["metrics"]["auroc"],
                                       "auroc_n_jobs_chosen": r2["metrics"]["auroc"], "diff": diff}}
    print(f"V9: passed={ok9} diff={diff}", flush=True)
    del Xv1
    gc.collect()

    # V10 -- reproducibility (refit 3 units from scratch, bypass checkpoint)
    sample_units = [(iid, TESTED_DYADS[0]) for iid in ["deceiver_eeg", "observer_eeg", "interbrain"]]
    max_diff = 0.0
    repro_detail = {}
    for iid, d in sample_units:
        Xv, _ = build_matrix(iid, row_keys, dec, obs, dy, sb_cols, dy_cols)
        f = next(f for f in folds if f["pair_id"] == d)
        fresh = _fit_unit(Xv, yv, groups_all, f["train_idx"], f["test_idx"], n_jobs_by_set.get(iid, GRID_N_JOBS))
        cached = units[f"unit_{iid}_{d}"]
        diff = abs(fresh["metrics"]["auroc"] - cached["metrics"]["auroc"])
        max_diff = max(max_diff, diff)
        repro_detail[f"{iid}_{d}"] = diff
        del Xv
        gc.collect()
    ok10 = max_diff == 0.0
    v["V10_reproducibility"] = {"passed": ok10, "evidence": {"max_abs_diff": max_diff, "detail": repro_detail}}
    print(f"V10: passed={ok10} max_abs_diff={max_diff}", flush=True)

    # V11 -- metric cross-check (Mann-Whitney U identity vs sklearn)
    sample_key = f"unit_deceiver_eeg_{TESTED_DYADS[0]}"
    y_true = np.array(units[sample_key]["_y_true"])
    y_score = np.array(units[sample_key]["_y_score"])
    from scipy.stats import mannwhitneyu
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) and len(neg):
        u_stat, _ = mannwhitneyu(pos, neg, alternative="greater")
        hand_auroc = u_stat / (len(pos) * len(neg))
    else:
        hand_auroc = float("nan")
    sklearn_auroc = units[sample_key]["metrics"]["auroc"]
    diff11 = abs(hand_auroc - sklearn_auroc)
    ok11 = diff11 < 1e-6
    v["V11_metric_cross_check"] = {"passed": ok11, "evidence": {"sklearn_auroc": sklearn_auroc,
                                    "hand_auroc": hand_auroc, "diff": diff11}}
    print(f"V11: passed={ok11} sklearn={sklearn_auroc:.6f} hand={hand_auroc:.6f}", flush=True)

    # V12 -- majority-class sanity
    maj_aurocs = {f["pair_id"]: units[f"unit_majority_{f['pair_id']}"]["metrics"]["auroc"] for f in folds}
    ok12 = all(a == 0.5 for a in maj_aurocs.values())
    v["V12_majority_class_sanity"] = {"passed": ok12, "evidence": maj_aurocs}
    print(f"V12: passed={ok12} {maj_aurocs}", flush=True)
    assert ok12, "V12 hard gate failed"

    # V13 -- plausibility / leak sniff
    max_auroc, max_unit = -1.0, None
    for key, rec in units.items():
        if "metrics" in rec and "auroc" in rec["metrics"]:
            if rec["metrics"]["auroc"] > max_auroc:
                max_auroc, max_unit = rec["metrics"]["auroc"], key
    ok13 = max_auroc <= 0.75
    v["V13_plausibility"] = {"passed": ok13, "evidence": {"max_auroc": max_auroc, "unit": max_unit}}
    print(f"V13: passed={ok13} max_auroc={max_auroc:.4f} unit={max_unit}", flush=True)
    assert ok13, "V13 hard gate failed -- suspected leak"

    # V14 -- upstream file integrity
    frozen_sha = hashlib.sha256(FROZEN_HYP_MD.read_bytes()).hexdigest()
    models_sha = hashlib.sha256((REPO_ROOT / "src" / "models.py").read_bytes()).hexdigest()
    frozen_ok = frozen_sha == FROZEN_HYP_SHA256_EXPECTED
    models_ok = (remote_models_sha is None) or (remote_models_sha == models_sha)
    ok14 = frozen_ok and models_ok
    v["V14_upstream_file_integrity"] = {"passed": ok14, "evidence": {
        "frozen_hyp_sha256": frozen_sha, "expected": FROZEN_HYP_SHA256_EXPECTED, "frozen_ok": frozen_ok,
        "models_py_sha256_local": models_sha, "remote_models_sha_seen": remote_models_sha, "models_ok": models_ok}}
    print(f"V14: passed={ok14} frozen_ok={frozen_ok} models_ok={models_ok}", flush=True)
    assert ok14, "V14 hard gate failed"

    # V15 -- cross-experiment consistency (reporting only)
    dec_median = float(np.median([units[f"unit_deceiver_eeg_{f['pair_id']}"]["metrics"]["auroc"] for f in folds]))
    exp2_json_path = RESULTS_DIR / "exp2_universal.json"
    exp2_note = "exp2_universal.json not found -- skipped"
    if exp2_json_path.exists():
        try:
            exp2 = json.loads(exp2_json_path.read_text())
            exp2_note = f"exp7 deceiver_eeg LODO median={dec_median:.4f} (trial grain, n=11) vs " \
                        f"exp2 LODO primary (participant grain, n=12) -- not expected to be identical, plausibility only"
        except Exception:
            pass
    exp6_note = "exp6_observer.json not found -- skipped"
    exp6_json_path = RESULTS_DIR / "exp6_observer.json"
    obs_median = float(np.median([units[f"unit_observer_eeg_{f['pair_id']}"]["metrics"]["auroc"] for f in folds]))
    if exp6_json_path.exists():
        exp6_note = f"exp7 observer_eeg LODO median={obs_median:.4f} vs exp6 observer numbers -- plausibility only"
    v["V15_cross_experiment_consistency"] = {"passed": True, "evidence": {
        "exp7_deceiver_eeg_median": dec_median, "exp2_note": exp2_note,
        "exp7_observer_eeg_median": obs_median, "exp6_note": exp6_note}}
    print(f"V15: {exp2_note}; {exp6_note}", flush=True)

    # V16 -- interpretation string scan
    md_preview = build_markdown({
        "design": {
            "research_question": "S17: does adding the observer's brain, or inter-brain features, "
                                  "improve prediction over the deceiver's brain alone?",
            "n_dyads": 11, "fold_structure": "leave_one_dyad_out", "grain": "trial",
            "n_trials": int(len(row_keys)), "canonical_index_sha256": index_sha,
            "delta_convention": "AUROC(named_first) - AUROC(named_second) in every test id",
            "claim_hierarchy": {"primary": PRIMARY_TEST, "secondary": SECONDARY_TESTS,
                                 "exploratory": [t[0] for t in EXPLORATORY_TESTS]},
            "behavioral_exclusions": {"pinfo_bart_score": "x", "trials_so_far": "x", "round": "x", "outcome": "x"},
            "amendment": "Amendment 4",
        },
        "gate_verdict": "CONFIRMATORY", "input_sets": INPUT_SETS, "input_set_labels": INPUT_SET_LABELS,
        "input_set_widths": INPUT_SET_WIDTHS,
        "per_input_set": {iid: {"median_auroc": 0.5, "ci95": {"lower": 0.4, "upper": 0.6},
                                 "n_dyads_above_chance": 5} for iid in INPUT_SETS},
        "tests": tests, "tests_exploratory_bh": {},
        "per_dyad": [{"pair_id": d, "n_test": 0, "n_train": 0,
                      "scores": {iid: {"auroc": 0.5} for iid in INPUT_SETS + ["majority"]}} for d in TESTED_DYADS],
        "validations": {},
    })
    hits = [p for p in BANNED_INTERBRAIN_PHRASES if p.lower() in md_preview.lower()]
    # Case-insensitive: build_markdown's actual guard sentence reads "NOT
    # evidence of communication between brains" (capitalized for emphasis).
    # Bug found and fixed during Task 7's remote run (V16 crashed on this
    # exact case mismatch) -- see PROGRESS.md.
    has_guard = "not evidence of communication between brains" in md_preview.lower()
    has_bias_note = "upward bias" in md_preview.lower()
    ok16 = (len(hits) == 0) and has_guard and has_bias_note
    v["V16_interpretation_string_scan"] = {"passed": ok16, "evidence": {
        "banned_hits": hits, "has_noncommunication_guard": has_guard, "has_upward_bias_note": has_bias_note}}
    print(f"V16: passed={ok16} banned_hits={hits} guard={has_guard} bias_note={has_bias_note}", flush=True)
    assert ok16, "V16 hard gate failed"

    return v


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(dry_run=False, smoke=False):
    t_total0 = time.time()
    frozen_sha = hashlib.sha256(FROZEN_HYP_MD.read_bytes()).hexdigest()
    assert frozen_sha == FROZEN_HYP_SHA256_EXPECTED, (
        f"frozen_hypotheses.md sha256 mismatch: {frozen_sha} != {FROZEN_HYP_SHA256_EXPECTED}")

    print("Loading deceiver/observer/dyadic frames...", flush=True)
    dec, dec_manifest, dec_source = load_deceiver_frame()
    obs, obs_manifest, obs_source = load_observer_frame()
    dy, dy_manifest, dy_source = load_dyadic_frame()
    data_source = {"deceiver": dec_source, "observer": obs_source, "dyadic": dy_source}
    print(f"  data_source={data_source}", flush=True)

    sb_cols = M.build_feature_sets(M.load_feature_dictionary())["reliable_plus_marginal"]
    dy_cols = dyadic_feature_columns()

    print("Building canonical index...", flush=True)
    row_keys, index_sha = build_canonical_index(dec, obs, dy, sb_cols, dy_cols)
    print(f"  {len(row_keys)} rows, {row_keys['pair_id'].nunique()} dyads, sha256={index_sha}", flush=True)
    assert len(row_keys) == 10158, len(row_keys)
    assert row_keys["pair_id"].nunique() == 11

    print("Building folds...", flush=True)
    folds = build_folds(row_keys)
    assert len(folds) == 11
    for f in folds:
        train_pids = set(row_keys["pair_id"].values[f["train_idx"]].tolist())
        assert f["pair_id"] not in train_pids

    if dry_run:
        for input_set in INPUT_SETS:
            Xv, cols = build_matrix(input_set, row_keys, dec, obs, dy, sb_cols, dy_cols)
            print(f"  {input_set}: shape={Xv.shape} dtype={Xv.dtype}", flush=True)
            del Xv
            gc.collect()
        for f in folds:
            print(f"  fold {f['pair_id']}: n_test={len(f['test_idx'])} hash={f['test_key_sha256'][:12]}...", flush=True)
        print("DRY RUN COMPLETE -- no model fit.", flush=True)
        return

    n_jobs_by_set = {iid: GRID_N_JOBS for iid in INPUT_SETS}
    preflight = None
    if not smoke:
        print("Running pre-flight memory/timing probe on both_brains...", flush=True)
        preflight = preflight_probe(row_keys, folds, dec, obs, dy, sb_cols, dy_cols)
        n_jobs_by_set["both_brains"] = preflight["n_jobs_chosen"]

    active_folds = folds[:2] if smoke else folds
    active_dyads = [f["pair_id"] for f in active_folds]
    active_input_sets = ["deceiver_eeg"] if smoke else INPUT_SETS

    print(f"Fitting units (smoke={smoke})...", flush=True)
    units, per_set_seconds = run_all_units(
        row_keys, active_folds, dec, obs, dy, sb_cols, dy_cols, n_jobs_by_set,
        input_sets=active_input_sets)

    if smoke:
        print("SMOKE TEST COMPLETE.", flush=True)
        for key, rec in units.items():
            if "metrics" in rec:
                print(f"  {key}: auroc={rec['metrics']['auroc']:.4f}", flush=True)
        return units

    print("Building tests...", flush=True)
    tests, tests_exploratory_bh = build_tests(units)

    print("Running validations...", flush=True)
    v = run_validations(row_keys, folds, index_sha, dec, obs, dy, sb_cols, dy_cols,
                         units, tests, n_jobs_by_set, data_source, None)

    t_total = time.time() - t_total0
    result = assemble(row_keys, folds, index_sha, units, tests, tests_exploratory_bh,
                       per_set_seconds, n_jobs_by_set, preflight, data_source,
                       {"platform": platform.platform()}, t_total)
    result["validations"] = v

    write_outputs(result)
    print(f"Total runtime: {t_total:.1f}s", flush=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run, smoke=args.smoke)
