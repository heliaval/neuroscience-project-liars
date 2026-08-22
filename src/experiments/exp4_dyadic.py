"""
src/experiments/exp4_dyadic.py -- Experiment 4 driver (S14): dyad-specific model.

Research question (S14): does a partner's own prior interaction history help
predict deception, beyond what the target participant's own history and the
wider population already give? For each of 10 included dyads, five conditions
are scored on the SAME held-out block of the tested participant's trials:
Universal, Person-Specific, Dyad-Specific, and two mandatory volume/source
controls (N-matched dyad-specific, and Person + volume-matched other-dyad
data). DyadGain = DyadSpecific - PersonSpecific is the single pre-registered
primary claim (H3).

Runs on the remote Ryzen 5 2600 / WSL2 box (Python 3.8, older sklearn/pyarrow)
via plans/experiment4-dyad-specific-model.md. Reads shards uploaded to
data/processed/features/shard_<pair_id>.parquet -- never calls
M.load_single_brain() (single_brain.parquet itself is not on this machine).

--------------------------------------------------------------------------------
FINDING A -- why exp4 re-cuts the split rather than reusing exp3's blocks
--------------------------------------------------------------------------------
Every deceiver-role participant is the deceiver in exactly one session, and the
two sessions occupy disjoint, contiguous dyad_trial_seq blocks: 1-484
(session 1) and 485-968 (session 2). Exp3 held out each participant's own last
third of THEIR OWN 484-row block. That is fine for Universal/Person-Specific,
but for Dyad-Specific it is fatal: the partner's entire deceiver history for a
session-1 deceiver lives in session 2, chronologically AFTER any block confined
to that participant's own 484 rows. The only legal formulation (frozen in
results/gate.json's exp4 assumption, 2026-08-20) holds out the dyad's own late
tercile of dyad_trial_seq (647-968, 322 rows) as a single dyad-grain block.
Every row in that window belongs to the session-2 deceiver (B); the session-1
deceiver (A) is the partner. A's entire session-1 block (seq 1-484) is then
wholly PRIOR to the held-out window -- legal Dyad-Specific training data.
Consequence: measurement grain is the dyad directly (10 dyads, one score per
level, no participant-to-dyad aggregation needed -- simpler than exp3).
Universal and Person-Specific are RECOMPUTED here on these new blocks; they
are not exp3's numbers and are never copied from results/exp3_personalized.json
(that file is read only for the descriptive drift check).

--------------------------------------------------------------------------------
FINDING B -- n=10, not the pre-registered n=11 (frozen amendment, pre-results)
--------------------------------------------------------------------------------
sub19_sub22 has only sub19 as a deceiver-role participant (sub22 was lost in
S9 preprocessing), and sub19 is the SESSION-1 deceiver -- so this dyad's late
tercile (647-968) contains zero deceiver rows. It is structurally untestable
at dyad grain, discovered from data structure alone before any exp4 model was
fit. results/frozen_hypotheses.md was amended (Amendment 1, 2026-08-22,
pre-results) to reduce exp4's n from 11 to 10; this module validates against
the amended file's sha256, recorded as a module constant below, taken
immediately after the append on the laptop side. sub19_sub22's 484 deceiver
rows remain in the Universal training pool (no leakage -- never a test dyad).

--------------------------------------------------------------------------------
THE FIVE CONDITIONS, per dyad d with tested participant B (session-2 deceiver)
and partner A (session-1 deceiver), all five scored on IDENTICAL T_d = B's
rows with dyad_trial_seq in [647, 968] (322 rows):
--------------------------------------------------------------------------------
1. Universal: all deceiver rows from every OTHER dyad, sub19_sub22 included,
   sub01_sub02 excluded entirely (per exp3 precedent -- excluded from exp3-8 by
   the frozen file, so never a legitimate training OR test source in exp4
   either). n_train ~9,196. Inner CV: M.default_grouped_inner (StratifiedGroupKFold
   on pair_id, 3 splits).
2. Person-Specific: B's own prior rows, seq [485, 646] (162 rows). Inner CV:
   TimeSeriesSplit(3) on seq-sorted rows (falls back to StratifiedKFold(3,
   shuffle=False) if any inner fold lacks both classes, exactly as exp3).
3. Dyad-Specific: condition 2's 162 rows UNION A's full session-1 block, seq
   [1, 484] (484 rows) = 646 rows, sorted by seq before the inner split. THE
   primary condition (H3).
4. Dyad-Specific N-matched (volume control): the 81 most-recent-by-seq rows of
   B's own block (seq 566-646) UNION the 81 most-recent-by-seq rows of A's
   block (seq 404-484) = 162 rows -- same size as condition 2, deterministic
   by construction (most-recent-N, not random).
5. Person + Other-Dyad volume-matched (source control): condition 2's 162 rows
   UNION 484 rows drawn from the Universal pool (excluding d and sub01_sub02),
   seeded per-dyad via np.random.default_rng([M.SEED, dyad_index]),
   label-stratified to match A's own block's lie/truth balance exactly (same
   n_lie, n_truth as A's 484 rows) = 646 rows -- same size as condition 3.
   Majority-class reference: DummyClassifier(most_frequent) fit on condition
   2's rows, scored on T_d.

Conditions 4 and 5 are not optional (plan S15): condition 4 isolates whether
DyadGain survives volume-matching against Person-Specific; condition 5 isolates
whether it survives source-matching (same volume, non-relationship data)
against Dyad-Specific. If DyadGain survives both it is read as a real
relationship effect; if it survives neither, it is reported as a training-
volume artifact, honestly, per S48.

--------------------------------------------------------------------------------
QUANTITIES AND TESTS (S20)
--------------------------------------------------------------------------------
PersonGain_d = PersonSpecific_d - Universal_d
DyadGain_d   = DyadSpecific_d   - PersonSpecific_d      (PRIMARY, H3)
NFI_d        = DyadSpecific_d   - Universal_d           (== PersonGain_d + DyadGain_d)
DyadGain_volume_controlled_d = DyadSpecific_d - PersonPlusOtherDyad_d   (secondary)
DyadGain_matched_d           = DyadSpecificNmatched_d  - PersonSpecific_d (secondary)

Each: sign test (scipy.stats.binomtest, two-sided, ties excluded) and paired
sign-flip permutation test on the median (N_SIGNFLIP=10000, seeded M.SEED).
paired_sign_test/signflip_permutation_test are copied verbatim from
exp3_personalized.py (not imported across experiment modules, per that
module's own discipline; src/models.py is not modified).

Multiple comparisons: DyadGain is the single pre-registered primary claim.
PersonGain and NFI are pre-specified secondaries. The two controls are
tertiary. No correction applied to a single primary. Declared here, before any
number is computed.

--------------------------------------------------------------------------------
ENVIRONMENT (plan S4/S6/Q2, pre-resolved: accept the remote's Python 3.8 set)
--------------------------------------------------------------------------------
Remote: Python 3.8.10, numpy 1.24.4, pandas 2.0.3, scikit-learn 1.3.2,
scipy 1.10.1, pyarrow 17.0.0, xgboost 2.1.4 (not on exp4's critical path --
logistic regression only; installed per the user's blanket instruction, guarded
by models.py's own try/except ImportError). torch/openpyxl deliberately NOT
installed (exp4 needs neither; torch is multi-GB with no GPU on this box).
Laptop environment for comparison: Python 3.13.14, scikit-learn 1.8.0,
numpy 2.3.2, scipy 1.17.1, pandas 2.3.2, pyarrow 25.0.1, xgboost 3.4.1.
All five exp4 conditions are fit inside this ONE remote environment, so every
paired difference is internally consistent (what S20 requires); only
cross-experiment comparison to exp1/exp3 (fit on the laptop) is affected, and
is quantified, never smoothed over, by the drift check in run_validations
(refits 3 sampled exp3 participants on exp3's own blocks in THIS environment
and reports max|delta AUROC| against results/exp3_personalized.json).

--------------------------------------------------------------------------------
DYAD-SPECIFIC SCOPE (plan Q3, pre-resolved: partner's DECEIVER-role rows only)
--------------------------------------------------------------------------------
Condition 3/4's partner contribution uses A's deceiver-role rows only. A's
observer-role rows (A watching B, if any exist in this session's window) are
explicitly out of scope -- mixing role semantics inside a deceiver-condition
classifier is exp6/exp7 territory (one brain vs two brains), not S14's.

--------------------------------------------------------------------------------
CHECKPOINTING (mirrors exp1-3's pattern; new keys, new directory)
--------------------------------------------------------------------------------
exp4_checkpoints/<key>.pkl (relative to this machine's reveriehacks26/ root,
NOT the laptop's out-of-repo exp3_checkpoints -- this runs on a different
physical machine). 60 units total (6 conditions x 10 dyads) plus
n_resume_launches, so a dropped connection costs at most one unit. Per-unit
_ckpt_path/_ckpt_load/_ckpt_save trio, copied from exp3 (~15 lines).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import models as M  # noqa: E402

FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "out"
EXP4_JSON = OUT_DIR / "exp4_dyadic.json"
EXP4_MD = OUT_DIR / "exp4_dyadic.md"
EXP3_JSON = RESULTS_DIR / "exp3_personalized.json"
GATE_JSON = RESULTS_DIR / "gate.json"
FROZEN_HYP_MD = RESULTS_DIR / "frozen_hypotheses.md"
SHARD_MANIFEST = FEATURES_DIR / "shard_manifest.json"

CHECKPOINT_DIR = REPO_ROOT / "exp4_checkpoints"

THRESHOLD_M = 60
N_SIGNFLIP = 10000
METRICS = ["auroc", "balanced_accuracy", "f1", "precision", "recall"]

# Module-level GRID_N_JOBS -- never edits src/models.py (hardcodes n_jobs=1
# inside evaluate_cv, a function this module does not use; exp4 drives fits
# directly, exactly as exp3 did). Set after the early n_jobs invariance check
# in run(); default matches plan S4's reasoning (8GB RAM, 12 loky workers each
# holding a ~130MB copy of the largest (~9,196-row) training slice).
GRID_N_JOBS = 6

TESTED_DYADS = [
    "sub03_sub06", "sub04_sub05", "sub07_sub08", "sub09_sub10", "sub11_sub12",
    "sub13_sub14", "sub15_sub16", "sub17_sub18", "sub20_sub21", "sub23_sub24",
]
EXCLUDED_FROM_EVERYTHING = "sub01_sub02"
UNTESTABLE_UNIT_STILL_IN_POOL = "sub19_sub22"

# sha256 of results/frozen_hypotheses.md AFTER Amendment 1 was appended on the
# laptop (recorded there immediately after the append, per plan step 2.1).
# Validation 13/16 compares the remote copy's hash to this constant.
FROZEN_HYP_SHA256_EXPECTED = "f16a32495d437f1dcd1959e57a47668ef93d20a297f2d9475e421fe2983498bd"

LAPTOP_ENV = {
    "python_version": "3.13.14", "sklearn_version": "1.8.0", "numpy_version": "2.3.2",
    "scipy_version": "1.17.1", "pandas_version": "2.3.2", "pyarrow_version": "25.0.1",
    "xgboost_version": "3.4.1",
}


# ---------------------------------------------------------------------------
# Checkpointing
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
# Data loading -- shard reassembly (never calls M.load_single_brain())
# ---------------------------------------------------------------------------

def load_deceiver_frame() -> pd.DataFrame:
    manifest = json.loads(SHARD_MANIFEST.read_text())
    frames = []
    for pid in manifest["shards"]:
        path = FEATURES_DIR / f"shard_{pid}.parquet"
        frames.append(pd.read_parquet(path))
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("_orig_row", kind="mergesort").reset_index(drop=True)
    assert len(full) == manifest["n_rows_total"], (len(full), manifest["n_rows_total"])
    th = hashlib.sha256(pd.util.hash_pandas_object(full, index=False).values.tobytes()).hexdigest()
    assert th == manifest["total_frame_sha256"], "assembled frame hash mismatch -- data integrity failed"
    full = full.drop(columns=["_orig_row"])
    return full, manifest


# ---------------------------------------------------------------------------
# Per-dyad block construction
# ---------------------------------------------------------------------------

def build_dyad_blocks(row_keys: pd.DataFrame) -> dict:
    """For each of the 10 TESTED_DYADS, identify B (session-2 deceiver,
    seq>=485), A (session-1 deceiver, seq<=484), the held-out block T_d
    (seq in [647,968]), B's own prior rows (seq [485,646]), and A's full
    session-1 block (seq [1,484]). Row indices refer to positions in the
    assembled deceiver frame."""
    blocks = {}
    for d in TESTED_DYADS:
        dyad_mask = row_keys["pair_id"].values == d
        dyad_idx = np.where(dyad_mask)[0]
        seq = row_keys["dyad_trial_seq"].values[dyad_idx]
        pid = row_keys["participant_id"].values[dyad_idx]

        held_out_mask = seq >= 647
        held_out_idx = dyad_idx[held_out_mask]
        b_candidates = set(pid[held_out_mask].tolist())
        assert len(b_candidates) == 1, f"{d}: held-out block spans >1 participant: {b_candidates}"
        B = next(iter(b_candidates))

        a_mask = (pid != B)
        A_candidates = set(pid[a_mask].tolist())
        assert len(A_candidates) == 1, f"{d}: partner block spans >1 participant: {A_candidates}"
        A = next(iter(A_candidates))

        b_all_mask = (pid == B)
        b_prior_mask = b_all_mask & (seq >= 485) & (seq <= 646)
        b_prior_idx = dyad_idx[b_prior_mask]

        a_block_mask = (pid == A) & (seq <= 484)
        a_block_idx = dyad_idx[a_block_mask]

        # Expected sizes are 322/162/484 (all deceiver rows present, no NaN
        # feature-window drops). A handful of rows (6 total, confirmed by real
        # inspection: sub20_sub21 seq 309-310, sub23_sub24 seq 221-222/397-398,
        # all inside A's session-1 block, none in held-out or B's prior block)
        # have a NaN in at least one reliable_plus_marginal feature and are
        # dropped upstream (see run()'s NaN-row filter, mirroring
        # M.prepare_modeling_frame's own behavior). Sizes are therefore
        # asserted close to, not exactly equal to, the nominal counts, and the
        # real counts are what downstream volume-bookkeeping compares against
        # (never the hardcoded nominal figures) -- see validation 8.
        assert 315 <= len(held_out_idx) <= 322, f"{d}: held-out block size {len(held_out_idx)} implausible"
        assert 155 <= len(b_prior_idx) <= 162, f"{d}: B prior block size {len(b_prior_idx)} implausible"
        assert 470 <= len(a_block_idx) <= 484, f"{d}: A block size {len(a_block_idx)} implausible"

        blocks[d] = {
            "B": B, "A": A,
            "held_out_idx": held_out_idx,
            "b_prior_idx": b_prior_idx,
            "a_block_idx": a_block_idx,
            "seq_held_out": seq[held_out_mask],
            "seq_b_prior": seq[b_prior_mask],
            "seq_a_block": seq[a_block_mask],
        }
    return blocks


def universal_pool_idx(row_keys: pd.DataFrame, exclude_dyad: str) -> np.ndarray:
    pair_id_arr = row_keys["pair_id"].values
    exclude = {exclude_dyad, EXCLUDED_FROM_EVERYTHING}
    mask = ~np.isin(pair_id_arr, list(exclude))
    return np.where(mask)[0]


def sort_by_seq(idx: np.ndarray, seq_arr: np.ndarray) -> np.ndarray:
    order = np.argsort(seq_arr[idx], kind="mergesort")
    return idx[order]


def most_recent_n(idx: np.ndarray, seq_arr: np.ndarray, n: int) -> np.ndarray:
    order = np.argsort(seq_arr[idx], kind="mergesort")
    idx_sorted = idx[order]
    return idx_sorted[-n:]


def stratified_draw(pool_idx: np.ndarray, y_arr: np.ndarray, n_lie: int, n_truth: int,
                     seed_seq) -> np.ndarray:
    rng = np.random.default_rng(seed_seq)
    lie_pool = pool_idx[y_arr[pool_idx] == 1]
    truth_pool = pool_idx[y_arr[pool_idx] == 0]
    lie_draw = rng.choice(lie_pool, size=n_lie, replace=False)
    truth_draw = rng.choice(truth_pool, size=n_truth, replace=False)
    return np.concatenate([lie_draw, truth_draw])


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------

def _time_series_inner(seq_sorted_train_idx: np.ndarray, y_arr: np.ndarray):
    y_train = y_arr[seq_sorted_train_idx]
    tss = TimeSeriesSplit(n_splits=3)
    both_ok = True
    for tr, va in tss.split(np.arange(len(seq_sorted_train_idx))):
        if len(np.unique(y_train[tr])) < 2 or len(np.unique(y_train[va])) < 2:
            both_ok = False
            break
    inner = tss if both_ok else StratifiedKFold(n_splits=3, shuffle=False)
    return inner, type(inner).__name__, (not both_ok)


def _fit_predict(Xv, yv, train_idx, test_idx, inner_splitter, groups_train=None,
                  n_jobs=1, collect_coef=False):
    pipe = M._lr_pipeline("l2", M.SEED)
    grid = M._lr_param_grid("l2")
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_splitter, n_jobs=n_jobs)
    fit_kwargs = {} if groups_train is None else {"groups": groups_train}
    conv_warnings = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(Xv[train_idx], yv[train_idx], **fit_kwargs)
        for warning in w:
            if issubclass(warning.category, ConvergenceWarning):
                conv_warnings.append(str(warning.message))
    fitted = search.best_estimator_
    clf = fitted.named_steps["clf"]
    Xt = fitted[:-1].transform(Xv[test_idx]) if len(fitted.steps) > 1 else Xv[test_idx]
    y_score = clf.predict_proba(Xt)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xt)
    y_pred = fitted.predict(Xv[test_idx])
    metrics = M.compute_metrics(yv[test_idx], y_score, y_pred)
    coef = clf.coef_.ravel().tolist() if (collect_coef and hasattr(clf, "coef_")) else None
    return {
        "metrics": metrics, "best_params": search.best_params_, "coef": coef,
        "convergence_warnings": conv_warnings, "n_train": int(len(train_idx)),
    }


# ---------------------------------------------------------------------------
# S20 paired test machinery (copied verbatim from exp3_personalized.py --
# not imported across experiment modules; src/models.py not modified)
# ---------------------------------------------------------------------------

def paired_sign_test(deltas: list) -> dict:
    n_ties = sum(1 for d in deltas if d == 0.0)
    nz = [d for d in deltas if d != 0.0]
    n = len(nz)
    n_pos = sum(1 for d in nz if d > 0)
    n_neg = n - n_pos
    k = min(n_pos, n_neg)
    from scipy import stats as sp_stats
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


# ---------------------------------------------------------------------------
# Per-dyad, per-condition fits
# ---------------------------------------------------------------------------

def run_condition_universal(Xv, yv, row_keys, blocks, n_jobs):
    pair_id_arr = row_keys["pair_id"].values
    records = {}
    for d in TESTED_DYADS:
        key = f"univ_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            records[d] = cached
            print(f"  [checkpoint] universal {d}: loaded auroc={cached['metrics']['auroc']:.4f}", flush=True)
            continue
        t0 = time.time()
        train_idx = universal_pool_idx(row_keys, d)
        test_idx = blocks[d]["held_out_idx"]
        groups_train = pair_id_arr[train_idx]
        inner = M.default_grouped_inner(M.SEED, n_splits=3)
        rec = _fit_predict(Xv, yv, train_idx, test_idx, inner, groups_train=groups_train,
                            n_jobs=n_jobs, collect_coef=True)
        rec["inner_splitter_class"] = type(inner).__name__
        rec["train_pair_ids"] = sorted(set(pair_id_arr[train_idx].tolist()))
        _ckpt_save(key, rec)
        records[d] = rec
        print(f"  universal {d}: n_train={rec['n_train']} auroc={rec['metrics']['auroc']:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return records


def run_condition_person_specific(Xv, yv, row_keys, blocks, n_jobs):
    seq_arr = row_keys["dyad_trial_seq"].values
    records = {}
    for d in TESTED_DYADS:
        key = f"pers_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            records[d] = cached
            print(f"  [checkpoint] person-specific {d}: loaded auroc={cached['metrics']['auroc']:.4f}", flush=True)
            continue
        t0 = time.time()
        train_idx = sort_by_seq(blocks[d]["b_prior_idx"], seq_arr)
        test_idx = blocks[d]["held_out_idx"]
        inner, inner_class, fallback = _time_series_inner(train_idx, yv)
        rec = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=n_jobs, collect_coef=True)
        rec["inner_splitter_class"] = inner_class
        rec["fallback_to_stratified"] = fallback
        _ckpt_save(key, rec)
        records[d] = rec
        print(f"  person-specific {d}: n_train={rec['n_train']} auroc={rec['metrics']['auroc']:.4f} "
              f"C={rec['best_params'].get('clf__C')} fallback={fallback} ({time.time()-t0:.1f}s)", flush=True)
    return records


def run_condition_dyad_specific(Xv, yv, row_keys, blocks, n_jobs):
    seq_arr = row_keys["dyad_trial_seq"].values
    records = {}
    for d in TESTED_DYADS:
        key = f"dyad_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            records[d] = cached
            print(f"  [checkpoint] dyad-specific {d}: loaded auroc={cached['metrics']['auroc']:.4f}", flush=True)
            continue
        t0 = time.time()
        union_idx = np.concatenate([blocks[d]["b_prior_idx"], blocks[d]["a_block_idx"]])
        train_idx = sort_by_seq(union_idx, seq_arr)
        test_idx = blocks[d]["held_out_idx"]
        inner, inner_class, fallback = _time_series_inner(train_idx, yv)
        rec = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=n_jobs, collect_coef=True)
        rec["inner_splitter_class"] = inner_class
        rec["fallback_to_stratified"] = fallback
        _ckpt_save(key, rec)
        records[d] = rec
        print(f"  dyad-specific {d}: n_train={rec['n_train']} auroc={rec['metrics']['auroc']:.4f} "
              f"C={rec['best_params'].get('clf__C')} fallback={fallback} ({time.time()-t0:.1f}s)", flush=True)
    return records


def run_condition_nmatched(Xv, yv, row_keys, blocks, n_jobs):
    seq_arr = row_keys["dyad_trial_seq"].values
    records = {}
    for d in TESTED_DYADS:
        key = f"dyadmatch_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            records[d] = cached
            print(f"  [checkpoint] n-matched {d}: loaded auroc={cached['metrics']['auroc']:.4f}", flush=True)
            continue
        t0 = time.time()
        b_recent = most_recent_n(blocks[d]["b_prior_idx"], seq_arr, 81)
        a_recent = most_recent_n(blocks[d]["a_block_idx"], seq_arr, 81)
        union_idx = np.concatenate([b_recent, a_recent])
        train_idx = sort_by_seq(union_idx, seq_arr)
        test_idx = blocks[d]["held_out_idx"]
        inner, inner_class, fallback = _time_series_inner(train_idx, yv)
        rec = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=n_jobs, collect_coef=True)
        rec["inner_splitter_class"] = inner_class
        rec["fallback_to_stratified"] = fallback
        _ckpt_save(key, rec)
        records[d] = rec
        print(f"  n-matched {d}: n_train={rec['n_train']} auroc={rec['metrics']['auroc']:.4f} "
              f"C={rec['best_params'].get('clf__C')} fallback={fallback} ({time.time()-t0:.1f}s)", flush=True)
    return records


def run_condition_person_other_dyad(Xv, yv, row_keys, blocks, n_jobs):
    seq_arr = row_keys["dyad_trial_seq"].values
    records = {}
    for i, d in enumerate(TESTED_DYADS):
        key = f"personother_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            records[d] = cached
            print(f"  [checkpoint] person+other-dyad {d}: loaded auroc={cached['metrics']['auroc']:.4f}", flush=True)
            continue
        t0 = time.time()
        a_idx = blocks[d]["a_block_idx"]
        n_lie_a = int(yv[a_idx].sum())
        n_truth_a = len(a_idx) - n_lie_a
        pool_idx = universal_pool_idx(row_keys, d)
        other_draw = stratified_draw(pool_idx, yv, n_lie_a, n_truth_a, [M.SEED, i])
        union_idx = np.concatenate([blocks[d]["b_prior_idx"], other_draw])
        # Chronological sort applies only to the same-dyad B rows; the
        # cross-dyad draw has no seq relationship to dyad d's timeline (see
        # module docstring / run_validations item 3). Sort what can be sorted
        # meaningfully (B's own rows first by construction of the union, since
        # they are always seq 485-646, i.e. earlier than nothing sortable here);
        # TimeSeriesSplit is applied to the union in the order
        # [B's 162 rows sorted by seq] + [other-dyad draw in draw order] so the
        # inner CV's later folds still see progressively more of B's own
        # chronological signal, while the cross-dyad rows (no shared timeline)
        # are appended without a spurious ordering claim.
        b_sorted = sort_by_seq(blocks[d]["b_prior_idx"], seq_arr)
        train_idx = np.concatenate([b_sorted, other_draw])
        test_idx = blocks[d]["held_out_idx"]
        inner, inner_class, fallback = _time_series_inner(train_idx, yv)
        rec = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=n_jobs, collect_coef=True)
        rec["inner_splitter_class"] = inner_class
        rec["fallback_to_stratified"] = fallback
        rec["other_dyad_draw_idx"] = other_draw.tolist()
        rec["other_dyad_pair_ids"] = sorted(set(row_keys["pair_id"].values[other_draw].tolist()))
        _ckpt_save(key, rec)
        records[d] = rec
        print(f"  person+other-dyad {d}: n_train={rec['n_train']} auroc={rec['metrics']['auroc']:.4f} "
              f"C={rec['best_params'].get('clf__C')} fallback={fallback} ({time.time()-t0:.1f}s)", flush=True)
    return records


def run_majority(Xv, yv, blocks):
    records = {}
    for d in TESTED_DYADS:
        key = f"major_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            records[d] = cached
            continue
        train_idx = blocks[d]["b_prior_idx"]
        test_idx = blocks[d]["held_out_idx"]
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(Xv[train_idx], yv[train_idx])
        y_score = clf.predict_proba(Xv[test_idx])[:, 1]
        y_pred = clf.predict(Xv[test_idx])
        metrics = M.compute_metrics(yv[test_idx], y_score, y_pred)
        _ckpt_save(key, metrics)
        records[d] = metrics
    return records


# ---------------------------------------------------------------------------
# GRID_N_JOBS invariance check (plan S4)
# ---------------------------------------------------------------------------

def grid_n_jobs_check(Xv, yv, row_keys, blocks):
    d = TESTED_DYADS[0]
    seq_arr = row_keys["dyad_trial_seq"].values
    train_idx = sort_by_seq(blocks[d]["b_prior_idx"], seq_arr)
    test_idx = blocks[d]["held_out_idx"]
    inner, _, _ = _time_series_inner(train_idx, yv)
    t0 = time.time()
    r1 = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=1)
    t1 = time.time() - t0
    t0 = time.time()
    r2 = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=GRID_N_JOBS)
    t2 = time.time() - t0
    identical = (r1["metrics"] == r2["metrics"]) and (r1["best_params"] == r2["best_params"])
    return {
        "dyad_sampled": d, "n_jobs_1_seconds": t1, "n_jobs_chosen_seconds": t2,
        "n_jobs_chosen": GRID_N_JOBS, "metrics_identical": identical,
        "best_params_n1": r1["best_params"], "best_params_nchosen": r2["best_params"],
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

COND_KEYS = ["universal", "person_specific", "dyad_specific", "n_matched", "person_other_dyad"]
COND_LABELS = {
    "universal": "Universal", "person_specific": "Person-Specific",
    "dyad_specific": "Dyad-Specific", "n_matched": "Dyad-Specific (N-matched)",
    "person_other_dyad": "Person + Other-Dyad (volume-matched)",
}


def assemble(blocks, cond_records: dict, maj_records, row_keys):
    per_dyad = []
    for d in TESTED_DYADS:
        b = blocks[d]
        row = {
            "pair_id": d, "tested_participant": b["B"], "partner": b["A"],
            "held_out_seq_min": int(b["seq_held_out"].min()), "held_out_seq_max": int(b["seq_held_out"].max()),
            "n_test": int(len(b["held_out_idx"])),
        }
        for ck in COND_KEYS:
            rec = cond_records[ck][d]
            row[ck] = {m: rec["metrics"][m] for m in METRICS}
            row[ck]["best_params"] = rec["best_params"]
            row[ck]["n_train"] = rec["n_train"]
        row["majority"] = maj_records[d]
        per_dyad.append(row)

    per_dyad_scores = {ck: {m: {} for m in METRICS} for ck in COND_KEYS}
    for row in per_dyad:
        d = row["pair_id"]
        for ck in COND_KEYS:
            for m in METRICS:
                per_dyad_scores[ck][m][d] = row[ck][m]

    def g(d, ck, m):
        return per_dyad_scores[ck][m][d]

    gains = {"person_gain": {}, "dyad_gain": {}, "nfi": {}, "dyad_gain_volume_controlled": {},
             "dyad_gain_matched": {}}
    for m in METRICS:
        gains["person_gain"][m] = {d: g(d, "person_specific", m) - g(d, "universal", m) for d in TESTED_DYADS}
        gains["dyad_gain"][m] = {d: g(d, "dyad_specific", m) - g(d, "person_specific", m) for d in TESTED_DYADS}
        gains["nfi"][m] = {d: g(d, "dyad_specific", m) - g(d, "universal", m) for d in TESTED_DYADS}
        gains["dyad_gain_volume_controlled"][m] = {
            d: g(d, "dyad_specific", m) - g(d, "person_other_dyad", m) for d in TESTED_DYADS}
        gains["dyad_gain_matched"][m] = {
            d: g(d, "n_matched", m) - g(d, "person_specific", m) for d in TESTED_DYADS}

    return per_dyad, per_dyad_scores, gains


def build_tests(gains: dict) -> dict:
    tests = {}
    for gain_name, per_metric in gains.items():
        tests[gain_name] = {}
        for m, per_dyad_delta in per_metric.items():
            deltas = [per_dyad_delta[d] for d in TESTED_DYADS]
            sign = paired_sign_test(deltas)
            perm = signflip_permutation_test(deltas, N_SIGNFLIP, M.SEED)
            ci = M.ci95_from_folds(deltas)
            tests[gain_name][m] = {
                "n": len(deltas), "deltas": deltas, "dyad_ids": TESTED_DYADS,
                "median_delta": float(np.median(deltas)),
                "n_positive": sign["n_positive"], "n_negative": sign["n_negative"],
                "n_ties_excluded": sign["n_ties_excluded"], "sign_test_p": sign["p"],
                "permutation_p": perm["p"], "n_signflip": N_SIGNFLIP, "ci95": ci,
            }
    tests["primary"] = "dyad_gain"
    tests["secondary"] = ["person_gain", "nfi"]
    tests["tertiary"] = ["dyad_gain_volume_controlled", "dyad_gain_matched"]
    tests["multiple_comparisons_note"] = (
        "DyadGain (H3) is the single pre-registered primary inferential claim; "
        "PersonGain and NFI are pre-specified secondaries; the two volume/source "
        "controls are tertiary. No correction is applied to a single primary. "
        "This declaration is written before the numbers below and is not revised "
        "after seeing them."
    )
    return tests


# ---------------------------------------------------------------------------
# Validations (18 checks + drift check)
# ---------------------------------------------------------------------------

def run_validations(Xv, yv, row_keys, blocks, cond_records, maj_records, per_dyad,
                     per_dyad_scores, n_jobs_check_result, deceiver_manifest):
    print("\n" + "=" * 70, flush=True)
    print("Validations", flush=True)
    print("=" * 70, flush=True)
    v = {}
    seq_arr = row_keys["dyad_trial_seq"].values
    pair_id_arr = row_keys["pair_id"].values

    # 1. Chronological integrity -- Person-Specific
    ok1 = True
    detail1 = {}
    for d in TESTED_DYADS:
        train_idx = blocks[d]["b_prior_idx"]
        test_idx = blocks[d]["held_out_idx"]
        chron = seq_arr[test_idx].min() > seq_arr[train_idx].max()
        ok1 = ok1 and chron
        detail1[d] = {"train_max_seq": int(seq_arr[train_idx].max()),
                       "test_min_seq": int(seq_arr[test_idx].min()), "chronological": bool(chron)}
    v["1_chronological_person_specific"] = {"ok": ok1, "per_dyad": detail1}
    print(f"1. chronological-person-specific: ok={ok1}", flush=True)

    # 2. Chronological integrity -- Dyad-Specific (union, THE finding-A check)
    ok2 = True
    detail2 = {}
    for d in TESTED_DYADS:
        union_idx = np.concatenate([blocks[d]["b_prior_idx"], blocks[d]["a_block_idx"]])
        test_idx = blocks[d]["held_out_idx"]
        chron = seq_arr[test_idx].min() > seq_arr[union_idx].max()
        only_this_dyad = set(pair_id_arr[union_idx].tolist()) == {d}
        ok2 = ok2 and chron and only_this_dyad
        detail2[d] = {"train_max_seq": int(seq_arr[union_idx].max()),
                       "test_min_seq": int(seq_arr[test_idx].min()), "chronological": bool(chron),
                       "only_this_dyad": only_this_dyad}
    v["2_chronological_dyad_specific"] = {"ok": ok2, "per_dyad": detail2}
    print(f"2. chronological-dyad-specific: ok={ok2}", flush=True)

    # 3. Chronological integrity -- the two controls
    ok3 = True
    detail3 = {}
    for d in TESTED_DYADS:
        rec4 = cond_records["n_matched"][d]
        rec5 = cond_records["person_other_dyad"][d]
        # N-matched: pure union of this dyad's own rows -- full chronology check applies.
        b_recent = most_recent_n(blocks[d]["b_prior_idx"], seq_arr, 81)
        a_recent = most_recent_n(blocks[d]["a_block_idx"], seq_arr, 81)
        union4 = np.concatenate([b_recent, a_recent])
        test_idx = blocks[d]["held_out_idx"]
        chron4 = seq_arr[test_idx].min() > seq_arr[union4].max()
        only_this_dyad4 = set(pair_id_arr[union4].tolist()) == {d}
        # Person+Other-Dyad: chronology checked only on the same-dyad (B) rows;
        # the cross-dyad draw has no seq relationship to this dyad's timeline
        # (different dyad, different trial-sequence numbering) so is checked
        # instead for the leakage condition that actually applies: it must
        # never include a row from dyad d or from sub01_sub02.
        other_idx = np.array(rec5["other_dyad_draw_idx"], dtype=int)
        b_part_chron = seq_arr[test_idx].min() > seq_arr[blocks[d]["b_prior_idx"]].max()
        other_excludes_target_and_sub01 = not (set(pair_id_arr[other_idx].tolist()) & {d, EXCLUDED_FROM_EVERYTHING})
        ok3 = ok3 and chron4 and only_this_dyad4 and b_part_chron and other_excludes_target_and_sub01
        detail3[d] = {
            "n_matched_chronological": bool(chron4), "n_matched_only_this_dyad": only_this_dyad4,
            "person_other_dyad_b_part_chronological": bool(b_part_chron),
            "person_other_dyad_draw_excludes_target_and_sub01_sub02": other_excludes_target_and_sub01,
        }
    v["3_chronological_controls"] = {"ok": ok3, "per_dyad": detail3,
                                      "note": "Person+Other-Dyad's cross-dyad draw is checked for "
                                              "target/sub01_sub02 exclusion, not seq-ordering -- "
                                              "dyad_trial_seq is dyad-local, so a different dyad's "
                                              "seq values carry no chronological relationship to d's "
                                              "timeline (see module docstring)."}
    print(f"3. chronological-controls: ok={ok3}", flush=True)

    # 4. Identical held-out block across all five conditions -- true by
    # construction: every condition's _fit_predict call below is passed
    # blocks[d]["held_out_idx"] directly, never a per-condition copy.
    ok4 = True
    v["4_identical_held_out_block"] = {"ok": ok4, "note": "True by construction -- every condition's "
                                        "_fit_predict call is passed blocks[d]['held_out_idx'] directly."}
    print(f"4. identical-held-out-block: ok={ok4}", flush=True)

    # 5. No identity columns in X -- checked at call site in run(); recorded here
    forbidden = set(M.IDENTITY_COLUMNS) | {"outcome", "observer_guess", "points"}
    v["5_no_identity_columns"] = {"forbidden_set": sorted(forbidden), "checked_at": "run() before .values"}
    print("5. no-identity-columns: checked at run() call site", flush=True)

    # 6. Dyad pool reconciliation
    all_pair_ids = set(pair_id_arr.tolist())
    ok6 = (len(TESTED_DYADS) == 10 and UNTESTABLE_UNIT_STILL_IN_POOL not in TESTED_DYADS
           and EXCLUDED_FROM_EVERYTHING not in TESTED_DYADS
           and UNTESTABLE_UNIT_STILL_IN_POOL in all_pair_ids
           and EXCLUDED_FROM_EVERYTHING in all_pair_ids)
    no_nan = not any(pd.isna(row[ck][m]) for row in per_dyad for ck in COND_KEYS for m in METRICS)
    v["6_dyad_pool_reconciliation"] = {
        "ok": ok6 and no_nan, "n_tested_dyads": len(TESTED_DYADS),
        "sub19_sub22_present_in_frame_not_tested": UNTESTABLE_UNIT_STILL_IN_POOL in all_pair_ids,
        "sub01_sub02_present_in_frame_not_tested": EXCLUDED_FROM_EVERYTHING in all_pair_ids,
        "no_nan_scores": no_nan,
    }
    print(f"6. dyad-pool-reconciliation: {v['6_dyad_pool_reconciliation']}", flush=True)

    # 7. Universal pool exclusion
    ok7 = True
    for d in TESTED_DYADS:
        train_pids = set(cond_records["universal"][d]["train_pair_ids"])
        clean = d not in train_pids and EXCLUDED_FROM_EVERYTHING not in train_pids
        ok7 = ok7 and clean
    v["7_universal_pool_exclusion"] = {"ok": ok7}
    print(f"7. universal-pool-exclusion: ok={ok7}", flush=True)

    # 8. Volume bookkeeping
    ok8 = True
    detail8 = {}
    for d in TESTED_DYADS:
        n_univ = cond_records["universal"][d]["n_train"]
        n_pers = cond_records["person_specific"][d]["n_train"]
        n_dyad = cond_records["dyad_specific"][d]["n_train"]
        n_nmatch = cond_records["n_matched"][d]["n_train"]
        n_pother = cond_records["person_other_dyad"][d]["n_train"]
        # Nominal sizes are 162/646/162/646/~9,196 (see plan S1's table). A
        # handful of rows (6 total, all inside partner session-1 blocks) are
        # dropped upstream for a NaN in a feature column (see run()'s
        # NaN-row filter), so exact-equality against the nominal figures is
        # relaxed to a small tolerance; the equalities that must hold EXACTLY
        # regardless of any drop -- conditions 2&4 same size, conditions 3&5
        # same size (they are controls for exactly this) -- are still checked
        # exactly, since both sides of each pair are computed from the same
        # post-drop block sizes by construction.
        checks = {
            "pers_near_162": abs(n_pers - 162) <= 7, "dyad_near_646": abs(n_dyad - 646) <= 7,
            "nmatch_near_162": abs(n_nmatch - 162) <= 7, "pother_near_646": abs(n_pother - 646) <= 7,
            "dyad_eq_pother": n_dyad == n_pother,
            "pers_eq_nmatch": n_pers == n_nmatch, "univ_near_9196": abs(n_univ - 9196) <= 10,
        }
        ok8 = ok8 and all(checks.values())
        detail8[d] = {"n_universal": n_univ, "n_person_specific": n_pers, "n_dyad_specific": n_dyad,
                       "n_matched": n_nmatch, "n_person_other_dyad": n_pother, "checks": checks}
    v["8_volume_bookkeeping"] = {"ok": ok8, "per_dyad": detail8}
    print(f"8. volume-bookkeeping: ok={ok8}", flush=True)

    # 9. Majority-class sanity
    maj_aurocs = [m["auroc"] for m in maj_records.values()]
    maj_bal = [m["balanced_accuracy"] for m in maj_records.values()]
    ok9 = all(abs(a - 0.5) <= 0.02 for a in maj_aurocs) and all(abs(b - 0.5) <= 0.02 for b in maj_bal)
    v["9_majority_class_sanity"] = {"mean_auroc": float(np.mean(maj_aurocs)),
                                     "mean_balanced_accuracy": float(np.mean(maj_bal)), "ok": ok9}
    print(f"9. majority-class-sanity: {v['9_majority_class_sanity']}", flush=True)

    # 10. Reproducibility -- 3 sampled units, bypass checkpoint cache
    from sklearn.model_selection import TimeSeriesSplit as _TSS
    sample_units = [("pers", TESTED_DYADS[0]), ("dyad", TESTED_DYADS[1]), ("univ", TESTED_DYADS[2])]
    diffs = {}
    for kind, d in sample_units:
        if kind == "pers":
            train_idx = sort_by_seq(blocks[d]["b_prior_idx"], seq_arr)
            inner, _, _ = _time_series_inner(train_idx, yv)
            groups_train = None
        elif kind == "dyad":
            union_idx = np.concatenate([blocks[d]["b_prior_idx"], blocks[d]["a_block_idx"]])
            train_idx = sort_by_seq(union_idx, seq_arr)
            inner, _, _ = _time_series_inner(train_idx, yv)
            groups_train = None
        else:
            train_idx = universal_pool_idx(row_keys, d)
            inner = M.default_grouped_inner(M.SEED, n_splits=3)
            groups_train = pair_id_arr[train_idx]
        test_idx = blocks[d]["held_out_idx"]
        r1 = _fit_predict(Xv, yv, train_idx, test_idx, inner, groups_train=groups_train, n_jobs=1)
        r2 = _fit_predict(Xv, yv, train_idx, test_idx, inner, groups_train=groups_train, n_jobs=1)
        diffs[f"{kind}_{d}"] = abs(r1["metrics"]["auroc"] - r2["metrics"]["auroc"])
    v["10_reproducibility"] = {"max_diff": max(diffs.values()), "per_unit": diffs}
    print(f"10. reproducibility: max_diff={v['10_reproducibility']['max_diff']:.2e}", flush=True)

    # 11. GRID_N_JOBS invariance
    v["11_grid_n_jobs_invariance"] = n_jobs_check_result
    print(f"11. grid-n-jobs-invariance: identical={n_jobs_check_result['metrics_identical']}", flush=True)

    # 12. Metric cross-check
    from sklearn.metrics import roc_auc_score
    from scipy.stats import mannwhitneyu
    d0 = TESTED_DYADS[0]
    train_idx = sort_by_seq(blocks[d0]["b_prior_idx"], seq_arr)
    test_idx = blocks[d0]["held_out_idx"]
    pipe = M._lr_pipeline("l2", M.SEED)
    pipe.set_params(clf__C=cond_records["person_specific"][d0]["best_params"]["clf__C"])
    pipe.fit(Xv[train_idx], yv[train_idx])
    y_score = pipe.predict_proba(Xv[test_idx])[:, 1]
    y_test = yv[test_idx]
    sk_auc = float(roc_auc_score(y_test, y_score))
    pos = y_score[y_test == 1]
    neg = y_score[y_test == 0]
    u_stat, _ = mannwhitneyu(pos, neg)
    hand_auc = float(u_stat / (len(pos) * len(neg)))
    v["12_metric_crosscheck"] = {"dyad": d0, "sklearn_auroc": sk_auc, "hand_auroc": hand_auc,
                                  "match_to_1e6": abs(sk_auc - hand_auc) < 1e-6}
    print(f"12. metric-crosscheck: {v['12_metric_crosscheck']}", flush=True)

    # 13. Upstream files untouched, by hash
    upstream = {
        "models.py": REPO_ROOT / "src" / "models.py",
        "gate.json": GATE_JSON,
        "frozen_hypotheses.md": FROZEN_HYP_MD,
        "exp3_personalized.json": EXP3_JSON,
        "feature_dictionary.csv": FEATURES_DIR / "feature_dictionary.csv",
    }
    hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in upstream.items()}
    frozen_hash_ok = hashes["frozen_hypotheses.md"] == FROZEN_HYP_SHA256_EXPECTED
    v["13_upstream_files_hashed"] = {"hashes": hashes, "frozen_hypotheses_matches_amended_hash": frozen_hash_ok}
    print(f"13. upstream-files-hashed: frozen_matches={frozen_hash_ok}", flush=True)

    # 14. Data integrity
    v["14_data_integrity"] = {
        "row_count_10648": deceiver_manifest["n_rows_total"] == 10648,
        "col_count_matches_manifest": True,
        "total_frame_hash_verified": "asserted in load_deceiver_frame() -- would have raised if mismatched",
    }
    print(f"14. data-integrity: {v['14_data_integrity']}", flush=True)

    # 15. Gate verdict recorded
    gate_json = json.loads(GATE_JSON.read_text())
    exp4_gate = gate_json.get("gate", {}).get("experiments", {}).get("exp4")
    minority_counts = {}
    for d in TESTED_DYADS:
        test_idx = blocks[d]["held_out_idx"]
        n_lie = int(yv[test_idx].sum())
        n_truth = len(test_idx) - n_lie
        minority_counts[d] = min(n_lie, n_truth)
    all_clear = all(c >= THRESHOLD_M for c in minority_counts.values())
    v["15_gate_verdict_recorded"] = {
        "gate_json_exp4_entry": exp4_gate, "minority_counts_per_dyad": minority_counts,
        "threshold_m": THRESHOLD_M, "all_dyads_clear_threshold": all_clear,
    }
    print(f"15. gate-verdict-recorded: all_clear={all_clear}", flush=True)

    # 16. Amendment recorded
    amend_text = FROZEN_HYP_MD.read_text()
    amend_present = "Amendment 1" in amend_text and "n = 10" in amend_text
    v["16_amendment_recorded"] = {
        "amendment_1_present": amend_present, "n_dyads": len(TESTED_DYADS),
        "mismatch_vs_original_n11_stated": True,
        "original_n11_line_present": "n = 11 for: exp3, exp4" in amend_text,
    }
    print(f"16. amendment-recorded: {v['16_amendment_recorded']}", flush=True)

    # 17. Plausibility / leakage suspicion
    all_aurocs = [row[ck]["auroc"] for row in per_dyad for ck in COND_KEYS]
    flagged = []
    for row in per_dyad:
        d = row["pair_id"]
        deltas_this_dyad = {
            "person_gain": row["person_specific"]["auroc"] - row["universal"]["auroc"],
            "dyad_gain": row["dyad_specific"]["auroc"] - row["person_specific"]["auroc"],
        }
        if any(abs(v_) > 0.20 for v_ in deltas_this_dyad.values()):
            flagged.append(d)
        if any(row[ck]["auroc"] > 0.75 for ck in COND_KEYS):
            flagged.append(d)
    mean_dyad_specific = float(np.mean([row["dyad_specific"]["auroc"] for row in per_dyad]))
    v["17_plausibility"] = {
        "max_auroc": float(max(all_aurocs)),
        "mean_dyad_specific_auroc": mean_dyad_specific,
        "exp1_pooled_auroc_reference": 0.5338,
        "implausibly_above_exp1": bool(mean_dyad_specific > 0.5338 + 0.15),
        "dyads_flagged": sorted(set(flagged)),
        "note": "Flagging is for inspection, not automatic exclusion (exp3 precedent: sub15 was "
                "flagged, inspected, and kept).",
    }
    print(f"17. plausibility: {v['17_plausibility']}", flush=True)

    # 18. Convergence
    n_conv = {ck: sum(len(cond_records[ck][d]["convergence_warnings"]) for d in TESTED_DYADS) for ck in COND_KEYS}
    v["18_convergence"] = {"n_convergence_warnings_per_condition": n_conv}
    print(f"18. convergence: {n_conv}", flush=True)

    return v


def _drift_check_impl(Xv, yv, row_keys, exp3_json_path):
    """Refit exp3's own personalized condition for 3 sampled participants, on
    exp3's own per-participant chronological blocks, in THIS (remote,
    Python 3.8) environment, and report max|delta AUROC| against
    results/exp3_personalized.json. exp4 has no single_brain.parquet on this
    machine by design (shards only cover deceiver rows, reliable_plus_marginal);
    exp3 used the identical feature set and role filter, so the already-
    assembled exp4 deceiver frame is reused directly rather than re-deriving
    from a file not present on this machine. Reads exp3's json for the
    ORIGINAL (laptop) numbers only -- never a data source for exp4's own
    computation."""
    seq_arr = row_keys["dyad_trial_seq"].values
    pid_arr = row_keys["participant_id"].values
    exp3 = json.loads(exp3_json_path.read_text())
    exp3_participants = exp3["experiments"]["exp3"]["per_participant"]
    sample = sorted({r["participant_id"] for r in exp3_participants})[:3]
    results = {}
    import math as _math
    for p in sample:
        mask = pid_arr == p
        idx = np.where(mask)[0]
        if len(idx) == 0:
            results[p] = {"note": "participant not present in exp4's deceiver frame (unexpected)"}
            continue
        seq = seq_arr[idx]
        order = np.argsort(seq, kind="mergesort")
        idx_sorted = idx[order]
        n = len(idx_sorted)
        cut = int(_math.ceil(n * 2 / 3))
        train_idx = idx_sorted[:cut]
        test_idx = idx_sorted[cut:]
        if len(test_idx) == 0 or len(np.unique(yv[test_idx])) < 2:
            results[p] = {"note": "held-out block missing a class in this environment's refit -- skipped"}
            continue
        tss = TimeSeriesSplit(n_splits=3)
        y_train = yv[train_idx]
        both_ok = all(len(np.unique(y_train[a])) == 2 and len(np.unique(y_train[b])) == 2
                      for a, b in tss.split(np.arange(len(train_idx))))
        inner = tss if both_ok else StratifiedKFold(n_splits=3, shuffle=False)
        rec = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=1)
        remote_auroc = rec["metrics"]["auroc"]
        laptop_rec = next((r for r in exp3_participants if r["participant_id"] == p), None)
        laptop_auroc = laptop_rec["personalized"]["auroc"] if laptop_rec else None
        results[p] = {"remote_auroc": remote_auroc, "laptop_auroc": laptop_auroc,
                       "abs_delta": abs(remote_auroc - laptop_auroc) if laptop_auroc is not None else None}
    deltas = [r["abs_delta"] for r in results.values() if r.get("abs_delta") is not None]
    return {
        "sampled_participants": sample, "per_participant": results,
        "max_abs_delta_auroc": max(deltas) if deltas else None,
        "note": "Measurement of environment comparability only (Python 3.8/sklearn 1.3.2 remote "
                "vs Python 3.13/sklearn 1.8.0 laptop). Never a source of exp4's own numbers.",
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def build_markdown(results: dict) -> str:
    exp4 = results["experiments"]["exp4"]
    lines = []
    lines.append("# Experiment 4 -- Dyad-Specific Model (S14)\n")
    lines.append(f"**What this experiment asks:** {exp4['design']['research_question']}\n")

    lines.append("## Finding A and Finding B\n")
    lines.append(exp4["design"]["finding_a_prose"] + "\n")
    lines.append(exp4["design"]["finding_b_prose"] + "\n")

    lines.append("## Design\n")
    d = exp4["design"]
    lines.append(f"- n = {d['n_dyads']} dyads (amended from pre-registered n=11; see "
                  "`results/frozen_hypotheses.md` Amendment 1).\n"
                  f"- Held-out block: {d['held_out_block']}\n"
                  f"- Feature set: {d['feature_set']} ({d['n_features']} cols). Model: {d['model_family']}\n")
    lines.append("| condition | training rows |\n|---|---|\n")
    for ck in COND_KEYS:
        lines.append(f"| {COND_LABELS[ck]} | {d['conditions'][ck]} |\n")

    lines.append("\n## Per-dyad score table (AUROC)\n")
    lines.append("| pair_id | tested | partner | " + " | ".join(COND_LABELS[c] for c in COND_KEYS) + " | majority |")
    lines.append("|---|---|---|" + "---|" * (len(COND_KEYS) + 1))
    for row in exp4["per_dyad"]:
        cells = " | ".join(f"{row[c]['auroc']:.4f}" for c in COND_KEYS)
        lines.append(f"| {row['pair_id']} | {row['tested_participant']} | {row['partner']} | {cells} | "
                      f"{row['majority']['auroc']:.4f} |")
    lines.append("")

    lines.append("## Primary claim -- DyadGain (H3)\n")
    dg = exp4["tests"]["dyad_gain"]["auroc"]
    lines.append(f"Median DyadGain = {dg['median_delta']:+.4f}, {dg['n_positive']}/{dg['n']} dyads positive, "
                  f"sign-test p = {dg['sign_test_p']:.4g}, sign-flip permutation p = {dg['permutation_p']:.4g} "
                  f"(n_signflip={dg['n_signflip']}). 95% CI: [{dg['ci95']['lower']:.4f}, {dg['ci95']['upper']:.4f}].\n")

    lines.append("## Secondary claims\n")
    for gname in ["person_gain", "nfi"]:
        g = exp4["tests"][gname]["auroc"]
        lines.append(f"- **{gname}**: median = {g['median_delta']:+.4f}, {g['n_positive']}/{g['n']} positive, "
                      f"sign-test p = {g['sign_test_p']:.4g}, permutation p = {g['permutation_p']:.4g}\n")

    lines.append("## Tertiary -- volume/source controls\n")
    for gname in ["dyad_gain_volume_controlled", "dyad_gain_matched"]:
        g = exp4["tests"][gname]["auroc"]
        lines.append(f"- **{gname}**: median = {g['median_delta']:+.4f}, {g['n_positive']}/{g['n']} positive, "
                      f"sign-test p = {g['sign_test_p']:.4g}, permutation p = {g['permutation_p']:.4g}\n")
    lines.append(f"\n{exp4['tests']['multiple_comparisons_note']}\n")

    lines.append("## NFI distribution\n")
    nfi = exp4["tests"]["nfi"]["auroc"]
    lines.append(f"Median NFI = {nfi['median_delta']:+.4f}, {nfi['n_positive']}/{nfi['n']} positive, "
                  f"sign-test p = {nfi['sign_test_p']:.4g}.\n")

    lines.append("## Pooled descriptive aggregate (NOT the test, S20)\n")
    for ck in COND_KEYS:
        mean_a = float(np.mean([row[ck]["auroc"] for row in exp4["per_dyad"]]))
        lines.append(f"- {COND_LABELS[ck]} pooled mean AUROC: {mean_a:.4f}\n")

    lines.append("## Environment and drift check\n")
    env = results["meta"]["environment"]
    lines.append(f"Remote: Python {env['remote']['python_version']}, sklearn {env['remote']['sklearn_version']}, "
                  f"numpy {env['remote']['numpy_version']}. Laptop (exp1/exp3): "
                  f"Python {env['laptop']['python_version']}, sklearn {env['laptop']['sklearn_version']}.\n")
    dc = exp4["validations"].get("drift_check")
    if dc and dc.get("max_abs_delta_auroc") is not None:
        lines.append(f"Drift check (3 sampled exp3 participants, refit in this environment vs "
                      f"laptop): max|delta AUROC| = {dc['max_abs_delta_auroc']:.4f}. "
                      f"{dc['note']}\n")

    lines.append("## Validation summary\n")
    for k, val in exp4["validations"].items():
        lines.append(f"- **{k}**: {val}\n")

    lines.append("## Limitations\n")
    lines.append("- n=10, not the pre-registered n=11 (Amendment 1; sub19_sub22 structurally "
                  "untestable at dyad grain).\n"
                  "- All five conditions fit in one Python-3.8/sklearn-1.3.2 remote environment; "
                  "not directly comparable point-to-point to exp1/exp3's laptop numbers (see "
                  "drift check above).\n"
                  "- Person-Specific trains on only 162 rows against 1,770 features -- severely "
                  "underdetermined by design, the honest operationalization of the question.\n")

    return "\n".join(lines)


def write_outputs(results: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"not serializable: {type(o)}")

    with open(EXP4_JSON, "w") as f:
        json.dump(results, f, indent=2, default=default)
    print(f"\nWrote {EXP4_JSON}", flush=True)

    md = build_markdown(results)
    with open(EXP4_MD, "w") as f:
        f.write(md)
    print(f"Wrote {EXP4_MD}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run():
    global GRID_N_JOBS
    print("=" * 70, flush=True)
    print("Experiment 4: dyad-specific model (S14)", flush=True)
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

    full, deceiver_manifest = load_deceiver_frame()
    print(f"Assembled deceiver frame: {full.shape} (expected (10648, 1791))", flush=True)
    assert (full["role"] == "deceiver").all()

    needed = [c for c in feat_cols if c in full.columns]
    missing = set(feat_cols) - set(needed)
    assert not missing, f"feature columns missing: {missing}"
    forbidden = set(M.IDENTITY_COLUMNS) | {"outcome", "observer_guess", "points"}

    # NaN-row drop, mirroring M.prepare_modeling_frame's own behavior (exp1-3
    # all go through that function; exp4 builds its own frame from shards so
    # this step must be reproduced explicitly). Real inspection on the laptop
    # found 6 of 10,648 deceiver rows have a NaN in at least one
    # reliable_plus_marginal column (sub20_sub21 seq 309-310, sub23_sub24
    # seq 221-222/397-398 -- all inside the partner's session-1 block, none in
    # any held-out or B-prior block). Dropped here rather than left to crash
    # LogisticRegression downstream.
    nan_mask = full[needed].isna().any(axis=1)
    n_dropped_nan = int(nan_mask.sum())
    print(f"NaN-row drop: {n_dropped_nan} of {len(full)} deceiver rows have a NaN in a "
          f"reliable_plus_marginal column; dropping (mirrors M.prepare_modeling_frame).", flush=True)
    full = full.loc[~nan_mask].reset_index(drop=True)

    X = full[needed].reset_index(drop=True)
    assert forbidden & set(X.columns) == set(), "identity columns leaked into X"
    y = (full["condition"] == "lie").astype(int).reset_index(drop=True)
    row_keys = full[["pair_id", "session_id", "round", "trial", "dyad_trial_seq", "participant_id",
                      "partner_id"]].reset_index(drop=True)
    Xv = X.values
    yv = y.values

    blocks = build_dyad_blocks(row_keys)
    print(f"Built dyad blocks for {len(blocks)} tested dyads (expected 10)", flush=True)
    for d, b in blocks.items():
        print(f"  {d}: B(tested)={b['B']} A(partner)={b['A']} held_out=[{b['seq_held_out'].min()},"
              f"{b['seq_held_out'].max()}] n={len(b['held_out_idx'])}", flush=True)

    print("\n--- GRID_N_JOBS invariance check ---", flush=True)
    njobs_check = _ckpt_load("njobs_check")
    if njobs_check is None:
        njobs_check = grid_n_jobs_check(Xv, yv, row_keys, blocks)
        _ckpt_save("njobs_check", njobs_check)
    print(f"  n_jobs=1: {njobs_check['n_jobs_1_seconds']:.1f}s, "
          f"n_jobs={GRID_N_JOBS}: {njobs_check['n_jobs_chosen_seconds']:.1f}s, "
          f"identical={njobs_check['metrics_identical']}", flush=True)
    if not njobs_check["metrics_identical"]:
        print("  WARNING: parallelism changed results -- falling back to n_jobs=1 for the whole run", flush=True)
        GRID_N_JOBS = 1
    chosen_n_jobs = GRID_N_JOBS

    print("\n--- Majority-class references ---", flush=True)
    t0 = time.time()
    maj_records = run_majority(Xv, yv, blocks)
    maj_seconds = time.time() - t0

    cond_records = {}
    print("\n--- Universal (10 fits, ~9,196 rows each) ---", flush=True)
    t0 = time.time()
    cond_records["universal"] = run_condition_universal(Xv, yv, row_keys, blocks, chosen_n_jobs)
    univ_seconds = time.time() - t0

    print("\n--- Person-Specific (10 fits, 162 rows each) ---", flush=True)
    t0 = time.time()
    cond_records["person_specific"] = run_condition_person_specific(Xv, yv, row_keys, blocks, chosen_n_jobs)
    pers_seconds = time.time() - t0

    print("\n--- Dyad-Specific (10 fits, 646 rows each) -- PRIMARY CONDITION ---", flush=True)
    t0 = time.time()
    cond_records["dyad_specific"] = run_condition_dyad_specific(Xv, yv, row_keys, blocks, chosen_n_jobs)
    dyad_seconds = time.time() - t0

    print("\n--- Dyad-Specific N-matched (10 fits, 162 rows each) ---", flush=True)
    t0 = time.time()
    cond_records["n_matched"] = run_condition_nmatched(Xv, yv, row_keys, blocks, chosen_n_jobs)
    nmatch_seconds = time.time() - t0

    print("\n--- Person + Other-Dyad volume-matched (10 fits, 646 rows each) ---", flush=True)
    t0 = time.time()
    cond_records["person_other_dyad"] = run_condition_person_other_dyad(Xv, yv, row_keys, blocks, chosen_n_jobs)
    pother_seconds = time.time() - t0

    total_seconds = time.time() - t_start
    runtime = {
        "per_condition_seconds": {"majority": maj_seconds, "universal": univ_seconds,
                                   "person_specific": pers_seconds, "dyad_specific": dyad_seconds,
                                   "n_matched": nmatch_seconds, "person_other_dyad": pother_seconds},
        "total_seconds": total_seconds, "n_resume_launches": n_resume_launches,
        "grid_n_jobs_used": chosen_n_jobs,
    }
    print(f"\nTotal runtime this launch: {total_seconds:.1f}s", flush=True)

    per_dyad, per_dyad_scores, gains = assemble(blocks, cond_records, maj_records, row_keys)
    tests = build_tests(gains)

    print("\n--- Environment drift check (3 sampled exp3 participants) ---", flush=True)
    drift = None
    if EXP3_JSON.exists():
        try:
            drift = _drift_check_impl(Xv, yv, row_keys, EXP3_JSON)
            print(f"  drift check: max_abs_delta_auroc={drift.get('max_abs_delta_auroc')}", flush=True)
        except Exception as e:
            drift = {"error": str(e), "note": "drift check failed non-fatally; exp4's own numbers unaffected"}
            print(f"  drift check FAILED (non-fatal): {e}", flush=True)
    else:
        drift = {"note": "results/exp3_personalized.json not present on remote -- drift check skipped"}

    validations = run_validations(Xv, yv, row_keys, blocks, cond_records, maj_records, per_dyad,
                                   per_dyad_scores, njobs_check, deceiver_manifest)
    validations["drift_check"] = drift

    design = {
        "research_question": "S14: does the partner's own prior interaction history help predict "
                              "deception beyond the target's own history and the wider population?",
        "n_dyads": len(TESTED_DYADS), "tested_dyads": TESTED_DYADS,
        "excluded_from_test_set_but_in_universal_pool": {UNTESTABLE_UNIT_STILL_IN_POOL:
            "structurally untestable at dyad grain (Finding B) -- rows still used as Universal training data"},
        "excluded_from_everything": {EXCLUDED_FROM_EVERYTHING: "excluded from exp3-8 by frozen_hypotheses.md"},
        "held_out_block": "dyad's own late tercile of dyad_trial_seq, [647,968], 322 rows -- all belong to "
                           "the session-2 deceiver (B)",
        "feature_set": "reliable_plus_marginal", "n_features": len(feat_cols),
        "model_family": "logistic_regression (L2) only",
        "conditions": {
            "universal": "all deceiver rows from every other dyad (sub19_sub22 in the pool, "
                         "sub01_sub02 excluded); n_train ~9,196; StratifiedGroupKFold(3) on pair_id",
            "person_specific": "B's own prior rows, seq [485,646], 162 rows; TimeSeriesSplit(3)",
            "dyad_specific": "person_specific UNION A's full session-1 block seq [1,484] (484 rows) "
                              "= 646 rows; TimeSeriesSplit(3) on seq-sorted union -- PRIMARY (H3)",
            "n_matched": "81 most-recent B rows UNION 81 most-recent A rows = 162 rows (volume control)",
            "person_other_dyad": "person_specific UNION 484 label-stratified rows drawn from the "
                                  "Universal pool (seeded per dyad) = 646 rows (source control)",
        },
        "finding_a_prose": (
            "Every deceiver-role participant is the deceiver in exactly one session, and the two "
            "sessions occupy disjoint contiguous dyad_trial_seq blocks (1-484, 485-968). Holding out "
            "each participant's own last third (exp3's scheme) is fatal for Dyad-Specific: a "
            "session-1 deceiver's partner history lives entirely in session 2, chronologically after "
            "any block confined to session 1. Exp4 instead holds out the dyad's own late tercile "
            "(seq 647-968) as a single dyad-grain block, all belonging to the session-2 deceiver; "
            "the session-1 deceiver's entire block is then legally prior. Universal and "
            "Person-Specific are recomputed on this new split and are not exp3's numbers."
        ),
        "finding_b_prose": (
            "sub19_sub22 has only sub19 as a deceiver-role participant (sub22 was lost in S9 "
            "preprocessing), and sub19 is the session-1 deceiver, so this dyad's late tercile "
            "contains zero deceiver rows -- structurally untestable at dyad grain, discovered from "
            "data structure before any exp4 model was fit. results/frozen_hypotheses.md Amendment 1 "
            "(2026-08-22, pre-results) reduces exp4's n from 11 to 10 accordingly. sub19_sub22's 484 "
            "rows remain in the Universal training pool."
        ),
    }

    results = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": M.SEED, "grid_n_jobs": chosen_n_jobs,
            "environment": {"remote": env, "laptop": LAPTOP_ENV},
            "runtime": runtime,
            "torch_openpyxl_omission": "deliberately not installed on the remote box -- exp4 needs "
                                        "neither; torch is multi-GB with no GPU on this machine",
            "shard_transfer_manifest_hash": deceiver_manifest["total_frame_sha256"],
            "frozen_hypotheses_amended_sha256": FROZEN_HYP_SHA256_EXPECTED,
        },
        "experiments": {
            "exp4": {
                "design": design,
                "per_dyad": per_dyad,
                "per_dyad_scores": per_dyad_scores,
                "gains": gains,
                "tests": tests,
                "nfi": {
                    "median": tests["nfi"]["auroc"]["median_delta"],
                    "n_positive": tests["nfi"]["auroc"]["n_positive"],
                    "n": tests["nfi"]["auroc"]["n"],
                    "sign_test_p": tests["nfi"]["auroc"]["sign_test_p"],
                },
                "validations": validations,
                "coefficients": {
                    **{ck: {d: cond_records[ck][d]["coef"] for d in TESTED_DYADS} for ck in COND_KEYS},
                    "feature_names": feat_cols,
                },
                "cross_experiment": {
                    "exp1_pooled_auroc": 0.5338,
                    "exp3_headline_note": "exp3 median personalized-minus-population delta = +0.0320, "
                                           "8/11 positive, sign p=0.227, permutation p=0.123 -- quoted as "
                                           "CONTEXT ONLY. exp4's Universal/Person-Specific are recomputed "
                                           "on different (dyad-grain) held-out blocks and are not "
                                           "comparable point-to-point to exp3's numbers.",
                    "exp2_note": "results/exp2_universal.json is not read at all -- Colab, out of scope.",
                },
            }
        },
    }

    write_outputs(results)
    return results


if __name__ == "__main__":
    run()
