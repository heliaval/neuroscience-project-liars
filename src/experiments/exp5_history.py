"""
src/experiments/exp5_history.py -- Experiment 5 driver (S15): interaction-history model.

Research question (S15): does deception become more distinguishable after two
participants have interacted for longer? This module answers it three ways, all
on the same ten dyads and the same held-out block T_d exp4 used, so every exp5
number is directly comparable to exp4's on identical trials:

  1. Positional analysis (fixed model, moving test bin) -- is a Universal model
     more accurate on later trials than earlier ones? Two grains: 1A (dyad-grain
     terciles, powered but confounded with participant identity) and 1B
     (participant-grain terciles, unconfounded but underpowered/exploratory).
  2. The relationship-learning curve -- a 5-rung prefix ladder (k in
     {81,162,322,484,646}) training on d's own rows with dyad_trial_seq<=k,
     always testing on T_d. The frozen 2-point design (k=322 vs k=646) is the
     confirmatory primary claim; the other 3 rungs are pre-declared exploratory.
  3. The mandatory same-dyad vs other-dyad volume control -- at every rung, a
     size- and label-matched draw from other dyads, so "more history helps" can
     be told apart from "more training rows of any kind help".

Runs on the same remote Ryzen 5 2600 / WSL2 box exp4 ran on, against data
already uploaded there (12 per-dyad shards + manifest + feature dictionary).
src/models.py is not modified. Structurally cloned from exp4_dyadic.py: same
shard loader, same helpers, same S20 paired-test machinery copied verbatim
(never imported across experiment modules), same per-unit checkpointing, same
JSON+MD output shape.

--------------------------------------------------------------------------------
FACT 1 -- the session structure forces the design (why 1A and 1B both exist)
--------------------------------------------------------------------------------
Every participant is the deceiver in exactly one session, and the two sessions
occupy disjoint contiguous dyad_trial_seq blocks: 1-484 (deceiver A, session 1)
and 485-968 (deceiver B, session 2). A naive dyad-grain early/middle/late split
is therefore CONFOUNDED with participant identity -- the early tercile [1,322]
is entirely A's trials, the late tercile [647,968] is entirely B's. "Late is
more distinguishable than early" would be indistinguishable from "B is more
distinguishable than A" under that split (Analysis 1A). Analysis 1B instead
splits EACH deceiver's own 484-row block into three, so the same relative
position (early/middle/late within a person's own block) is compared across
both A and B -- unconfounded with identity, at the cost of 1/3 the per-bin row
count, which is what fails the gate (see Fact 3).

--------------------------------------------------------------------------------
FACT 2 -- exp5 is n=10, not the pre-registered n=11 (Amendment 2, pre-results)
--------------------------------------------------------------------------------
sub19_sub22's only deceiver-role participant (sub19) is the session-1 deceiver,
so this dyad's late tercile [647,968] -- exp5's held-out block, identical to
exp4's -- contains zero deceiver rows. Structurally untestable, the same
situation Amendment 1 already resolved for exp4. results/frozen_hypotheses.md
Amendment 2 (2026-08-22, pre-results) reduces exp5's n from 11 to 10
accordingly; sub19_sub22's rows remain in the other-dyad/Universal training
pool (never a test dyad, so no leakage). sub01_sub02 stays excluded from
exp3-8 as already frozen. Tested dyads are exp4's ten, unchanged:

    sub03_sub06  sub04_sub05  sub07_sub08  sub09_sub10  sub11_sub12
    sub13_sub14  sub15_sub16  sub17_sub18  sub20_sub21  sub23_sub24

--------------------------------------------------------------------------------
FACT 3 -- exp5's own gate re-check, real measured minority counts
--------------------------------------------------------------------------------
The frozen gate.json's exp5 row was computed at exp3's coarser participant
grain for a dyad (sub01_sub02) exp5 doesn't even test, and pre-registers only
a 2-point curve -- it doesn't cover 1A/1B/Analysis-3's finer grains. Per
Amendment 2, THRESHOLD_M=60 and the Clause A/B wording are re-applied
UNCHANGED (not re-derived) at exp5's actual grain:

  | analysis | test-fold grain      | Clause A | Clause B  | designation                    |
  |----------|-----------------------|----------|-----------|---------------------------------|
  | 1A       | 322/324/322, min 130  | PASS     | 10/10 PASS| confirmatory on power, confound stated in headline |
  | 1B       | 162/161/161, min 45   | FAIL     | 8/10 FAIL | EXPLORATORY -- underpowered     |
  | 2        | 322, min 134          | PASS     | 10/10 PASS| CONFIRMATORY at k=322,646; other rungs exploratory |
  | 3        | 322, min 134          | PASS     | 10/10 PASS| CONFIRMATORY at k=646; other rungs exploratory |

1B fails on sub24 (45, its middle within-participant bin -- the same
participant exp3 already flagged as its single sub-threshold case) and sub17
(59, its middle bin). Kept and reported per S7's rule: demote, don't drop.

--------------------------------------------------------------------------------
THE LADDER, per dyad d (B=session-2 deceiver, A=session-1 deceiver/partner)
--------------------------------------------------------------------------------
Prefix training set = d's own rows with dyad_trial_seq<=k, always tested on
T_d = B's rows with seq in [647,968] (322 rows, identical to exp4's).

  rung | k   | n_train | composition        | designation
  -----|-----|---------|---------------------|---------------------------------
  1    | 81  | 81      | A only              | exploratory
  2    | 162 | 162     | A only              | exploratory
  3    | 322 | 322     | A only              | CONFIRMATORY (frozen: "early tercile only")
  4    | 484 | 484     | A only              | exploratory
  5    | 646 | 646     | A(484)+B(162)       | CONFIRMATORY (frozen: "early+middle")

The composition change at rung 5 (own-participant rows enter for the first
time) is a real confound, published per rung (n_rows_A/n_rows_B), not buried:
a rise from rung 4 to rung 5 could be relationship learning OR the arrival of
B's own-person data. This is exactly why Analysis 3's same-vs-other-dyad
control exists.

--------------------------------------------------------------------------------
ANALYSIS 3 -- inner-CV symmetry, a real design decision
--------------------------------------------------------------------------------
The other-dyad arm's rows have no shared timeline with d, so TimeSeriesSplit on
them would assert an ordering that does not exist. To avoid the paired
difference partly reflecting an inner-CV artifact, BOTH arms of the control use
StratifiedKFold(3, shuffle=False) -- the same-dyad arm is therefore refit a
second time under a different inner scheme than Analysis 2's TimeSeriesSplit,
which doubles as a free inner-CV sensitivity check (validation 7), reported,
not treated as a pass/fail gate.

--------------------------------------------------------------------------------
CLAIM HIERARCHY (declared before any number exists)
--------------------------------------------------------------------------------
  PRIMARY:     history_gain = AUROC(k=646) - AUROC(k=322), same-dyad. No correction.
  SECONDARY:   same_vs_other_dyad_at_646; positional_late_minus_early (1A). No correction (pre-registered).
  EXPLORATORY: all other rung comparisons, curve slopes, history_gain_volume_controlled
               (difference-in-differences), within_person_late_minus_early (1B).
               Benjamini-Hochberg, alpha=0.05, within this family, stated before any fit.

within_person_late_minus_early is exploratory on two independent grounds: the
gate failure above, and the fact that it was never in the frozen 2-point design.

--------------------------------------------------------------------------------
S19 LEAKAGE RULES / S20 TEST / PYTHON 3.8 HAZARD
--------------------------------------------------------------------------------
Chronological splits only within-dyad; every training row's dyad_trial_seq is
strictly less than every test row's. Sign test (scipy.stats.binomtest,
two-sided, ties excluded) + 10,000-draw sign-flip permutation test on the
median, per S20; the pooled aggregate is descriptive only, never the test.
Remote is Python 3.8.10 -- NO `{...} | {...}` dict union (3.9+ syntax; this
crashed exp4 once, after a full successful run, at output assembly). Use
`{**a, **b}` throughout. Verified by pre-upload grep (see Task 3's checklist).

--------------------------------------------------------------------------------
CHECKPOINTING
--------------------------------------------------------------------------------
exp5_checkpoints/<key>.pkl (relative to this machine's reveriehacks26/ root --
a NEW directory, never exp4_checkpoints/). ~170 units total (10 Universal +
90 positional-bin scorings folded into the Universal unit + 50 TSS-ladder +
50 SKF-ladder-control + 50 other-dyad-control + 10 majority), so a dropped
connection costs at most one unit.
"""

from __future__ import annotations

import hashlib
import json
import math
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
EXP5_JSON = OUT_DIR / "exp5_history.json"
EXP5_MD = OUT_DIR / "exp5_history.md"
EXP4_JSON = OUT_DIR / "exp4_dyadic.json"
GATE_JSON = RESULTS_DIR / "gate.json"
FROZEN_HYP_MD = RESULTS_DIR / "frozen_hypotheses.md"
SHARD_MANIFEST = FEATURES_DIR / "shard_manifest.json"

CHECKPOINT_DIR = REPO_ROOT / "exp5_checkpoints"

THRESHOLD_M = 60
N_SIGNFLIP = 10000
METRICS = ["auroc", "balanced_accuracy", "f1", "precision", "recall"]
GRID_N_JOBS = 6

LADDER_K = [81, 162, 322, 484, 646]
CONFIRMATORY_RUNGS = (322, 646)

TESTED_DYADS = [
    "sub03_sub06", "sub04_sub05", "sub07_sub08", "sub09_sub10", "sub11_sub12",
    "sub13_sub14", "sub15_sub16", "sub17_sub18", "sub20_sub21", "sub23_sub24",
]
EXCLUDED_FROM_EVERYTHING = "sub01_sub02"
UNTESTABLE_UNIT_STILL_IN_POOL = "sub19_sub22"

# sha256 of results/frozen_hypotheses.md AFTER Amendment 2 was appended on the
# laptop (recorded immediately after the append, per plan Task 1 step 4).
FROZEN_HYP_SHA256_EXPECTED = "e8f7235601abc6dc1f6b4441af24959728598b78bd6792c346fb29aad4ebda31"

LAPTOP_ENV = {
    "python_version": "3.13.14", "sklearn_version": "1.8.0", "numpy_version": "2.3.2",
    "scipy_version": "1.17.1", "pandas_version": "2.3.2", "pyarrow_version": "25.0.1",
    "xgboost_version": "3.4.1",
}


# ---------------------------------------------------------------------------
# Checkpointing (copied from exp4_dyadic.py)
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
# (copied from exp4_dyadic.py)
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


def build_dyad_blocks_exp5(row_keys: pd.DataFrame) -> dict:
    """Per dyad: A (session-1 deceiver), B (session-2 deceiver), held_out_idx
    (B's rows seq>=647), dyad_tercile_bins (3 dyad-grain bins over the full
    [1,968] block), person_tercile_bins (6 bins: A's own 484-row block in 3
    contiguous seq-ordered pieces, then B's own 484-row block in 3), and
    prefix_idx[k] for each k in LADDER_K (d's own rows with seq<=k)."""
    blocks = {}
    seq_full = row_keys["dyad_trial_seq"].values  # indexed by GLOBAL row position -- sort_by_seq needs this, not the dyad-local slice
    for d in TESTED_DYADS:
        dyad_mask = row_keys["pair_id"].values == d
        dyad_idx = np.where(dyad_mask)[0]
        seq = seq_full[dyad_idx]
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

        # Dyad-grain terciles over the full seq range [1,968].
        all_sorted = sort_by_seq(dyad_idx, seq_full)
        dyad_tercile_bins = _contig_bins(all_sorted, 3)

        # Participant-grain: A's own block (seq<=484), then B's own block
        # (seq>=485), each split into 3 contiguous seq-ordered pieces.
        a_block_idx = dyad_idx[(pid == A) & (seq <= 484)]
        b_block_idx = dyad_idx[(pid == B) & (seq >= 485)]
        a_sorted = sort_by_seq(a_block_idx, seq_full)
        b_sorted = sort_by_seq(b_block_idx, seq_full)
        person_tercile_bins = _contig_bins(a_sorted, 3) + _contig_bins(b_sorted, 3)

        # Prefix ladder: d's own rows with seq<=k, sorted by seq.
        prefix_idx = {}
        for k in LADDER_K:
            mask_k = seq <= k
            prefix_idx[k] = sort_by_seq(dyad_idx[mask_k], seq_full)

        assert 315 <= len(held_out_idx) <= 322, f"{d}: held-out block size {len(held_out_idx)} implausible"
        assert 470 <= len(a_block_idx) <= 484, f"{d}: A block size {len(a_block_idx)} implausible"
        assert 470 <= len(b_block_idx) <= 484, f"{d}: B block size {len(b_block_idx)} implausible"

        blocks[d] = {
            "B": B, "A": A,
            "held_out_idx": held_out_idx,
            "seq_held_out": seq[held_out_mask],
            "dyad_tercile_bins": dyad_tercile_bins,
            "person_tercile_bins": person_tercile_bins,
            "prefix_idx": prefix_idx,
            "a_block_idx": a_block_idx,
            "b_block_idx": b_block_idx,
        }
    return blocks


# ---------------------------------------------------------------------------
# Fit helpers (copied from exp4_dyadic.py)
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
# S20 paired test machinery (copied verbatim from exp4_dyadic.py -- not
# imported across experiment modules; src/models.py not modified)
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


def _benjamini_hochberg(pvals: list, alpha: float = 0.05) -> list:
    """Returns BH-adjusted p-values (same order as input), standard step-up."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p-value downward
    adj_sorted = np.minimum.accumulate(adj[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    out = np.empty(n)
    out[order] = adj_sorted
    return out.tolist()


# ---------------------------------------------------------------------------
# GRID_N_JOBS invariance check (copied pattern from exp4)
# ---------------------------------------------------------------------------

def grid_n_jobs_check(Xv, yv, blocks):
    d = TESTED_DYADS[0]
    train_idx = blocks[d]["prefix_idx"][322]
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
# Analysis 1 -- Universal fit + positional scoring (10 units)
# ---------------------------------------------------------------------------

def _score_on_idx(Xv, yv, fitted_search, idx):
    fitted = fitted_search.best_estimator_
    clf = fitted.named_steps["clf"]
    Xt = fitted[:-1].transform(Xv[idx]) if len(fitted.steps) > 1 else Xv[idx]
    y_score = clf.predict_proba(Xt)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xt)
    y_pred = fitted.predict(Xv[idx])
    return M.compute_metrics(yv[idx], y_score, y_pred)


def run_universal_and_score_bins(Xv, yv, row_keys, blocks, n_jobs):
    pair_id_arr = row_keys["pair_id"].values
    records = {}
    for d in TESTED_DYADS:
        key = f"univ_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            records[d] = cached
            print(f"  [checkpoint] universal {d}: loaded", flush=True)
            continue
        t0 = time.time()
        train_idx = universal_pool_idx(row_keys, d)
        test_idx = blocks[d]["held_out_idx"]
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

        held_out_metrics = _score_on_idx(Xv, yv, search, test_idx)

        # 1A -- dyad-grain terciles (3 bins)
        positional_bins = []
        bin_names = ["early", "middle", "late"]
        for bname, bidx in zip(bin_names, blocks[d]["dyad_tercile_bins"]):
            seq_bin = row_keys["dyad_trial_seq"].values[bidx]
            pids_bin = sorted(set(row_keys["participant_id"].values[bidx].tolist()))
            m = _score_on_idx(Xv, yv, search, bidx)
            n_lie = int(yv[bidx].sum())
            positional_bins.append({
                "bin": bname, "seq_min": int(seq_bin.min()), "seq_max": int(seq_bin.max()),
                "n": int(len(bidx)), "n_minority": int(min(n_lie, len(bidx) - n_lie)),
                "participants": pids_bin, **{m_: m[m_] for m_ in METRICS},
            })

        # 1B -- participant-grain terciles (6 bins: A's 3, then B's 3)
        wp_bin_names = ["early", "middle", "late"]
        wp_bins_raw = []
        for person_label, offset in [("A", 0), ("B", 3)]:
            pid_val = blocks[d][person_label]
            for i in range(3):
                bidx = blocks[d]["person_tercile_bins"][offset + i]
                m = _score_on_idx(Xv, yv, search, bidx)
                n_lie = int(yv[bidx].sum())
                wp_bins_raw.append({
                    "participant": pid_val, "position": wp_bin_names[i],
                    "n": int(len(bidx)), "n_minority": int(min(n_lie, len(bidx) - n_lie)),
                    **{m_: m[m_] for m_ in METRICS},
                })
        # aggregate per position (mean of A's and B's score at same position)
        within_person_bins = []
        for i, pos in enumerate(wp_bin_names):
            a_rec = wp_bins_raw[i]
            b_rec = wp_bins_raw[3 + i]
            mean_metrics = {m_: float(np.mean([a_rec[m_], b_rec[m_]])) for m_ in METRICS}
            within_person_bins.append({
                "bin": pos, "n": a_rec["n"] + b_rec["n"],
                "n_minority": a_rec["n_minority"] + b_rec["n_minority"],
                "per_participant": [a_rec, b_rec],
                "mean_auroc": mean_metrics["auroc"], **mean_metrics,
            })

        rec = {
            "metrics": held_out_metrics, "best_params": search.best_params_,
            "n_train": int(len(train_idx)), "convergence_warnings": conv_warnings,
            "inner_splitter_class": type(inner).__name__,
            "train_pair_ids": sorted(set(pair_id_arr[train_idx].tolist())),
            "positional_bins": positional_bins,
            "within_person_bins": within_person_bins,
        }
        _ckpt_save(key, rec)
        records[d] = rec
        print(f"  universal {d}: n_train={rec['n_train']} held_out_auroc={held_out_metrics['auroc']:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return records


# ---------------------------------------------------------------------------
# Analysis 2 -- prefix ladder, same-dyad, TimeSeriesSplit inner (50 units)
# ---------------------------------------------------------------------------

def run_ladder_tss(Xv, yv, row_keys, blocks, n_jobs):
    records = {}
    for d in TESTED_DYADS:
        for k in LADDER_K:
            key = f"ladder_tss_{d}_k{k}"
            cached = _ckpt_load(key)
            if cached is not None:
                records[(d, k)] = cached
                print(f"  [checkpoint] ladder-tss {d} k={k}: loaded", flush=True)
                continue
            t0 = time.time()
            train_idx = blocks[d]["prefix_idx"][k]
            test_idx = blocks[d]["held_out_idx"]
            inner, inner_class, fallback = _time_series_inner(train_idx, yv)
            rec = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=n_jobs, collect_coef=False)
            rec["inner_splitter_class"] = inner_class
            rec["fallback_to_stratified"] = fallback
            n_rows_a = int((row_keys["participant_id"].values[train_idx] == blocks[d]["A"]).sum())
            n_rows_b = int((row_keys["participant_id"].values[train_idx] == blocks[d]["B"]).sum())
            rec["n_rows_A"] = n_rows_a
            rec["n_rows_B"] = n_rows_b
            rec["n_lie"] = int(yv[train_idx].sum())
            rec["n_truth"] = int(len(train_idx) - rec["n_lie"])
            _ckpt_save(key, rec)
            records[(d, k)] = rec
            print(f"  ladder-tss {d} k={k}: n_train={rec['n_train']} auroc={rec['metrics']['auroc']:.4f} "
                  f"fallback={fallback} ({time.time()-t0:.1f}s)", flush=True)
    return records


def run_ladder_skf(Xv, yv, row_keys, blocks, n_jobs):
    records = {}
    for d in TESTED_DYADS:
        for k in LADDER_K:
            key = f"ladder_skf_{d}_k{k}"
            cached = _ckpt_load(key)
            if cached is not None:
                records[(d, k)] = cached
                print(f"  [checkpoint] ladder-skf {d} k={k}: loaded", flush=True)
                continue
            t0 = time.time()
            train_idx = blocks[d]["prefix_idx"][k]
            test_idx = blocks[d]["held_out_idx"]
            inner = StratifiedKFold(n_splits=3, shuffle=False)
            rec = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=n_jobs, collect_coef=False)
            rec["inner_splitter_class"] = "StratifiedKFold"
            _ckpt_save(key, rec)
            records[(d, k)] = rec
            print(f"  ladder-skf {d} k={k}: n_train={rec['n_train']} auroc={rec['metrics']['auroc']:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return records


# ---------------------------------------------------------------------------
# Analysis 3 -- other-dyad matched draw, StratifiedKFold inner (50 units)
# ---------------------------------------------------------------------------

def run_ladder_other(Xv, yv, row_keys, blocks, n_jobs):
    records = {}
    for dyad_index, d in enumerate(TESTED_DYADS):
        for k in LADDER_K:
            key = f"ladder_other_{d}_k{k}"
            cached = _ckpt_load(key)
            if cached is not None:
                records[(d, k)] = cached
                print(f"  [checkpoint] ladder-other {d} k={k}: loaded", flush=True)
                continue
            t0 = time.time()
            same_idx = blocks[d]["prefix_idx"][k]
            n_lie = int(yv[same_idx].sum())
            n_truth = len(same_idx) - n_lie
            pool_idx = universal_pool_idx(row_keys, d)
            draw_idx = stratified_draw(pool_idx, yv, n_lie, n_truth, [M.SEED, dyad_index, k])
            test_idx = blocks[d]["held_out_idx"]
            inner = StratifiedKFold(n_splits=3, shuffle=False)
            rec = _fit_predict(Xv, yv, draw_idx, test_idx, inner, n_jobs=n_jobs, collect_coef=False)
            rec["inner_splitter_class"] = "StratifiedKFold"
            rec["drawn_idx"] = draw_idx.tolist()
            rec["other_dyad_pair_ids"] = sorted(set(row_keys["pair_id"].values[draw_idx].tolist()))
            rec["n_lie"] = n_lie
            rec["n_truth"] = n_truth
            _ckpt_save(key, rec)
            records[(d, k)] = rec
            print(f"  ladder-other {d} k={k}: n_train={rec['n_train']} auroc={rec['metrics']['auroc']:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return records


# ---------------------------------------------------------------------------
# Majority-class reference (10 units)
# ---------------------------------------------------------------------------

def run_majority(Xv, yv, blocks):
    records = {}
    for d in TESTED_DYADS:
        key = f"major_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            records[d] = cached
            continue
        train_idx = blocks[d]["prefix_idx"][646]
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
# Assembly
# ---------------------------------------------------------------------------

def _slope_log2k(k_list, auroc_list):
    x = np.log2(np.array(k_list, dtype=float))
    y = np.array(auroc_list, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def assemble(blocks, univ_records, tss_records, skf_records, other_records, maj_records):
    per_dyad = []
    curve_slopes = []
    for d in TESTED_DYADS:
        b = blocks[d]
        curve = []
        curve_skf = []
        control_curve_other = []
        tss_aurocs, other_aurocs = [], []
        for k in LADDER_K:
            r_tss = tss_records[(d, k)]
            r_skf = skf_records[(d, k)]
            r_other = other_records[(d, k)]
            curve.append({
                "k": k, "n_train": r_tss["n_train"], "n_train_trials": r_tss["n_train"],
                "n_rows_A": r_tss["n_rows_A"], "n_rows_B": r_tss["n_rows_B"],
                "n_lie": r_tss["n_lie"], "n_truth": r_tss["n_truth"],
                "best_params": r_tss["best_params"], "inner_splitter_class": r_tss["inner_splitter_class"],
                **{m: r_tss["metrics"][m] for m in METRICS},
            })
            curve_skf.append({
                "k": k, "n_train": r_skf["n_train"], "best_params": r_skf["best_params"],
                "inner_splitter_class": r_skf["inner_splitter_class"],
                **{m: r_skf["metrics"][m] for m in METRICS},
            })
            control_curve_other.append({
                "k": k, "n_train": r_other["n_train"], "n_train_trials": r_other["n_train"],
                "n_lie": r_other["n_lie"], "n_truth": r_other["n_truth"],
                "other_dyad_pair_ids": r_other["other_dyad_pair_ids"],
                "best_params": r_other["best_params"], "inner_splitter_class": r_other["inner_splitter_class"],
                **{m: r_other["metrics"][m] for m in METRICS},
            })
            tss_aurocs.append(r_tss["metrics"]["auroc"])
            other_aurocs.append(r_other["metrics"]["auroc"])

        slope_same = _slope_log2k(LADDER_K, tss_aurocs)
        slope_other = _slope_log2k(LADDER_K, other_aurocs)
        curve_slopes.append({"pair_id": d, "slope_same_dyad": slope_same, "slope_other_dyad": slope_other})

        u = univ_records[d]
        row = {
            "pair_id": d, "tested_participant": b["B"], "partner": b["A"],
            "held_out_seq_min": int(b["seq_held_out"].min()), "held_out_seq_max": int(b["seq_held_out"].max()),
            "n_test": int(len(b["held_out_idx"])),
            "curve": curve, "curve_skf": curve_skf,
            "control_curve_other_dyad": control_curve_other,
            "positional_bins": u["positional_bins"],
            "within_person_bins": u["within_person_bins"],
            "majority_reference": maj_records[d],
        }
        per_dyad.append(row)

    return per_dyad, curve_slopes


def build_tests(per_dyad, curve_slopes):
    by_d = {row["pair_id"]: row for row in per_dyad}
    curve_by_k = {d: {c["k"]: c for c in by_d[d]["curve"]} for d in TESTED_DYADS}
    other_by_k = {d: {c["k"]: c for c in by_d[d]["control_curve_other_dyad"]} for d in TESTED_DYADS}

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

    tests = {}

    # PRIMARY: history_gain = AUROC(646) - AUROC(322), same-dyad
    deltas = [curve_by_k[d][646]["auroc"] - curve_by_k[d][322]["auroc"] for d in TESTED_DYADS]
    tests["history_gain"] = {"result": _paired_test(deltas), "designation": "confirmatory", "primary": True}

    # SECONDARY: same_vs_other_dyad_at_646
    deltas = [curve_by_k[d][646]["auroc"] - other_by_k[d][646]["auroc"] for d in TESTED_DYADS]
    tests["same_vs_other_dyad_at_646"] = {"result": _paired_test(deltas), "designation": "confirmatory",
                                           "primary": False, "tier": "secondary"}

    # SECONDARY: positional_late_minus_early (1A)
    def _bin_auroc(d, bname):
        for pb in by_d[d]["positional_bins"]:
            if pb["bin"] == bname:
                return pb["auroc"]
        raise KeyError(bname)
    deltas = [_bin_auroc(d, "late") - _bin_auroc(d, "early") for d in TESTED_DYADS]
    tests["positional_late_minus_early"] = {"result": _paired_test(deltas), "designation": "confirmatory",
                                             "primary": False, "tier": "secondary"}

    # EXPLORATORY family (BH-corrected together)
    exploratory_raw = {}

    # within_person_late_minus_early (1B) -- exploratory on two grounds
    def _wp_auroc(d, bname):
        for wb in by_d[d]["within_person_bins"]:
            if wb["bin"] == bname:
                return wb["mean_auroc"]
        raise KeyError(bname)
    deltas = [_wp_auroc(d, "late") - _wp_auroc(d, "early") for d in TESTED_DYADS]
    exploratory_raw["within_person_late_minus_early"] = _paired_test(deltas)

    # same_vs_other_dyad_at_k for k in {81,162,322,484}
    for k in [81, 162, 322, 484]:
        deltas = [curve_by_k[d][k]["auroc"] - other_by_k[d][k]["auroc"] for d in TESTED_DYADS]
        exploratory_raw[f"same_vs_other_dyad_at_{k}"] = _paired_test(deltas)

    # history_gain_volume_controlled: DiD = [same(646)-same(322)] - [other(646)-other(322)]
    deltas = []
    for d in TESTED_DYADS:
        same_diff = curve_by_k[d][646]["auroc"] - curve_by_k[d][322]["auroc"]
        other_diff = other_by_k[d][646]["auroc"] - other_by_k[d][322]["auroc"]
        deltas.append(same_diff - other_diff)
    exploratory_raw["history_gain_volume_controlled"] = _paired_test(deltas)

    # adjacent-rung gains (4)
    for i in range(len(LADDER_K) - 1):
        k_lo, k_hi = LADDER_K[i], LADDER_K[i + 1]
        deltas = [curve_by_k[d][k_hi]["auroc"] - curve_by_k[d][k_lo]["auroc"] for d in TESTED_DYADS]
        exploratory_raw[f"adjacent_rung_gain_{k_lo}_to_{k_hi}"] = _paired_test(deltas)

    # slope test: same-dyad slope vs other-dyad slope (paired per-dyad difference)
    deltas = [cs["slope_same_dyad"] - cs["slope_other_dyad"] for cs in curve_slopes]
    exploratory_raw["slope_test"] = _paired_test(deltas)

    # BH correction across the exploratory family
    names = list(exploratory_raw.keys())
    pvals = [exploratory_raw[n_]["sign_test_p"] for n_ in names]
    adj = _benjamini_hochberg(pvals, alpha=0.05)
    for n_, p_adj in zip(names, adj):
        tests[n_] = {"result": exploratory_raw[n_], "designation": "exploratory", "primary": False,
                     "bh_adjusted_p": p_adj, "bh_family_size": len(names)}

    tests["adjacent_rung_gains"] = [f"adjacent_rung_gain_{LADDER_K[i]}_to_{LADDER_K[i+1]}"
                                     for i in range(len(LADDER_K) - 1)]
    tests["primary"] = "history_gain"
    tests["secondary"] = ["same_vs_other_dyad_at_646", "positional_late_minus_early"]
    tests["exploratory"] = names
    tests["multiple_comparisons_note"] = (
        "history_gain (k=646 vs k=322, same-dyad) is the single pre-registered primary claim, no "
        "correction. same_vs_other_dyad_at_646 and positional_late_minus_early are pre-registered "
        "secondaries, no correction. Every other rung comparison, curve slope, the "
        "difference-in-differences (history_gain_volume_controlled), and within_person_late_minus_early "
        "are exploratory, Benjamini-Hochberg corrected at alpha=0.05 within that family. Declared before "
        "any number below was computed, per results/frozen_hypotheses.md Amendment 2."
    )
    return tests


# ---------------------------------------------------------------------------
# Validations
# ---------------------------------------------------------------------------

def run_validations(Xv, yv, row_keys, blocks, univ_records, tss_records, skf_records,
                     other_records, maj_records, per_dyad, curve_slopes, tests,
                     n_jobs_check_result, deceiver_manifest):
    print("\n" + "=" * 70, flush=True)
    print("Validations", flush=True)
    print("=" * 70, flush=True)
    v = {}
    seq_arr = row_keys["dyad_trial_seq"].values
    pair_id_arr = row_keys["pair_id"].values

    # 1. chronological integrity, ladder
    ok1 = True
    detail1 = {}
    for d in TESTED_DYADS:
        test_idx = blocks[d]["held_out_idx"]
        test_min = int(seq_arr[test_idx].min())
        per_k = {}
        for k in LADDER_K:
            train_idx = blocks[d]["prefix_idx"][k]
            train_max = int(seq_arr[train_idx].max())
            ok = train_max < test_min
            ok1 = ok1 and ok
            per_k[k] = {"train_max_seq": train_max, "test_min_seq": test_min, "chronological": ok}
        detail1[d] = per_k
    v["1_chronological_ladder"] = {"ok": ok1, "per_dyad": detail1}
    print(f"1. chronological-ladder: ok={ok1}", flush=True)

    # 2. nesting -- strict subset, monotone sizes
    ok2 = True
    detail2 = {}
    for d in TESTED_DYADS:
        sizes = [len(blocks[d]["prefix_idx"][k]) for k in LADDER_K]
        monotone = all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1))
        subset_ok = True
        for i in range(len(LADDER_K) - 1):
            s_lo = set(blocks[d]["prefix_idx"][LADDER_K[i]].tolist())
            s_hi = set(blocks[d]["prefix_idx"][LADDER_K[i + 1]].tolist())
            subset_ok = subset_ok and s_lo.issubset(s_hi) and len(s_lo) < len(s_hi)
        ok2 = ok2 and monotone and subset_ok
        detail2[d] = {"sizes": sizes, "monotone": monotone, "strict_subset": subset_ok}
    v["2_nesting"] = {"ok": ok2, "per_dyad": detail2}
    print(f"2. nesting: ok={ok2}", flush=True)

    # 3. identical held-out block across all ladder families + majority
    ok3 = True
    v["3_identical_held_out_block"] = {
        "ok": ok3, "note": "True by construction -- run_ladder_tss/run_ladder_skf/run_ladder_other/"
                            "run_universal_and_score_bins/run_majority all pass blocks[d]['held_out_idx'] "
                            "directly, never a per-condition copy."}
    print(f"3. identical-held-out-block: ok={ok3}", flush=True)

    # 4. bin integrity, both grains -- reproduces Fact 3's tables in-environment
    ok4 = True
    detail4_dyad = {}
    detail4_person = {}
    min_minority_dyad = None
    min_minority_person = None
    for d in TESTED_DYADS:
        # dyad-grain: partition [1,968] exactly, no overlap
        all_idx = np.concatenate(blocks[d]["dyad_tercile_bins"])
        full_dyad_idx = np.where(row_keys["pair_id"].values == d)[0]
        partitions_dyad = set(all_idx.tolist()) == set(full_dyad_idx.tolist()) and len(all_idx) == len(full_dyad_idx)
        for bname, bidx in zip(["early", "middle", "late"], blocks[d]["dyad_tercile_bins"]):
            n_lie = int(yv[bidx].sum())
            n_min = min(n_lie, len(bidx) - n_lie)
            detail4_dyad.setdefault(d, {})[bname] = {"n": len(bidx), "n_minority": n_min}
            min_minority_dyad = n_min if min_minority_dyad is None else min(min_minority_dyad, n_min)
        ok4 = ok4 and partitions_dyad

        # participant-grain: each deceiver's 484-row block partitioned exactly
        a_all = np.concatenate(blocks[d]["person_tercile_bins"][0:3])
        b_all = np.concatenate(blocks[d]["person_tercile_bins"][3:6])
        partitions_a = set(a_all.tolist()) == set(blocks[d]["a_block_idx"].tolist())
        partitions_b = set(b_all.tolist()) == set(blocks[d]["b_block_idx"].tolist())
        ok4 = ok4 and partitions_a and partitions_b
        for label, offset in [("A", 0), ("B", 3)]:
            for i, bname in enumerate(["early", "middle", "late"]):
                bidx = blocks[d]["person_tercile_bins"][offset + i]
                n_lie = int(yv[bidx].sum())
                n_min = min(n_lie, len(bidx) - n_lie)
                detail4_person.setdefault(d, {})[f"{label}_{bname}"] = {"n": len(bidx), "n_minority": n_min}
                min_minority_person = n_min if min_minority_person is None else min(min_minority_person, n_min)
    v["4_bin_integrity"] = {
        "ok": ok4, "dyad_grain": detail4_dyad, "person_grain": detail4_person,
        "min_minority_dyad_grain": min_minority_dyad, "min_minority_person_grain": min_minority_person,
        "reproduces_fact3_130": min_minority_dyad == 130 if min_minority_dyad is not None else None,
        "reproduces_fact3_45": min_minority_person == 45 if min_minority_person is not None else None,
    }
    print(f"4. bin-integrity: ok={ok4}, min_minority_dyad={min_minority_dyad}, "
          f"min_minority_person={min_minority_person}", flush=True)

    # 5. volume matching
    ok5 = True
    detail5 = {}
    for d in TESTED_DYADS:
        for k in LADDER_K:
            same_idx = blocks[d]["prefix_idx"][k]
            other_rec = other_records[(d, k)]
            n_same = len(same_idx)
            n_other = other_rec["n_train"]
            n_lie_same = int(yv[same_idx].sum())
            n_truth_same = n_same - n_lie_same
            ok = (n_same == n_other) and (n_lie_same == other_rec["n_lie"]) and (n_truth_same == other_rec["n_truth"])
            ok5 = ok5 and ok
            detail5[f"{d}_k{k}"] = {"n_same": n_same, "n_other": n_other,
                                     "n_lie_same": n_lie_same, "n_lie_other": other_rec["n_lie"], "ok": ok}
    v["5_volume_matching"] = {"ok": ok5, "sample": dict(list(detail5.items())[:5])}
    print(f"5. volume-matching: ok={ok5}", flush=True)

    # 6. other-dyad pool exclusion
    ok6 = True
    all_contributing_pids = set()
    for d in TESTED_DYADS:
        for k in LADDER_K:
            pids = set(other_records[(d, k)]["other_dyad_pair_ids"])
            all_contributing_pids |= pids
            clean = d not in pids and EXCLUDED_FROM_EVERYTHING not in pids
            ok6 = ok6 and clean
    v["6_other_dyad_pool_exclusion"] = {"ok": ok6, "contributing_pair_ids": sorted(all_contributing_pids)}
    print(f"6. other-dyad-pool-exclusion: ok={ok6}", flush=True)

    # 7. inner-CV sensitivity
    max_diff = 0.0
    diffs = {}
    for d in TESTED_DYADS:
        for k in LADDER_K:
            diff = abs(tss_records[(d, k)]["metrics"]["auroc"] - skf_records[(d, k)]["metrics"]["auroc"])
            diffs[f"{d}_k{k}"] = diff
            max_diff = max(max_diff, diff)
    v["7_inner_cv_sensitivity"] = {"max_abs_diff": max_diff, "exceeds_0.05": max_diff > 0.05,
                                    "note": "Reported measurement, not a pass/fail gate."}
    print(f"7. inner-cv-sensitivity: max_abs_diff={max_diff:.4f}", flush=True)

    # 8. no identity columns in X -- checked at run() call site, recorded here
    forbidden = set(M.IDENTITY_COLUMNS) | {"outcome", "observer_guess", "points"}
    v["8_no_identity_columns"] = {"forbidden_set": sorted(forbidden), "checked_at": "run() before .values"}
    print("8. no-identity-columns: checked at run() call site", flush=True)

    # 9. majority-class sanity
    maj_aurocs = [m["auroc"] for m in maj_records.values()]
    maj_bal = [m["balanced_accuracy"] for m in maj_records.values()]
    ok9 = all(abs(a - 0.5) <= 0.02 for a in maj_aurocs) and all(abs(b - 0.5) <= 0.02 for b in maj_bal)
    v["9_majority_class_sanity"] = {"mean_auroc": float(np.mean(maj_aurocs)),
                                     "mean_balanced_accuracy": float(np.mean(maj_bal)), "ok": ok9}
    print(f"9. majority-class-sanity: {v['9_majority_class_sanity']}", flush=True)

    # 10. reproducibility -- 3 sampled units, bypass checkpoint cache
    sample_units = [("tss", TESTED_DYADS[0], 322), ("tss", TESTED_DYADS[1], 646), ("skf", TESTED_DYADS[2], 646)]
    diffs10 = {}
    for kind, d, k in sample_units:
        train_idx = blocks[d]["prefix_idx"][k]
        test_idx = blocks[d]["held_out_idx"]
        if kind == "tss":
            inner, _, _ = _time_series_inner(train_idx, yv)
        else:
            inner = StratifiedKFold(n_splits=3, shuffle=False)
        r1 = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=1)
        r2 = _fit_predict(Xv, yv, train_idx, test_idx, inner, n_jobs=1)
        diffs10[f"{kind}_{d}_k{k}"] = abs(r1["metrics"]["auroc"] - r2["metrics"]["auroc"])
    v["10_reproducibility"] = {"max_diff": max(diffs10.values()), "per_unit": diffs10}
    print(f"10. reproducibility: max_diff={v['10_reproducibility']['max_diff']:.2e}", flush=True)

    # 11. GRID_N_JOBS invariance
    v["11_grid_n_jobs_invariance"] = n_jobs_check_result
    print(f"11. grid-n-jobs-invariance: identical={n_jobs_check_result['metrics_identical']}", flush=True)

    # 12. metric cross-check
    from sklearn.metrics import roc_auc_score
    from scipy.stats import mannwhitneyu
    d0 = TESTED_DYADS[0]
    train_idx = blocks[d0]["prefix_idx"][322]
    test_idx = blocks[d0]["held_out_idx"]
    pipe = M._lr_pipeline("l2", M.SEED)
    pipe.set_params(clf__C=tss_records[(d0, 322)]["best_params"]["clf__C"])
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

    # 13. upstream file hashes
    upstream = {
        "models.py": REPO_ROOT / "src" / "models.py",
        "gate.json": GATE_JSON,
        "shard_manifest.json": SHARD_MANIFEST,
        "frozen_hypotheses.md": FROZEN_HYP_MD,
    }
    hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in upstream.items()}
    frozen_hash_ok = hashes["frozen_hypotheses.md"] == FROZEN_HYP_SHA256_EXPECTED
    v["13_upstream_files_hashed"] = {"hashes": hashes, "frozen_hypotheses_matches_amended_hash": frozen_hash_ok}
    print(f"13. upstream-files-hashed: frozen_matches={frozen_hash_ok}", flush=True)

    # 14. data integrity
    v["14_data_integrity"] = {
        "row_count_10648_before_nan_drop": deceiver_manifest["n_rows_total"] == 10648,
        "total_frame_hash_verified": "asserted in load_deceiver_frame() -- would have raised if mismatched",
    }
    print(f"14. data-integrity: {v['14_data_integrity']}", flush=True)

    # 15. gate verdicts recorded
    v["15_gate_verdicts_recorded"] = {
        "designations_present": {
            "1A": "confirmatory_on_power_confounded",
            "1B": "exploratory_underpowered",
            "2_frozen_rungs": "confirmatory", "2_other_rungs": "exploratory",
            "3_k646": "confirmatory", "3_other_rungs": "exploratory",
        },
        "no_test_labelled_confirmatory_outside_hierarchy": (
            tests["history_gain"]["designation"] == "confirmatory"
            and tests["same_vs_other_dyad_at_646"]["designation"] == "confirmatory"
            and tests["positional_late_minus_early"]["designation"] == "confirmatory"
            and all(tests[n_]["designation"] == "exploratory" for n_ in tests["exploratory"])
        ),
    }
    print(f"15. gate-verdicts-recorded: {v['15_gate_verdicts_recorded']}", flush=True)

    # 16. amendment recorded
    amend_text = FROZEN_HYP_MD.read_text()
    amend_present = "Amendment 2" in amend_text and "n = 10" in amend_text
    v["16_amendment_recorded"] = {
        "amendment_2_present": amend_present, "n_dyads": len(TESTED_DYADS),
        "sub19_sub22_absent_from_tested_dyads": UNTESTABLE_UNIT_STILL_IN_POOL not in TESTED_DYADS,
    }
    print(f"16. amendment-recorded: {v['16_amendment_recorded']}", flush=True)

    # 17. plausibility
    all_aurocs = [c["auroc"] for row in per_dyad for c in row["curve"]]
    flagged = []
    for row in per_dyad:
        d = row["pair_id"]
        hg = next(c for c in row["curve"] if c["k"] == 646)["auroc"] - next(c for c in row["curve"] if c["k"] == 322)["auroc"]
        if abs(hg) > 0.20:
            flagged.append(d)
        if any(c["auroc"] > 0.75 for c in row["curve"]):
            flagged.append(d)
    v["17_plausibility"] = {
        "max_auroc": float(max(all_aurocs)),
        "dyads_flagged": sorted(set(flagged)),
        "leakage_suspicion": bool(len(flagged) > 0),
        "note": "Flagging is for inspection, not automatic exclusion (exp3/exp4 precedent).",
    }
    print(f"17. plausibility: {v['17_plausibility']}", flush=True)

    # 18. convergence
    n_conv_tss = sum(len(tss_records[(d, k)]["convergence_warnings"]) for d in TESTED_DYADS for k in LADDER_K)
    n_conv_skf = sum(len(skf_records[(d, k)]["convergence_warnings"]) for d in TESTED_DYADS for k in LADDER_K)
    n_conv_other = sum(len(other_records[(d, k)]["convergence_warnings"]) for d in TESTED_DYADS for k in LADDER_K)
    n_conv_univ = sum(len(univ_records[d]["convergence_warnings"]) for d in TESTED_DYADS)
    v["18_convergence"] = {"n_convergence_warnings": {
        "ladder_tss": n_conv_tss, "ladder_skf": n_conv_skf, "ladder_other": n_conv_other, "universal": n_conv_univ,
    }}
    print(f"18. convergence: {v['18_convergence']}", flush=True)

    return v


def _cross_experiment_check(per_dyad):
    """Descriptive comparison against exp4's dyad_specific condition -- same
    646 rows, same T_d, same environment -- should be ~0 up to inner-CV scheme."""
    if not EXP4_JSON.exists():
        return {"note": "out/exp4_dyadic.json not present on remote -- cross-experiment check skipped"}
    exp4 = json.loads(EXP4_JSON.read_text())
    exp4_per_dyad = {row["pair_id"]: row for row in exp4["experiments"]["exp4"]["per_dyad"]}
    per_dyad_diff = {}
    for row in per_dyad:
        d = row["pair_id"]
        if d not in exp4_per_dyad:
            continue
        exp5_646 = next(c["auroc"] for c in row["curve"] if c["k"] == 646)
        exp4_dyad_specific = exp4_per_dyad[d]["dyad_specific"]["auroc"]
        per_dyad_diff[d] = exp5_646 - exp4_dyad_specific
    return {
        "per_dyad_diff_exp5_k646_minus_exp4_dyad_specific": per_dyad_diff,
        "max_abs_diff": max((abs(v_) for v_ in per_dyad_diff.values()), default=None),
        "note": "exp5's k=646 same-dyad rung is fit on the identical 646 rows and scored on the "
                "identical T_d as exp4's dyad_specific condition, in the same environment. Expected "
                "~0 up to the inner-CV scheme (both use TimeSeriesSplit here); any systematic gap "
                "is worth reporting, not automatically a bug.",
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def build_markdown(results: dict) -> str:
    exp5 = results["experiments"]["exp5"]
    lines = []
    lines.append("# Experiment 5 -- Interaction-History Model (S15)\n")
    lines.append(f"**What this experiment asks:** {exp5['design']['research_question']}\n")

    lines.append("## Design\n")
    d = exp5["design"]
    lines.append(f"- n = {d['n_dyads']} dyads (amended from pre-registered n=11; see "
                  "`results/frozen_hypotheses.md` Amendment 2).\n"
                  f"- Held-out block: {d['held_out_block']}\n"
                  f"- Ladder rungs: {d['ladder_k']}, confirmatory rungs: {d['confirmatory_rungs']}\n"
                  f"- Feature set: {d['feature_set']} ({d['n_features']} cols). Model: {d['model_family']}\n")
    lines.append(d.get("identity_confound_prose", "") + "\n")
    lines.append(d.get("n10_prose", "") + "\n")

    lines.append("## Gate re-check (Amendment 2)\n")
    gr = exp5["gate_recheck"]
    lines.append(f"THRESHOLD_M = {gr['threshold']} (copied unchanged from the original gate).\n\n")
    lines.append("| analysis | test-fold grain | minority | Clause A | Clause B | designation |\n"
                  "|---|---|---|---|---|---|\n")
    for row in gr["rows"]:
        lines.append(f"| {row['analysis']} | {row['grain']} | {row['minority']} | {row['clause_a']} | "
                      f"{row['clause_b']} | {row['designation']} |\n")
    lines.append(f"\nFailing bins: {gr['failing_bins']}\n")

    lines.append("\n## Per-dyad curve table (AUROC, same-dyad TimeSeriesSplit)\n")
    lines.append("| pair_id | " + " | ".join(f"k={k}" for k in d["ladder_k"]) + " |\n")
    lines.append("|---|" + "---|" * len(d["ladder_k"]) + "\n")
    for row in exp5["per_dyad"]:
        cells = " | ".join(f"{c['auroc']:.4f}" for c in row["curve"])
        lines.append(f"| {row['pair_id']} | {cells} |\n")

    lines.append("\n## Primary claim -- history_gain\n")
    hg = exp5["tests"]["history_gain"]["result"]
    lines.append(f"Median = {hg['median_delta']:+.4f}, {hg['n_positive']}/{hg['n']} dyads positive, "
                  f"sign-test p = {hg['sign_test_p']:.4g}, permutation p = {hg['permutation_p']:.4g}. "
                  f"95% CI: [{hg['ci95']['lower']:.4f}, {hg['ci95']['upper']:.4f}].\n")

    lines.append("\n## Secondary claims\n")
    for name in ["same_vs_other_dyad_at_646", "positional_late_minus_early"]:
        r = exp5["tests"][name]["result"]
        lines.append(f"- **{name}**: median = {r['median_delta']:+.4f}, {r['n_positive']}/{r['n']} positive, "
                      f"sign-test p = {r['sign_test_p']:.4g}, permutation p = {r['permutation_p']:.4g}\n")

    lines.append("\n## Control -- same-dyad vs other-dyad volume\n")
    lines.append("Reported inside secondary claims (`same_vs_other_dyad_at_646`) and the exploratory "
                  "per-rung breakdown below.\n")

    lines.append("\n## Exploratory (Benjamini-Hochberg corrected)\n")
    for name in exp5["tests"]["exploratory"]:
        t = exp5["tests"][name]
        r = t["result"]
        lines.append(f"- **{name}**: median = {r['median_delta']:+.4f}, {r['n_positive']}/{r['n']} positive, "
                      f"sign-test p = {r['sign_test_p']:.4g} (BH-adjusted: {t['bh_adjusted_p']:.4g})\n")
    lines.append(f"\n{exp5['tests']['multiple_comparisons_note']}\n")

    lines.append("\n## Aggregate (NOT the test, S20)\n")
    for k in d["ladder_k"]:
        same_vals = [c["auroc"] for row in exp5["per_dyad"] for c in row["curve"] if c["k"] == k]
        other_vals = [c["auroc"] for row in exp5["per_dyad"] for c in row["control_curve_other_dyad"] if c["k"] == k]
        lines.append(f"- k={k}: same-dyad median AUROC = {np.median(same_vals):.4f}, "
                      f"other-dyad median AUROC = {np.median(other_vals):.4f}\n")

    lines.append("\n## Analysis 1B -- within-participant early/middle/late (EXPLORATORY, UNDERPOWERED)\n")
    lines.append("Fails the exp5-specific gate re-check (Clause A on sub24=45, Clause B at 8/10). "
                  "Reported with CIs per the frozen file's own rule, never dropped.\n")
    r = exp5["tests"]["within_person_late_minus_early"]["result"]
    lines.append(f"Median = {r['median_delta']:+.4f}, {r['n_positive']}/{r['n']} positive, "
                  f"sign-test p = {r['sign_test_p']:.4g}. 95% CI: [{r['ci95']['lower']:.4f}, "
                  f"{r['ci95']['upper']:.4f}].\n")

    lines.append("\n## Environment\n")
    env = results["meta"]["environment"]
    lines.append(f"Remote: Python {env['remote']['python_version']}, sklearn {env['remote']['sklearn_version']}. "
                  f"Laptop: Python {env['laptop']['python_version']}, sklearn {env['laptop']['sklearn_version']}.\n")

    lines.append("\n## Validation summary\n")
    for k, val in exp5["validations"].items():
        lines.append(f"- **{k}**: {val}\n")

    lines.append("\n## Cross-experiment comparison (exp4)\n")
    lines.append(f"{exp5['cross_experiment']}\n")

    lines.append("\n## Limitations\n")
    lines.append(
        "- n=10, not the pre-registered n=11 (Amendment 2; sub19_sub22 structurally untestable).\n"
        "- Analysis 1A (dyad-grain positional split) is confounded with participant identity: the early "
        "tercile is entirely deceiver A's trials, the late tercile entirely deceiver B's -- 'late is more "
        "distinguishable' cannot be told apart from 'B is more distinguishable than A' at this grain.\n"
        "- Analysis 1B (participant-grain, unconfounded) is underpowered and exploratory -- reported "
        "with CIs, not dropped.\n"
        "- The ladder's rung 5 (k=646) introduces the tested participant's own rows for the first time; "
        "a rise from rung 4 to rung 5 conflates relationship learning with the arrival of own-person data "
        "-- Analysis 3's control is the mitigation, not a full resolution.\n"
        "- A most-recent-k 'recency ladder' (train on the k most recent rows before the held-out block, "
        "rather than a growing prefix) would separate 'more history' from 'more recent history' -- a "
        "genuinely different question S15 does not ask. Deliberately out of scope for this pass.\n"
        "- All conditions fit in one Python-3.8/sklearn-1.3.2 remote environment; not directly comparable "
        "point-to-point to laptop-fit experiments.\n"
    )

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
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"not serializable: {type(o)}")

    with open(EXP5_JSON, "w") as f:
        json.dump(results, f, indent=2, default=default)
    print(f"\nWrote {EXP5_JSON}", flush=True)

    md = build_markdown(results)
    with open(EXP5_MD, "w") as f:
        f.write(md)
    print(f"Wrote {EXP5_MD}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run():
    global GRID_N_JOBS
    print("=" * 70, flush=True)
    print("Experiment 5: interaction-history model (S15)", flush=True)
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
    print(f"Assembled deceiver frame: {full.shape}", flush=True)
    assert (full["role"] == "deceiver").all()

    needed = [c for c in feat_cols if c in full.columns]
    missing = set(feat_cols) - set(needed)
    assert not missing, f"feature columns missing: {missing}"
    forbidden = set(M.IDENTITY_COLUMNS) | {"outcome", "observer_guess", "points"}

    nan_mask = full[needed].isna().any(axis=1)
    n_dropped_nan = int(nan_mask.sum())
    print(f"NaN-row drop: {n_dropped_nan} of {len(full)} deceiver rows dropped (mirrors exp4).", flush=True)
    full = full.loc[~nan_mask].reset_index(drop=True)

    X = full[needed].reset_index(drop=True)
    assert forbidden & set(X.columns) == set(), "identity columns leaked into X"
    y = (full["condition"] == "lie").astype(int).reset_index(drop=True)
    row_keys = full[["pair_id", "session_id", "round", "trial", "dyad_trial_seq", "participant_id",
                      "partner_id"]].reset_index(drop=True)
    Xv = X.values
    yv = y.values

    blocks = build_dyad_blocks_exp5(row_keys)
    print(f"Built dyad blocks for {len(blocks)} tested dyads (expected 10)", flush=True)

    print("\n--- GRID_N_JOBS invariance check ---", flush=True)
    njobs_check = _ckpt_load("njobs_check")
    if njobs_check is None:
        njobs_check = grid_n_jobs_check(Xv, yv, blocks)
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
    maj_records = run_majority(Xv, yv, blocks)
    maj_seconds = time.time() - t0

    print("\n--- Universal + positional scoring (10 units) ---", flush=True)
    t0 = time.time()
    univ_records = run_universal_and_score_bins(Xv, yv, row_keys, blocks, chosen_n_jobs)
    univ_seconds = time.time() - t0

    print("\n--- Ladder, same-dyad, TimeSeriesSplit (50 units) -- Analysis 2 ---", flush=True)
    t0 = time.time()
    tss_records = run_ladder_tss(Xv, yv, row_keys, blocks, chosen_n_jobs)
    tss_seconds = time.time() - t0

    print("\n--- Ladder, same-dyad, StratifiedKFold control (50 units) ---", flush=True)
    t0 = time.time()
    skf_records = run_ladder_skf(Xv, yv, row_keys, blocks, chosen_n_jobs)
    skf_seconds = time.time() - t0

    print("\n--- Ladder, other-dyad matched draw (50 units) -- Analysis 3 ---", flush=True)
    t0 = time.time()
    other_records = run_ladder_other(Xv, yv, row_keys, blocks, chosen_n_jobs)
    other_seconds = time.time() - t0

    total_seconds = time.time() - t_start
    runtime = {
        "per_condition_seconds": {"majority": maj_seconds, "universal": univ_seconds,
                                   "ladder_tss": tss_seconds, "ladder_skf": skf_seconds,
                                   "ladder_other": other_seconds},
        "total_seconds": total_seconds, "n_resume_launches": n_resume_launches,
        "grid_n_jobs_used": chosen_n_jobs,
    }
    print(f"\nTotal runtime this launch: {total_seconds:.1f}s", flush=True)

    per_dyad, curve_slopes = assemble(blocks, univ_records, tss_records, skf_records, other_records, maj_records)
    tests = build_tests(per_dyad, curve_slopes)

    aggregate = {"note": "descriptive only -- NOT the test (S20)", "by_k": {}}
    for k in LADDER_K:
        same_vals = [c["auroc"] for row in per_dyad for c in row["curve"] if c["k"] == k]
        other_vals = [c["auroc"] for row in per_dyad for c in row["control_curve_other_dyad"] if c["k"] == k]
        aggregate["by_k"][k] = {
            "same_dyad_median": float(np.median(same_vals)), "same_dyad_spread": [float(min(same_vals)), float(max(same_vals))],
            "other_dyad_median": float(np.median(other_vals)), "other_dyad_spread": [float(min(other_vals)), float(max(other_vals))],
        }

    cross_experiment = _cross_experiment_check(per_dyad)

    validations = run_validations(Xv, yv, row_keys, blocks, univ_records, tss_records, skf_records,
                                   other_records, maj_records, per_dyad, curve_slopes, tests,
                                   njobs_check, deceiver_manifest)

    gate_recheck = {
        "threshold": THRESHOLD_M, "threshold_copied_not_rederived": True,
        "clause_a": "smallest test-fold minority-class trial count >= THRESHOLD_M",
        "clause_b": "at least 10 of 12 dyads (or the equivalent fraction at reduced n; n=10 => >=9) individually clear Clause A",
        "rows": [
            {"analysis": "1A dyad-grain positional", "grain": "322/324/322", "minority": 130,
             "clause_a": "PASS", "clause_b": "10/10 PASS", "designation": "confirmatory_on_power_confounded"},
            {"analysis": "1B participant-grain positional", "grain": "162/161/161", "minority": 45,
             "clause_a": "FAIL", "clause_b": "8/10 FAIL", "designation": "exploratory_underpowered"},
            {"analysis": "2 learning curve", "grain": "322 (late tercile)", "minority": 134,
             "clause_a": "PASS", "clause_b": "10/10 PASS", "designation": "confirmatory_at_frozen_rungs"},
            {"analysis": "3 volume control", "grain": "322 (late tercile)", "minority": 134,
             "clause_a": "PASS", "clause_b": "10/10 PASS", "designation": "confirmatory_at_k646"},
        ],
        "failing_bins": {"sub24": 45, "sub17": 59},
        "amendment_reference": "results/frozen_hypotheses.md Amendment 2",
    }

    design = {
        "research_question": "S15: does deception become more distinguishable after participants have "
                              "interacted for longer?",
        "n_dyads": len(TESTED_DYADS), "tested_dyads": TESTED_DYADS,
        "excluded_from_test_set_but_in_universal_pool": {UNTESTABLE_UNIT_STILL_IN_POOL:
            "structurally untestable at dyad grain (Fact 2/Amendment 2) -- rows still used as other-dyad training data"},
        "excluded_from_everything": {EXCLUDED_FROM_EVERYTHING: "excluded from exp3-8 by frozen_hypotheses.md"},
        "held_out_block": "dyad's own late tercile of dyad_trial_seq, [647,968], 322 rows -- identical to exp4's, "
                           "all belong to the session-2 deceiver (B)",
        "ladder_k": LADDER_K, "confirmatory_rungs": list(CONFIRMATORY_RUNGS),
        "feature_set": "reliable_plus_marginal", "n_features": len(feat_cols),
        "model_family": "logistic_regression (L2) only",
        "analyses": ["positional_1a_dyad_grain", "positional_1b_participant_grain",
                     "learning_curve_ladder", "same_vs_other_dyad_volume_control"],
        "direction_convention": "positive delta = more history helps",
        "identity_confound_prose": (
            "Every participant is the deceiver in exactly one session, occupying disjoint contiguous "
            "dyad_trial_seq blocks (1-484, 485-968). Analysis 1A's dyad-grain terciles are therefore "
            "confounded with participant identity -- the early tercile is entirely deceiver A's trials, "
            "the late tercile entirely deceiver B's. This is published as data (per-bin participant ids "
            "and A/B row counts), not smoothed over."
        ),
        "n10_prose": (
            "sub19_sub22's only deceiver-role participant (sub19) is the session-1 deceiver, so its late "
            "tercile (exp5's held-out block) contains zero deceiver rows -- structurally untestable. "
            "results/frozen_hypotheses.md Amendment 2 reduces exp5's n from 11 to 10 accordingly; "
            "sub19_sub22's rows remain in the other-dyad training pool."
        ),
    }

    results = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": M.SEED, "grid_n_jobs": chosen_n_jobs,
            "environment": {"remote": env, "laptop": LAPTOP_ENV},
            "runtime": runtime,
            "shard_transfer_manifest_hash": deceiver_manifest["total_frame_sha256"],
            "frozen_hypotheses_amended_sha256": FROZEN_HYP_SHA256_EXPECTED,
        },
        "experiments": {
            "exp5": {
                "design": design,
                "gate_recheck": gate_recheck,
                "per_dyad": per_dyad,
                "curve_slopes": curve_slopes,
                "aggregate": aggregate,
                "tests": tests,
                "validations": validations,
                "cross_experiment": cross_experiment,
            }
        },
    }

    write_outputs(results)
    return results


if __name__ == "__main__":
    run()
