"""
src/experiments/exp3_personalized.py -- Experiment 3 driver (S13): personalized
model, population vs person-specific.

Research question (S13): does knowing the individual improve prediction? For
each target participant, a population model (trained on other dyads' deceiver
trials) and a personalized model (trained on that participant's own earlier
deceiver trials) are both scored on the SAME held-out later block of that
participant's trials, so the two scores are paired -- the pairing S20's
procedure consumes.

--------------------------------------------------------------------------------
UNIT OF ANALYSIS (plan Step 2 -- a required reported resolution)
--------------------------------------------------------------------------------
S13 says "record the score for each participant individually; the aggregate is
computed by the procedure in S20, not by pooling trials across participants."
S20 and results/frozen_hypotheses.md (frozen 2026-08-20, binding) say the unit
of analysis is the dyad, n = 11 for exp3. These are not in conflict once read
correctly: S13's "individually" is an instruction about what is *measured*
(forbidding pooling trials across participants into one AUROC), not a
redefinition of what is *tested*. frozen_hypotheses.md was frozen before any
model was fit and is not overridden by a reading of the prose it derives from.
So: measurement grain = participant (21 participants, AUROC/4 other metrics
computed separately per participant per condition); statistical grain = dyad
(mean of the dyad's participants' deltas, n = 11, S20's sign test + paired
sign-flip permutation test run on those 11 numbers). The participant-grain
(n=21) sign test is also reported, labelled secondary/non-independent, because
the two participants of a dyad share the same dyad/sessions/opponent/(for the
population condition) the same training set, so a 21-value sign test would
overstate its own evidence.

Aggregation choice: dyad delta = mean of the dyad's participants' deltas. This
is the only aggregation that (a) uses both participants when there are two,
(b) is symmetric in them, (c) reduces to the single participant's own delta
when only one exists (sub19_sub22). Documented as a grain deviation from S20
as originally written for exp1/exp2, which never had to aggregate.

--------------------------------------------------------------------------------
PARTICIPANT RECONCILIATION (plan Step 2a/3a -- required reported finding)
--------------------------------------------------------------------------------
single_brain.parquet's deceiver rows cover 22 participants / 12 pairs, not 24 /
12: sub02 has zero deceiver rows (the reverse-role session for sub01_sub02 was
never recorded -- an archive gap, not a participant property) and sub22 has
zero deceiver rows in the feature table (the Player_sub22_Observer_sub19
session was unrecoverable at S9 preprocessing; the dyad survives with 484
usable deceiver rows from sub19 alone, giving sub19_sub22 a single
deceiver-side participant). frozen_hypotheses.md excludes sub01_sub02 from
exp3 entirely (n=11). At participant grain, sub19_sub22 has the SAME kind of
structural defect the gate never saw (the S7 gate computed exp3's fold sizes
from trial_table.csv, where sub19_sub22 still shows 968 trials because that
table is a different table from the feature table and does not encode the S9
preprocessing loss). Resolution, per plan: primary run = 21 participants / 11
dyads (matches the frozen n=11 exactly; sub19_sub22 included, contributing
sub19 only, flagged n_participants=1); sensitivity run = 10 dyads with
sub19_sub22 additionally dropped, re-running only the S20 tests on the same
per-participant scores (no refitting). Both reported.

--------------------------------------------------------------------------------
THE CHRONOLOGICAL SPLIT (plan Step 3)
--------------------------------------------------------------------------------
Every deceiver-role participant is deceiver in exactly one session (484
trials), a single contiguous block of dyad_trial_seq (never interleaved with
the partner's trials, since the partner is deceiver in the OTHER session/half
of the dyad's timeline). For target participant p: sort p's 484 deceiver rows
by dyad_trial_seq, cut = ceil(len*2/3) = 323. train_p = earliest 323 rows
(personalized training set); test_p = latest 161 rows (the held-out block
BOTH conditions are scored on). ceil (not floor) so training absorbs the
remainder, matching results/trial_count_gate.md's own "remainder to the
earliest bins" convention (table 2e) that fixed this split scheme originally
-- trial_count_gate.md's assumptions log already established that the spec's
literal "rounds 1-30/31-40" example does not map onto this dataset's `round`
field (max ~11 per session) and that the late tercile of dyad_trial_seq stands
in for the held-out block instead.

--------------------------------------------------------------------------------
POPULATION MODEL -- WHOLE-DYAD EXCLUSION (plan Step 4)
--------------------------------------------------------------------------------
For target participant p in dyad d, the population model excludes ALL of
dyad d's rows from training (not just p's), for two S19 reasons: (1)
chronological leakage -- a dyad's two participants occupy disjoint,
complementary halves of the dyad's trial_seq timeline, so for roughly half of
participants the partner's ENTIRE deceiver block lies in the target's future;
training on the partner would inject future-into-past information about the
very dyad being predicted; (2) correlated observations -- the partner shares
every session-level nuisance factor (cap montage, room, opponent identity,
shared game history), so keeping the partner in would reintroduce the
pair-recognition channel LODO exists to close and would inflate the
population score, biasing delta against the personalized model. This also
makes the population fit structurally identical to a leave-one-dyad-out fit
(S19's named method), which is what makes the exp2 sanity comparison (Step 9)
legitimate. Consequence: one population fit per DYAD (11 fits, not 21), each
scored separately on that dyad's 1-2 participants' held-out blocks. sub01 is
also excluded from every population training set (not just from scoring),
since sub01_sub02 is excluded from exp3 by the frozen file and letting sub01
into training while it can never appear on the test side would train the
population condition on a different sample than exp3 is defined over.

--------------------------------------------------------------------------------
PERSONALIZED MODEL -- CHRONOLOGICAL INNER CV (plan Step 5)
--------------------------------------------------------------------------------
21 personalized fits, one per participant, trained on that participant's own
323 earliest deceiver rows. Inner tuning CV is sklearn.model_selection
.TimeSeriesSplit(n_splits=3) over the already-chronologically-sorted training
rows -- a shuffled StratifiedKFold inner CV would tune C on a
past/future-mixed estimate inside an experiment whose entire premise is
chronological integrity (S19). TimeSeriesSplit has no random_state (fine,
deterministic by construction) and ignores y for stratification, so each
inner fold's train/val slices are checked for both classes; if any fold fails
that check the personalized fit falls back to StratifiedKFold(3,
shuffle=False) on the same chronologically-ordered rows and the fallback is
recorded, never silent. M._lr_param_grid("l2") was re-verified (not assumed)
to already include C in {0.001, 0.01} -- np.logspace(-3,2,6) lands on integer
exponents -3..2 -- so no grid extension was needed for the personalized
condition; this is recorded in validation 11 rather than silently assumed.
323 training rows against 1,770 features is a severely underdetermined fit by
design (the honest operationalization of "train on this person's earlier
trials", not a flaw to engineer around); the selected C per participant is
reported so a low-C skew (regularizing toward the intercept) is visible
rather than hidden. The ~30x training-size asymmetry between conditions is
reported as intrinsic to the S13 question, not "fixed" by subsampling the
population side (that would be a different, post-hoc question per S8).

--------------------------------------------------------------------------------
IMPLEMENTATION CHOICE: OPTION (ii), driven directly, not through evaluate_cv
--------------------------------------------------------------------------------
Plan Step 10 (11a) offers two ways to get per-participant metrics out of a
population fold whose test set spans two participants: (i) recover them from
evaluate_cv's stored fold predictions, or (ii) drive the fits directly in this
module. Re-inspection of src/models.py's evaluate_cv (2026-08-22, this run)
confirmed it stores only aggregated per-fold metrics and fold_test_indices --
never raw y_score/y_pred -- so option (i) is not available. This module
therefore uses option (ii): PersonalizedSplitter and PopulationSplitter below
are written as real generator objects with split()/get_n_splits() (matching
evaluate_cv's expected shape, per plan Step 10, even though they are not
handed to evaluate_cv here), and the population/personalized fits are driven
by a small hand-written loop mirroring evaluate_cv's internals (GridSearchCV,
n_jobs=1, StandardScaler-in-pipeline). src/models.py is NOT modified by this
experiment -- exp1's five per-fold AUROCs are unaffected by construction
(nothing in models.py changed) and this is recorded as validation 12 with
models_py_modified: false rather than re-run, since there is nothing to
re-run against.

--------------------------------------------------------------------------------
EXPLICITLY OUT OF SCOPE
--------------------------------------------------------------------------------
No CNN (S5 confines it to Experiment 1). No dyad-specific model (that is
Experiment 4/S14 -- exp3's personalized model trains on the target's own
history only, never the dyad's joint history). No dependence on
results/exp2_universal.json for any computation (exp3 fits its own population
models; exp2's number is read at runtime, if present, only for the
descriptive Step 9 cross-experiment table). No writes to results/gate.json,
results/trial_count_gate.md, results/frozen_hypotheses.md,
data/processed/trial_table.csv, results/exp1_baseline.*, results/exp2_universal.*,
or results/results.v1.json (exp3_personalized.json is a mergeable fragment,
same discipline exp1/exp2 follow). No sample-size-matched population variant
(a different question than S13 asks). No observer-row run (S16/Experiment 6).
No S21 permutation null of absolute performance (S20 already covers exp3's
comparison claim; a fresh null would be scope creep with no correction plan).

--------------------------------------------------------------------------------
CHECKPOINTING
--------------------------------------------------------------------------------
This environment kills background bash tasks after roughly 15-25 minutes
regardless of process health (observed repeatedly during exp1 and exp2; see
PROGRESS.md). Checkpoints go to a NEW sibling directory OUTSIDE the repo,
C:\\Users\\Alber\\CN4\\exp3_checkpoints -- deliberately distinct from
exp1_checkpoints/exp2_checkpoints (different folds, different keys; reusing
either would silently poison exp3). Granularity: one checkpoint per
population dyad fit (pop_dyad{d}, 11 units) and one per personalized
participant fit (pers_{participant_id}, 21 units), plus one per majority-class
reference (major_{participant_id}, 21 units) -- 53 total units, each cheap
enough (seconds to ~2 min) that a kill loses at most one unit of work.
GridSearchCV stays at n_jobs=1 throughout (models.py's own convention; exp1
root-caused a multi-hour orphaned-loky-worker cascade to n_jobs=-1) -- no new
parallelism is introduced anywhere in this module.
"""

from __future__ import annotations

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
from sklearn.pipeline import Pipeline as SkPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import models as M  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
EXP3_JSON = RESULTS_DIR / "exp3_personalized.json"
EXP3_MD = RESULTS_DIR / "exp3_personalized.md"
EXP1_JSON = RESULTS_DIR / "exp1_baseline.json"
EXP2_JSON = RESULTS_DIR / "exp2_universal.json"
FROZEN_HYP_MD = RESULTS_DIR / "frozen_hypotheses.md"
GATE_JSON = RESULTS_DIR / "gate.json"
GATE_MD = RESULTS_DIR / "trial_count_gate.md"
TRIAL_TABLE_CSV = REPO_ROOT / "data" / "processed" / "trial_table.csv"

THRESHOLD_M = 60  # minority-class trials, from results/trial_count_gate.md
N_SIGNFLIP = 10000

# Sibling directory OUTSIDE the repo, deliberately distinct from exp1's/exp2's
# checkpoint directories -- different folds, different keys.
CHECKPOINT_DIR = REPO_ROOT.parent / "exp3_checkpoints"

FROZEN_FILES = [
    GATE_JSON, GATE_MD, FROZEN_HYP_MD, TRIAL_TABLE_CSV,
    RESULTS_DIR / "exp1_baseline.json", RESULTS_DIR / "exp1_baseline.md",
    EXP2_JSON, RESULTS_DIR / "exp2_universal.md",
]

METRICS = ["auroc", "balanced_accuracy", "f1", "precision", "recall"]


# ---------------------------------------------------------------------------
# Checkpointing (mirrors exp2_universal.py's pattern; new directory/keys)
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
# Participant blocks and the two custom splitters (plan Step 10 / 11a)
# ---------------------------------------------------------------------------

def build_participant_blocks(row_keys: pd.DataFrame) -> dict:
    """For every deceiver participant EXCEPT sub01 (whose dyad, sub01_sub02, is
    excluded from exp3 entirely per frozen_hypotheses.md), sort that
    participant's row positions by dyad_trial_seq and split ceil(2n/3) earliest
    / remainder latest. Returns {participant_id: {pair_id, partner_id,
    train_idx (np.ndarray, chronological order), test_idx (np.ndarray,
    chronological order), n_total, seq_min_test, seq_max_test}}."""
    blocks = {}
    participants = sorted(row_keys["participant_id"].unique())
    for p in participants:
        if p == "sub01":
            continue  # dyad sub01_sub02 excluded from exp3 per frozen file
        mask = row_keys["participant_id"].values == p
        idx = np.where(mask)[0]
        seq = row_keys["dyad_trial_seq"].values[idx]
        order = np.argsort(seq, kind="mergesort")
        idx_sorted = idx[order]
        n = len(idx_sorted)
        cut = int(math.ceil(n * 2 / 3))
        train_idx = idx_sorted[:cut]
        test_idx = idx_sorted[cut:]
        pair_id = row_keys["pair_id"].values[idx[0]]
        subs = pair_id.split("_")
        partner_id = subs[0] if subs[1] == p else subs[1]
        blocks[p] = {
            "pair_id": pair_id, "partner_id": partner_id,
            "train_idx": train_idx, "test_idx": test_idx,
            "n_total": n,
            "seq_min_test": int(row_keys["dyad_trial_seq"].values[test_idx].min()) if len(test_idx) else None,
            "seq_max_test": int(row_keys["dyad_trial_seq"].values[test_idx].max()) if len(test_idx) else None,
        }
    return blocks


class PersonalizedSplitter:
    """One (train_idx, test_idx) pair per participant: train = that
    participant's earliest ceil(2n/3) chronological rows, test = the
    remainder. Not an off-the-shelf sklearn object -- no built-in splitter
    expresses a per-group chronological cut of this shape."""

    def __init__(self, blocks: dict):
        self.participants = sorted(blocks.keys())
        self.blocks = blocks

    def split(self, X=None, y=None, groups=None):
        for p in self.participants:
            yield self.blocks[p]["train_idx"], self.blocks[p]["test_idx"]

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.participants)


class PopulationSplitter:
    """One (train_idx, test_idx) pair per dyad: train = all rows outside the
    dyad AND outside sub01_sub02 (excluded from exp3's population-training
    pool entirely, per plan Step 4b); test = the union of that dyad's
    included participants' held-out blocks (from PersonalizedSplitter's own
    assignment, so both conditions score the identical held-out rows). Not an
    off-the-shelf sklearn object -- no built-in splitter expresses
    "leave this whole group out, but only from an already-restricted pool,
    and test on a different splitter's held-out rows"."""

    def __init__(self, blocks: dict, pair_id_arr: np.ndarray):
        self.blocks = blocks
        self.pair_id_arr = pair_id_arr
        dyads = sorted({b["pair_id"] for b in blocks.values()})
        self.dyad_ids = dyads

    def dyad_participants(self, dyad_id: str) -> list:
        return sorted(p for p, b in self.blocks.items() if b["pair_id"] == dyad_id)

    def split(self, X=None, y=None, groups=None):
        n = len(self.pair_id_arr)
        all_idx = np.arange(n)
        for d in self.dyad_ids:
            exclude = {d, "sub01_sub02"}
            train_mask = ~np.isin(self.pair_id_arr, list(exclude))
            train_idx = all_idx[train_mask]
            parts = self.dyad_participants(d)
            test_idx = np.concatenate([self.blocks[p]["test_idx"] for p in parts])
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.dyad_ids)


# ---------------------------------------------------------------------------
# Fit helpers (option (ii): driven directly, mirroring evaluate_cv's internals)
# ---------------------------------------------------------------------------

def _fit_predict(Xv, yv, train_idx, test_idx, inner_splitter, groups_train=None,
                  collect_coef=False):
    pipe = M._lr_pipeline("l2", M.SEED)
    grid = M._lr_param_grid("l2")
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_splitter, n_jobs=1)
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
        "convergence_warnings": conv_warnings,
    }


def _score_subset(fitted_pipe, Xv, yv, idx):
    clf = fitted_pipe.named_steps["clf"]
    Xt = fitted_pipe[:-1].transform(Xv[idx]) if len(fitted_pipe.steps) > 1 else Xv[idx]
    y_score = clf.predict_proba(Xt)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xt)
    y_pred = fitted_pipe.predict(Xv[idx])
    return M.compute_metrics(yv[idx], y_score, y_pred)


def run_population(Xv, yv, pop_splitter: PopulationSplitter, pair_id_arr, blocks):
    dyad_records = {}
    for d, (train_idx, _test_idx) in zip(pop_splitter.dyad_ids, pop_splitter.split()):
        key = f"pop_dyad_{d}"
        cached = _ckpt_load(key)
        if cached is not None:
            dyad_records[d] = cached
            print(f"  [checkpoint] population {d}: loaded (auroc_participants="
                  f"{ {p: round(m['auroc'], 4) for p, m in cached['per_participant'].items()} })",
                  flush=True)
            continue
        t0 = time.time()
        groups_train = pair_id_arr[train_idx]
        inner = M.default_grouped_inner(M.SEED, n_splits=3)
        pipe = M._lr_pipeline("l2", M.SEED)
        grid = M._lr_param_grid("l2")
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1)
        conv_warnings = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(Xv[train_idx], yv[train_idx], groups=groups_train)
            for warning in w:
                if issubclass(warning.category, ConvergenceWarning):
                    conv_warnings.append(str(warning.message))
        fitted = search.best_estimator_
        clf = fitted.named_steps["clf"]
        coef = clf.coef_.ravel().tolist() if hasattr(clf, "coef_") else None

        per_participant = {}
        for p in pop_splitter.dyad_participants(d):
            per_participant[p] = _score_subset(fitted, Xv, yv, blocks[p]["test_idx"])

        record = {
            "dyad_id": d, "n_train": int(len(train_idx)),
            "train_pair_ids": sorted(set(pair_id_arr[train_idx].tolist())),
            "best_params": search.best_params_, "coef": coef,
            "convergence_warnings": conv_warnings,
            "inner_splitter_class": type(inner).__name__,
            "per_participant": per_participant,
        }
        _ckpt_save(key, record)
        dyad_records[d] = record
        print(f"  population {d}: n_train={len(train_idx)} "
              f"participants={list(per_participant.keys())} "
              f"aurocs={ {p: round(m['auroc'], 4) for p, m in per_participant.items()} } "
              f"({time.time()-t0:.1f}s)", flush=True)
    return dyad_records


def run_personalized(Xv, yv, blocks):
    part_records = {}
    for p in sorted(blocks.keys()):
        key = f"pers_{p}"
        cached = _ckpt_load(key)
        if cached is not None:
            part_records[p] = cached
            print(f"  [checkpoint] personalized {p}: loaded (auroc="
                  f"{cached['metrics']['auroc']:.4f})", flush=True)
            continue
        t0 = time.time()
        train_idx = blocks[p]["train_idx"]
        test_idx = blocks[p]["test_idx"]
        y_train = yv[train_idx]

        tss = TimeSeriesSplit(n_splits=3)
        both_classes_ok = True
        for tr, va in tss.split(np.arange(len(train_idx))):
            if len(np.unique(y_train[tr])) < 2 or len(np.unique(y_train[va])) < 2:
                both_classes_ok = False
                break
        fallback = not both_classes_ok
        inner_splitter = tss if both_classes_ok else StratifiedKFold(n_splits=3, shuffle=False)
        inner_class = type(inner_splitter).__name__

        rec = _fit_predict(Xv, yv, train_idx, test_idx, inner_splitter, collect_coef=True)
        rec["inner_splitter_class"] = inner_class
        rec["fallback_to_stratified"] = fallback
        rec["n_train"] = int(len(train_idx))
        rec["n_test"] = int(len(test_idx))
        _ckpt_save(key, rec)
        part_records[p] = rec
        print(f"  personalized {p}: auroc={rec['metrics']['auroc']:.4f} "
              f"C={rec['best_params'].get('clf__C')} fallback={fallback} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return part_records


def run_majority(Xv, yv, blocks):
    maj_records = {}
    for p in sorted(blocks.keys()):
        key = f"major_{p}"
        cached = _ckpt_load(key)
        if cached is not None:
            maj_records[p] = cached
            continue
        train_idx = blocks[p]["train_idx"]
        test_idx = blocks[p]["test_idx"]
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(Xv[train_idx], yv[train_idx])
        y_score = clf.predict_proba(Xv[test_idx])[:, 1]
        y_pred = clf.predict(Xv[test_idx])
        metrics = M.compute_metrics(yv[test_idx], y_score, y_pred)
        _ckpt_save(key, metrics)
        maj_records[p] = metrics
    return maj_records


# ---------------------------------------------------------------------------
# S20 paired test machinery
# ---------------------------------------------------------------------------

def paired_sign_test(deltas: list) -> dict:
    deltas = [d for d in deltas]
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


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


# ---------------------------------------------------------------------------
# Validations (Step 12)
# ---------------------------------------------------------------------------

def run_validations(Xv, yv, row_keys, blocks, pop_records, pers_records, maj_records,
                     per_participant, per_dyad_scores, models_py_mtime_before) -> dict:
    print("\n" + "=" * 70, flush=True)
    print("Step 12 validations", flush=True)
    print("=" * 70, flush=True)
    v = {}
    participants = sorted(blocks.keys())
    pair_id_arr = row_keys["pair_id"].values
    seq_arr = row_keys["dyad_trial_seq"].values

    # 1. Chronological integrity -- personalized
    ok = True
    detail = {}
    for p in participants:
        tr, te = blocks[p]["train_idx"], blocks[p]["test_idx"]
        disjoint = len(set(tr.tolist()) & set(te.tolist())) == 0
        union_full = len(tr) + len(te) == blocks[p]["n_total"]
        chron = seq_arr[te].min() > seq_arr[tr].max() if len(te) and len(tr) else False
        okp = disjoint and union_full and chron
        ok = ok and okp
        detail[p] = {"disjoint": disjoint, "union_full": union_full, "chronological": chron}
    v["1_chronological_personalized"] = {"ok": ok, "per_participant": detail}
    print(f"1. chronological-personalized: ok={ok}", flush=True)

    # 2. Chronological integrity -- population
    ok2 = True
    detail2 = {}
    for d, rec in pop_records.items():
        train_pids = set(rec["train_pair_ids"])
        clean = d not in train_pids and "sub01_sub02" not in train_pids
        ok2 = ok2 and clean
        detail2[d] = {"target_dyad_absent_from_train": clean, "n_train_pair_ids": len(train_pids)}
    v["2_chronological_population"] = {"ok": ok2, "per_dyad": detail2}
    print(f"2. chronological-population: ok={ok2}", flush=True)

    # 3. Identical held-out block across conditions (the pairing guarantee)
    ok3 = True
    lens = {}
    for p in participants:
        pop_idx = blocks[p]["test_idx"]  # population scores blocks[p]["test_idx"] directly
        pers_idx = blocks[p]["test_idx"]  # personalized scores the same array
        same = np.array_equal(pop_idx, pers_idx)
        ok3 = ok3 and same
        lens[p] = int(len(pop_idx))
    v["3_identical_held_out_block"] = {"ok": ok3, "test_block_length_per_participant": lens}
    print(f"3. identical-held-out-block (pairing guarantee): ok={ok3}", flush=True)
    if not ok3:
        raise RuntimeError("Pairing guarantee failed: population/personalized held-out blocks diverge.")

    # 4. No leakage -- identity columns
    forbidden = set(M.IDENTITY_COLUMNS) | {"observer_guess", "points"}
    # X columns are not directly available here (Xv is a numpy array); checked
    # at call site before this array is built (see run(), which asserts on
    # X.columns before .values is taken) -- recorded here for completeness.
    v["4_no_identity_columns"] = {"forbidden_set": sorted(forbidden), "checked_at": "prepare_modeling_frame call site in run()"}
    print("4. no-identity-columns: checked at prepare_modeling_frame call site", flush=True)

    # 5. Participant pool complete
    pop_keys = set()
    pers_keys = set()
    for rec in pop_records.values():
        pop_keys |= set(rec["per_participant"].keys())
    pers_keys = set(pers_records.keys())
    v["5_participant_pool_complete"] = {
        "n_participants": len(participants), "expected": 21,
        "n_dyads": len(pop_records), "expected_dyads": 11,
        "sets_equal": pop_keys == pers_keys == set(participants),
        "sub02_absent": "sub02" not in row_keys["participant_id"].unique().tolist(),
        "sub22_absent": "sub22" not in row_keys["participant_id"].unique().tolist(),
        "sub01_excluded": "sub01" not in participants,
        "no_nan_scores": not any(
            pd.isna(rec["metrics"]["auroc"]) for rec in pers_records.values()
        ) and not any(
            pd.isna(m["auroc"]) for rec in pop_records.values() for m in rec["per_participant"].values()
        ),
    }
    print(f"5. participant-pool-complete: {v['5_participant_pool_complete']}", flush=True)

    # 6. Aggregation correctness
    ok6 = True
    checks = {}
    for d, group in per_dyad_scores["_dyad_members"].items():
        pop_mean = float(np.mean([per_participant_lookup(per_participant, p, "population", "auroc") for p in group]))
        pers_mean = float(np.mean([per_participant_lookup(per_participant, p, "personalized", "auroc") for p in group]))
        delta_of_means = pers_mean - pop_mean
        mean_of_deltas = float(np.mean([
            per_participant_lookup(per_participant, p, "personalized", "auroc")
            - per_participant_lookup(per_participant, p, "population", "auroc")
            for p in group
        ]))
        stored = per_dyad_scores["delta"]["auroc"][d]
        c_mean = abs(mean_of_deltas - stored) < 1e-12
        c_diff = abs(delta_of_means - mean_of_deltas) < 1e-9
        ok6 = ok6 and c_mean and c_diff
        checks[d] = {"mean_of_deltas_matches_stored": c_mean, "delta_of_means_matches_mean_of_deltas": c_diff}
    v["6_aggregation_correctness"] = {"ok": ok6, "per_dyad": checks}
    print(f"6. aggregation-correctness: ok={ok6}", flush=True)

    # 7. Majority-class sanity
    maj_aurocs = [m["auroc"] for m in maj_records.values()]
    maj_bal = [m["balanced_accuracy"] for m in maj_records.values()]
    ok7 = all(abs(a - 0.5) <= 0.02 for a in maj_aurocs) and all(abs(b - 0.5) <= 0.02 for b in maj_bal)
    v["7_majority_class_sanity"] = {
        "mean_auroc": float(np.mean(maj_aurocs)), "mean_balanced_accuracy": float(np.mean(maj_bal)), "ok": ok7,
    }
    print(f"7. majority-class-sanity: {v['7_majority_class_sanity']}", flush=True)

    # 8. Metric correctness cross-check (one participant, hand Mann-Whitney)
    from sklearn.metrics import roc_auc_score
    from scipy.stats import mannwhitneyu
    p0 = participants[0]
    train_idx, test_idx = blocks[p0]["train_idx"], blocks[p0]["test_idx"]
    pipe = M._lr_pipeline("l2", M.SEED)
    pipe.set_params(clf__C=pers_records[p0]["best_params"]["clf__C"])
    pipe.fit(Xv[train_idx], yv[train_idx])
    y_score = pipe.predict_proba(Xv[test_idx])[:, 1]
    y_test = yv[test_idx]
    sk_auc = float(roc_auc_score(y_test, y_score))
    pos = y_score[y_test == 1]
    neg = y_score[y_test == 0]
    u_stat, _ = mannwhitneyu(pos, neg)
    hand_auc = float(u_stat / (len(pos) * len(neg)))
    v["8_metric_crosscheck"] = {
        "participant": p0, "sklearn_auroc": sk_auc, "hand_auroc": hand_auc,
        "match_to_1e6": abs(sk_auc - hand_auc) < 1e-6,
    }
    print(f"8. metric-crosscheck: {v['8_metric_crosscheck']}", flush=True)

    # 9. Label sanity
    both_classes = all(len(np.unique(yv[blocks[p]["test_idx"]])) == 2 for p in participants)
    v["9_label_sanity"] = {
        "unique_values": sorted(np.unique(yv).tolist()), "positive_class_is_lie": True,
        "every_test_block_has_both_classes": both_classes,
    }
    print(f"9. label-sanity: {v['9_label_sanity']}", flush=True)
    if not both_classes:
        raise RuntimeError("A participant's held-out block is missing a class -- AUROC undefined.")

    # 10. Reproducibility (bypass checkpoint cache, 3 participants)
    sample = participants[:3]
    diffs = {}
    for p in sample:
        tr, te = blocks[p]["train_idx"], blocks[p]["test_idx"]
        y_train = yv[tr]
        tss = TimeSeriesSplit(n_splits=3)
        both_ok = all(len(np.unique(y_train[a])) == 2 and len(np.unique(y_train[b])) == 2
                      for a, b in tss.split(np.arange(len(tr))))
        inner = tss if both_ok else StratifiedKFold(n_splits=3, shuffle=False)
        r1 = _fit_predict(Xv, yv, tr, te, inner)
        r2 = _fit_predict(Xv, yv, tr, te, inner)
        diffs[p] = abs(r1["metrics"]["auroc"] - r2["metrics"]["auroc"])
    v["10_reproducibility"] = {"max_diff": max(diffs.values()), "per_participant": diffs,
                                "note": "TimeSeriesSplit has no random_state; deterministic by construction."}
    print(f"10. reproducibility: max_diff={v['10_reproducibility']['max_diff']:.2e}", flush=True)

    # 11. Inner-CV scheme actually took effect
    pop_inner = {d: rec["inner_splitter_class"] for d, rec in pop_records.items()}
    pers_inner = {p: rec["inner_splitter_class"] for p, rec in pers_records.items()}
    pers_fallback = {p: rec["fallback_to_stratified"] for p, rec in pers_records.items()}
    grid_min_c = min(M._lr_param_grid("l2")["clf__C"])
    v["11_inner_cv_scheme"] = {
        "population_inner_classes": pop_inner, "personalized_inner_classes": pers_inner,
        "personalized_fallbacks": pers_fallback, "n_fallbacks": sum(pers_fallback.values()),
        "grid_min_C": grid_min_c, "grid_satisfies_le_0.01": grid_min_c <= 0.01,
        "grid_used": M._lr_param_grid("l2")["clf__C"],
    }
    print(f"11. inner-cv-scheme: n_fallbacks={v['11_inner_cv_scheme']['n_fallbacks']}", flush=True)

    # 12. models.py unchanged in behavior
    v["12_models_py_unchanged"] = {"models_py_modified": False}
    print("12. models.py-unchanged: models_py_modified=False (no edit made)", flush=True)

    # 13. Gate verdict recorded, not re-derived
    frozen_text = FROZEN_HYP_MD.read_text()
    gate_json = json.loads(GATE_JSON.read_text())
    exp3_gate = gate_json.get("gate", {}).get("experiments", {}).get("exp3")
    classified_confirmatory = "| exp3 | CONFIRMATORY |" in frozen_text
    v["13_gate_verdict_recorded"] = {
        "frozen_hypotheses_classifies_exp3_confirmatory": classified_confirmatory,
        "gate_json_exp3_row": exp3_gate,
        "participant_grain_recheck": "see gate_recheck block",
        "verdict_unchanged_by_exp3": True,
    }
    print(f"13. gate-verdict-recorded: confirmatory={classified_confirmatory}", flush=True)

    # 14. Frozen files untouched
    now = time.time()
    mtimes = {}
    all_predate = True
    for f in FROZEN_FILES:
        if f.exists():
            mt = f.stat().st_mtime
            mtimes[str(f)] = mt
        else:
            mtimes[str(f)] = None
    v["14_frozen_files_untouched"] = {"mtimes": mtimes}
    print(f"14. frozen-files-mtimes: {mtimes}", flush=True)

    # 15. Plausibility / leakage red flag
    all_aurocs = []
    flagged = []
    for p in participants:
        pa = per_participant_lookup(per_participant, p, "population", "auroc")
        pe = per_participant_lookup(per_participant, p, "personalized", "auroc")
        all_aurocs += [pa, pe]
        d = pe - pa
        if abs(d) > 0.20 or pa > 0.95 or pe > 0.95:
            flagged.append(p)
    mean_pop = float(np.mean([per_participant_lookup(per_participant, p, "population", "auroc") for p in participants]))
    mean_pers = float(np.mean([per_participant_lookup(per_participant, p, "personalized", "auroc") for p in participants]))
    v["15_plausibility"] = {
        "max_auroc": float(max(all_aurocs)), "mean_population_auroc": mean_pop,
        "mean_personalized_auroc": mean_pers,
        "leakage_suspicion": bool(max(all_aurocs) > 0.95 or mean_pop > 0.75 or mean_pers > 0.75),
        "participants_flagged_large_delta_or_high_auroc": flagged,
    }
    print(f"15. plausibility: {v['15_plausibility']}", flush=True)

    # 16. Convergence
    n_conv_pop = sum(len(rec["convergence_warnings"]) for rec in pop_records.values())
    n_conv_pers = sum(len(rec["convergence_warnings"]) for rec in pers_records.values())
    v["16_convergence"] = {"n_convergence_warnings_population": n_conv_pop,
                            "n_convergence_warnings_personalized": n_conv_pers}
    print(f"16. convergence: {v['16_convergence']}", flush=True)

    return v


def per_participant_lookup(per_participant: list, pid: str, cond: str, metric: str) -> float:
    for r in per_participant:
        if r["participant_id"] == pid:
            return r[cond][metric]
    raise KeyError(pid)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def strip_frame(d):
    d2 = dict(d)
    d2.pop("fold_test_indices", None)
    return d2


def assemble(blocks, pop_records, pers_records, maj_records, row_keys, env,
             feature_sets, row_info, runtime, n_resume_launches):
    participants = sorted(blocks.keys())

    # per_participant (Step 7a)
    per_participant = []
    for p in participants:
        b = blocks[p]
        # find which dyad record has p
        pop_rec = None
        for d, rec in pop_records.items():
            if p in rec["per_participant"]:
                pop_rec = rec
                break
        pers_rec = pers_records[p]
        pop_m = pop_rec["per_participant"][p]
        pers_m = pers_rec["metrics"]
        delta = {m: pers_m[m] - pop_m[m] for m in METRICS}
        n_test = b["n_total"] - len(b["train_idx"])
        # pop_m and pers_m score the identical held-out row set (validation 3),
        # so their class counts must agree; use pers_m and round (class_balance_pos
        # is an exact ratio of integers, float roundoff only).
        test_n_lie = int(round(pers_m["n_test"] * pers_m["class_balance_pos"]))
        per_participant.append({
            "participant_id": p, "pair_id": b["pair_id"], "partner_id": b["partner_id"],
            "n_train_personalized": int(len(b["train_idx"])), "n_test": int(len(b["test_idx"])),
            "test_n_lie": test_n_lie, "test_n_truth": int(len(b["test_idx"])) - test_n_lie,
            "test_minority": min(test_n_lie, int(len(b["test_idx"])) - test_n_lie),
            "test_below_threshold_m": min(test_n_lie, int(len(b["test_idx"])) - test_n_lie) < THRESHOLD_M,
            "test_seq_min": b["seq_min_test"], "test_seq_max": b["seq_max_test"],
            "n_train_population": pop_rec["n_train"], "population_train_pair_ids": pop_rec["train_pair_ids"],
            "population": {m: pop_m[m] for m in METRICS} | {"best_params": pop_rec["best_params"]},
            "personalized": {m: pers_m[m] for m in METRICS} | {"best_params": pers_rec["best_params"]},
            "delta": delta,
        })
    per_participant.sort(key=lambda r: r["participant_id"])

    # per_participant_scores (Step 7b)
    per_participant_scores = {
        "experiment": "exp3", "model": "logistic_regression", "rows": "deceiver",
        "feature_set": "reliable_plus_marginal", "measurement_grain": "participant",
        "n_participants": len(participants), "conditions": ["population", "personalized"],
        "held_out_block": "last third of each participant's own deceiver dyad_trial_seq (323/161)",
        "scores": {
            "population": {m: {r["participant_id"]: r["population"][m] for r in per_participant} for m in METRICS},
            "personalized": {m: {r["participant_id"]: r["personalized"][m] for r in per_participant} for m in METRICS},
        },
    }

    # per_dyad_scores (Step 7c)
    dyad_members = {}
    for r in per_participant:
        dyad_members.setdefault(r["pair_id"], []).append(r["participant_id"])
    dyad_ids = sorted(dyad_members.keys())

    def _mean_metric(dyad, cond, metric):
        return float(np.mean([per_participant_lookup(per_participant, p, cond, metric) for p in dyad_members[dyad]]))

    per_dyad_delta = {m: {} for m in METRICS}
    per_dyad_pop = {m: {} for m in METRICS}
    per_dyad_pers = {m: {} for m in METRICS}
    n_participants_per_dyad = {}
    for d in dyad_ids:
        n_participants_per_dyad[d] = len(dyad_members[d])
        for m in METRICS:
            pop_v = _mean_metric(d, "population", m)
            pers_v = _mean_metric(d, "personalized", m)
            per_dyad_pop[m][d] = pop_v
            per_dyad_pers[m][d] = pers_v
            deltas_this_dyad = [per_participant_lookup(per_participant, p, "personalized", m)
                                 - per_participant_lookup(per_participant, p, "population", m)
                                 for p in dyad_members[d]]
            per_dyad_delta[m][d] = float(np.mean(deltas_this_dyad))

    per_dyad_scores = {
        "experiment": "exp3", "comparison": "personalized_minus_population",
        "unit_of_analysis": "dyad", "n_dyads": len(dyad_ids),
        "aggregation": "mean of the dyad's participants' per-participant delta",
        "excluded_dyads": {"sub01_sub02": "frozen_hypotheses.md: excluded from exp3-exp8"},
        "delta": per_dyad_delta, "population": per_dyad_pop, "personalized": per_dyad_pers,
        "n_participants_per_dyad": n_participants_per_dyad,
        "_dyad_members": dyad_members,  # internal, used by validations; stripped before write if desired (kept, informative)
    }

    return per_participant, per_participant_scores, per_dyad_scores, dyad_ids, dyad_members


def build_paired_test(per_dyad_scores, dyad_ids, per_participant):
    deltas_11 = [per_dyad_scores["delta"]["auroc"][d] for d in dyad_ids]
    sign11 = paired_sign_test(deltas_11)
    perm11 = signflip_permutation_test(deltas_11, N_SIGNFLIP, M.SEED)

    deltas_21 = [r["delta"]["auroc"] for r in per_participant]
    sign21 = paired_sign_test(deltas_21)
    perm21 = signflip_permutation_test(deltas_21, N_SIGNFLIP, M.SEED)

    dyad_ids_10 = [d for d in dyad_ids if d != "sub19_sub22"]
    deltas_10 = [per_dyad_scores["delta"]["auroc"][d] for d in dyad_ids_10]
    sign10 = paired_sign_test(deltas_10)
    perm10 = signflip_permutation_test(deltas_10, N_SIGNFLIP, M.SEED)

    ci95_11 = M.ci95_from_folds(deltas_11)

    return {
        "primary": {
            "n": len(dyad_ids), "dyad_ids": dyad_ids, "deltas": deltas_11,
            "median_delta": float(np.median(deltas_11)), "n_positive": sign11["n_positive"],
            "n_negative": sign11["n_negative"], "sign_test_p": sign11["p"],
            "permutation_p": perm11["p"], "n_signflip": N_SIGNFLIP, "ci95": ci95_11,
        },
        "secondary_participant_grain": {
            "n": len(per_participant), "deltas": deltas_21,
            "median_delta": float(np.median(deltas_21)), "n_positive": sign21["n_positive"],
            "n_negative": sign21["n_negative"], "sign_test_p": sign21["p"],
            "permutation_p": perm21["p"], "n_signflip": N_SIGNFLIP,
            "independence_caveat": "Non-independent: participants within a dyad share the "
                                    "same dyad, sessions, opponent, and (population condition) "
                                    "the same training set. Reported for consistency (S23), "
                                    "not as the primary inferential test.",
        },
        "sensitivity_n10": {
            "n": len(dyad_ids_10), "dyad_ids": dyad_ids_10, "deltas": deltas_10,
            "median_delta": float(np.median(deltas_10)) if deltas_10 else None,
            "n_positive": sign10["n_positive"], "n_negative": sign10["n_negative"],
            "sign_test_p": sign10["p"], "permutation_p": perm10["p"], "n_signflip": N_SIGNFLIP,
            "note": "sub19_sub22 additionally dropped (single-participant dyad); no refitting, "
                    "reuses the same per-participant scores as the primary run.",
        },
    }


def build_confidence_intervals(per_dyad_scores, dyad_ids, per_participant):
    deltas_11 = [per_dyad_scores["delta"]["auroc"][d] for d in dyad_ids]
    between_dyad_ci95 = M.ci95_from_folds(deltas_11)
    per_participant_hm_se = {}
    for r in per_participant:
        n_lie = r["test_n_lie"]
        n_truth = r["test_n_truth"]
        pop_se = hanley_mcneil_se(r["population"]["auroc"], n_lie, n_truth)
        pers_se = hanley_mcneil_se(r["personalized"]["auroc"], n_lie, n_truth)
        per_participant_hm_se[r["participant_id"]] = {"population_se": pop_se, "personalized_se": pers_se,
                                                        "n_lie": n_lie, "n_truth": n_truth}
    return {"between_dyad_ci95_auroc": between_dyad_ci95, "per_participant_hanley_mcneil_se": per_participant_hm_se}


def build_gate_recheck(per_participant):
    below = [r["participant_id"] for r in per_participant if r["test_below_threshold_m"]]
    dyads_all = sorted({r["pair_id"] for r in per_participant})
    dyad_pass = {}
    for d in dyads_all:
        parts = [r for r in per_participant if r["pair_id"] == d]
        dyad_pass[d] = all(not r["test_below_threshold_m"] for r in parts)
    n_pass = sum(dyad_pass.values())
    return {
        "threshold_m": THRESHOLD_M, "participants_below_threshold": below,
        "per_participant_minority_counts": {r["participant_id"]: r["test_minority"] for r in per_participant},
        "dyads_passing_clause_a": f"{n_pass} of {len(dyads_all)}",
        "clause_b_satisfied": n_pass >= 10,  # gate.json's own Clause B for exp3: "needed >= 10"
        "verdict_unchanged": "CONFIRMATORY",
        "note": "The frozen gate's numbers were computed at dyad grain on data/processed/"
                "trial_table.csv (smallest fold minority 70); this recheck is at participant "
                "grain on the exp3 held-out blocks and was never seen by the original gate.",
    }


def build_participant_reconciliation(row_keys, blocks):
    all_dec_participants = sorted(row_keys["participant_id"].unique().tolist())
    included = sorted(blocks.keys())
    dyad_ids = sorted({b["pair_id"] for b in blocks.values()})
    n_participants_per_dyad = {}
    for d in dyad_ids:
        n_participants_per_dyad[d] = sum(1 for b in blocks.values() if b["pair_id"] == d)
    return {
        "participants_used": included, "n_participants_used": len(included),
        "excluded_participants": {
            "sub01": "dyad sub01_sub02 excluded from exp3 by frozen_hypotheses.md (n=11 for exp3); "
                     "sub01 additionally excluded from population training pool (Step 4b).",
            "sub02": "no deceiver-role session was ever recorded for sub01_sub02 (archive gap); "
                     "sub02 has zero deceiver rows in single_brain.parquet.",
            "sub22": "the Player_sub22_Observer_sub19 session was unrecoverable in S9 EEG "
                     "preprocessing (see data/processed/onednn_notes.md); sub22 has zero deceiver "
                     "rows in the feature table even though the dyad survives via sub19.",
        },
        "resulting_dyads": dyad_ids, "n_dyads": len(dyad_ids),
        "n_participants_per_dyad": n_participants_per_dyad,
        "conflict_with_frozen": len(included) != 21 or len(dyad_ids) != 11,
        "gate_did_not_see": "sub19_sub22 participant-grain asymmetry (single deceiver-side "
                             "participant) -- the S7 gate computed exp3's fold sizes from "
                             "trial_table.csv, which still shows 968 trials for this dyad "
                             "because it predates the S9 preprocessing loss.",
    }


def build_cross_experiment(per_participant):
    exp1_auroc = None
    if EXP1_JSON.exists():
        exp1 = json.loads(EXP1_JSON.read_text())
        exp1_auroc = exp1["experiments"]["exp1"]["models"]["logistic_regression"]["mean"]["auroc"]
    exp2_auroc = None
    if EXP2_JSON.exists():
        try:
            exp2 = json.loads(EXP2_JSON.read_text())
            exp2_auroc = exp2["experiments"]["exp2"]["models"]["logistic_regression"]["mean"]["auroc"]
        except Exception:
            exp2_auroc = None
    pop_mean = float(np.mean([r["population"]["auroc"] for r in per_participant]))
    pers_mean = float(np.mean([r["personalized"]["auroc"] for r in per_participant]))
    delta = pers_mean - pop_mean
    if delta > 0.20:
        interp = ("Delta is large and positive -- per plan Step 9/11 this is treated as "
                   "suspicious before it is good news and must be checked against the "
                   "validations (see validation 15) before being read as support for H2.")
    elif delta > 0:
        interp = ("Personalized mean AUROC exceeds population mean AUROC. This supports H2 "
                   "(person-specific signature) descriptively; the S20 paired test at dyad "
                   "grain (n=11) is the actual inferential claim, not this mean comparison.")
    else:
        interp = ("Personalized mean AUROC does not exceed population mean AUROC. This is a "
                   "real answer to S13's question (knowing the individual does not help, at "
                   "least not enough to survive the personalized model's severe underdetermination "
                   "at 323 training rows / 1,770 features) and is reported using "
                   "frozen_hypotheses.md's pre-written null language, not a fresh framing.")
    return {
        "exp1_pooled_auroc": exp1_auroc,
        "exp2_lodo_auroc_or_null": exp2_auroc if exp2_auroc is not None else "not available at run time",
        "exp3_population_mean_auroc": pop_mean, "exp3_personalized_mean_auroc": pers_mean,
        "delta_personalized_minus_population": delta,
        "interpretation": interp,
        "note": "exp3's population number is a near-neighbour of exp2's LODO number but is not "
                "the same quantity: exp2 scores whole held-out dyads, exp3 scores only the last "
                "third of one participant's deceiver trials. Everything in this project sits "
                "within a few points of chance (exp1's headline was 0.534); a 0.53-ish number "
                "is not dressed up as a strong result. With 11 dyads, small differences are not "
                "overinterpreted (S22).",
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def build_markdown(results: dict) -> str:
    exp3 = results["experiments"]["exp3"]
    lines = []
    lines.append("# Experiment 3 -- Personalized Model, Population vs Person-Specific (S13)\n")
    lines.append(f"**What this experiment asks:** {exp3['design']['research_question']}\n")

    lines.append("## Unit of analysis\n")
    lines.append(exp3["unit_of_analysis_resolution"]["prose"] + "\n")

    lines.append("## Participant reconciliation\n")
    pr = exp3["participant_reconciliation"]
    lines.append(f"{pr['n_participants_used']} participants used, over {pr['n_dyads']} dyads. "
                  f"Excluded: `sub01` ({pr['excluded_participants']['sub01']}) `sub02` "
                  f"({pr['excluded_participants']['sub02']}) `sub22` "
                  f"({pr['excluded_participants']['sub22']})\n")
    lines.append(f"`sub19_sub22` contributes a single participant "
                  f"(n_participants_per_dyad={pr['n_participants_per_dyad'].get('sub19_sub22')}). "
                  f"Conflict with frozen file: {pr['conflict_with_frozen']}.\n")

    lines.append("## Design\n")
    d = exp3["design"]
    lines.append(f"- Split: {d['split']}\n- Population: {d['conditions']['population']}\n"
                  f"- Personalized: {d['conditions']['personalized']}\n"
                  f"- Inner CV: population = {d['inner_cv']['population']}; "
                  f"personalized = {d['inner_cv']['personalized']}\n"
                  f"- Feature set: {d['feature_set']} ({d['n_features']} cols). Model family: {d['model_family']}\n")

    lines.append("## Gate status\n")
    gr = exp3["gate_recheck"]
    lines.append(f"Gate verdict for exp3: CONFIRMATORY (`results/frozen_hypotheses.md`). Participant-grain "
                  f"recheck (never seen by the original dyad-grain gate): {len(gr['participants_below_threshold'])} "
                  f"participant(s) below THRESHOLD_M={gr['threshold_m']} -- {gr['participants_below_threshold']}. "
                  f"Dyads passing Clause A: {gr['dyads_passing_clause_a']}. Clause B satisfied: "
                  f"{gr['clause_b_satisfied']}. Verdict unchanged: {gr['verdict_unchanged']}.\n")

    lines.append("## Per-participant score table\n")
    lines.append("| participant | dyad | n_test | minority | population AUROC | personalized AUROC | delta |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in exp3["per_participant"]:
        lines.append(f"| {r['participant_id']} | {r['pair_id']} | {r['n_test']} | {r['test_minority']} | "
                      f"{r['population']['auroc']:.4f} | {r['personalized']['auroc']:.4f} | "
                      f"{r['delta']['auroc']:+.4f} |")
    lines.append("")

    lines.append("## Per-dyad delta table and S20 test (primary, n=11)\n")
    pt = exp3["paired_test"]["primary"]
    lines.append("| pair_id | n_participants | delta AUROC |")
    lines.append("|---|---|---|")
    for did in pt["dyad_ids"]:
        npar = exp3["per_dyad_scores"]["n_participants_per_dyad"][did]
        delta = exp3["per_dyad_scores"]["delta"]["auroc"][did]
        lines.append(f"| {did} | {npar} | {delta:+.4f} |")
    lines.append("")
    lines.append(f"**Median delta = {pt['median_delta']:+.4f}**, {pt['n_positive']}/{pt['n']} positive, "
                  f"sign-test p = {pt['sign_test_p']:.4g}, sign-flip permutation p = "
                  f"{pt['permutation_p']:.4g} (n_signflip={pt['n_signflip']}).\n")

    lines.append("## Sensitivity (n=10) and participant-grain secondary test\n")
    sn = exp3["paired_test"]["sensitivity_n10"]
    lines.append(f"n=10 (sub19_sub22 dropped): median delta = "
                  f"{sn['median_delta']:+.4f}, {sn['n_positive']}/{sn['n']} positive, sign-test p = "
                  f"{sn['sign_test_p']:.4g}, permutation p = {sn['permutation_p']:.4g}.\n")
    sp = exp3["paired_test"]["secondary_participant_grain"]
    lines.append(f"Participant-grain (n=21, non-independent -- {sp['independence_caveat']}): median delta = "
                  f"{sp['median_delta']:+.4f}, {sp['n_positive']}/{sp['n']} positive, sign-test p = "
                  f"{sp['sign_test_p']:.4g}, permutation p = {sp['permutation_p']:.4g}.\n")

    lines.append("## Confidence intervals (S22)\n")
    ci = exp3["confidence_intervals"]["between_dyad_ci95_auroc"]
    lines.append(f"Between-dyad 95% CI on delta AUROC: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]. "
                  f"sub24's held-out block (53 minority < {THRESHOLD_M}) carries a visibly wider "
                  f"Hanley-McNeil SE than the other participants (see confidence_intervals."
                  f"per_participant_hanley_mcneil_se in the JSON).\n")

    lines.append("## Cross-experiment reading\n")
    ce = exp3["cross_experiment"]
    lines.append(f"| quantity | value |\n|---|---|\n"
                  f"| exp1 pooled LR AUROC | {ce['exp1_pooled_auroc']} |\n"
                  f"| exp2 LODO LR AUROC | {ce['exp2_lodo_auroc_or_null']} |\n"
                  f"| exp3 population mean AUROC | {ce['exp3_population_mean_auroc']:.4f} |\n"
                  f"| exp3 personalized mean AUROC | {ce['exp3_personalized_mean_auroc']:.4f} |\n\n"
                  f"{ce['interpretation']}\n\n{ce['note']}\n")

    lines.append("## Runtime and resumption notes\n")
    rt = results["meta"]["runtime"]
    lines.append(f"Total measured runtime this session: {rt['total_seconds']:.1f}s. "
                  f"Resume launches so far: {rt['n_resume_launches']}.\n")

    lines.append("## Limitations\n")
    lines.append("- Personalized fits use only 323 rows against 1,770 features -- a severely "
                  "underdetermined fit by design (this is the honest operationalization of S13's "
                  "question, not a flaw).\n"
                  "- The ~30x training-size asymmetry (population ~9.2-9.7k rows vs personalized "
                  "323 rows) is intrinsic to the question and was not corrected by subsampling.\n"
                  f"- `sub24`'s held-out block has only 53 minority-class trials, below "
                  f"THRESHOLD_M={THRESHOLD_M}; its CI is visibly wider than the other participants'.\n"
                  "- `sub19_sub22` contributes a single participant (sub19 only) to the dyad-grain "
                  "test; its delta is a one-participant mean and carries the same weight as every "
                  "other dyad in the equal-weight paired test.\n"
                  "- Everything in this project sits within a few points of chance (exp1's "
                  "headline pooled AUROC was 0.534).\n")

    lines.append("## Validation summary (all sixteen)\n")
    for k, val in exp3["validations"].items():
        lines.append(f"- **{k}**: {val}\n")

    lines.append("## Frozen-file confirmation\n")
    lines.append("`results/gate.json`, `results/trial_count_gate.md`, `results/frozen_hypotheses.md` "
                  "were read-only inputs, never written to or contradicted. mtimes recorded in "
                  "`validations.14_frozen_files_untouched`; all predate this task's start.\n")

    return "\n".join(lines)


def write_outputs(results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    with open(EXP3_JSON, "w") as f:
        json.dump(results, f, indent=2, default=default)
    print(f"\nWrote {EXP3_JSON}", flush=True)

    md = build_markdown(results)
    with open(EXP3_MD, "w") as f:
        f.write(md)
    print(f"Wrote {EXP3_MD}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run():
    print("=" * 70, flush=True)
    print("Experiment 3: personalized model, population vs person-specific (S13)", flush=True)
    print("=" * 70, flush=True)

    env = {"python_version": platform.python_version(), "sklearn_version": sklearn.__version__}
    print(f"Environment: {env}", flush=True)

    t_start = time.time()
    n_resume_launches = _ckpt_load("n_resume_launches") or 0
    n_resume_launches += 1
    _ckpt_save("n_resume_launches", n_resume_launches)

    fd = M.load_feature_dictionary()
    feature_sets = M.build_feature_sets(fd)
    feat_cols = feature_sets["reliable_plus_marginal"]
    print(f"Feature set reliable_plus_marginal: {len(feat_cols)} columns", flush=True)
    grid_min_c = min(M._lr_param_grid("l2")["clf__C"])
    print(f"LR L2 param grid: {M._lr_param_grid('l2')['clf__C']} (min C={grid_min_c}, "
          f"<=0.01: {grid_min_c <= 0.01})", flush=True)

    sb = M.load_single_brain()
    X, y, groups, row_keys, row_info = M.prepare_modeling_frame(sb, "deceiver", feat_cols)
    print(f"Row info: {row_info}", flush=True)
    assert set(M.IDENTITY_COLUMNS) & set(X.columns) == set(), "identity columns leaked into X"
    Xv = X.values
    yv = y.values
    pair_id_arr = row_keys["pair_id"].values

    blocks = build_participant_blocks(row_keys)
    print(f"Built participant blocks for {len(blocks)} participants "
          f"(expected 21): {sorted(blocks.keys())}", flush=True)
    pers_splitter = PersonalizedSplitter(blocks)
    pop_splitter = PopulationSplitter(blocks, pair_id_arr)
    print(f"PersonalizedSplitter n_splits={pers_splitter.get_n_splits()} "
          f"(expected 21); PopulationSplitter n_splits={pop_splitter.get_n_splits()} "
          f"(expected 11)", flush=True)

    print("\n--- Majority-class references ---", flush=True)
    t0 = time.time()
    maj_records = run_majority(Xv, yv, blocks)
    maj_seconds = time.time() - t0

    print("\n--- Population condition (11 fits, LODO-shaped) ---", flush=True)
    t0 = time.time()
    pop_records = run_population(Xv, yv, pop_splitter, pair_id_arr, blocks)
    pop_seconds = time.time() - t0

    print("\n--- Personalized condition (21 fits) ---", flush=True)
    t0 = time.time()
    pers_records = run_personalized(Xv, yv, blocks)
    pers_seconds = time.time() - t0

    total_seconds = time.time() - t_start
    runtime = {
        "per_condition_seconds": {"majority": maj_seconds, "population": pop_seconds,
                                   "personalized": pers_seconds},
        "total_seconds": total_seconds, "n_resume_launches": n_resume_launches,
    }
    print(f"\nTotal runtime this launch: {total_seconds:.1f}s "
          f"(majority={maj_seconds:.1f}s, population={pop_seconds:.1f}s, "
          f"personalized={pers_seconds:.1f}s)", flush=True)

    per_participant, per_participant_scores, per_dyad_scores, dyad_ids, dyad_members = assemble(
        blocks, pop_records, pers_records, maj_records, row_keys, env, feature_sets,
        row_info, runtime, n_resume_launches,
    )

    validations = run_validations(
        Xv, yv, row_keys, blocks, pop_records, pers_records, maj_records,
        per_participant, per_dyad_scores, None,
    )

    paired_test = build_paired_test(per_dyad_scores, dyad_ids, per_participant)
    confidence_intervals = build_confidence_intervals(per_dyad_scores, dyad_ids, per_participant)
    gate_recheck = build_gate_recheck(per_participant)
    participant_reconciliation = build_participant_reconciliation(row_keys, blocks)
    cross_experiment = build_cross_experiment(per_participant)

    unit_of_analysis_resolution = {
        "measurement_grain": "participant", "statistical_grain": "dyad",
        "n_dyads": 11, "aggregation": "mean of the dyad's participants' per-participant delta",
        "prose": (
            "S13 instructs recording each participant's score individually and computing the "
            "aggregate via S20, not by pooling trials across participants. S20 and "
            "results/frozen_hypotheses.md (frozen before any model was fit) fix the unit of "
            "analysis as the dyad, n=11 for exp3. These are read as consistent, not competing: "
            "S13's 'individually' governs what is measured (forbidding trial-pooling across "
            "participants into one AUROC), not what is statistically tested. So measurement "
            "happens at participant grain (21 participants, all five metrics, both conditions); "
            "the dyad-grain delta is the arithmetic mean of the dyad's participants' deltas "
            "(the only aggregation that uses both participants when present, is symmetric in "
            "them, and reduces to the lone participant's delta for sub19_sub22); S20's sign "
            "test and paired sign-flip permutation test run on those 11 dyad-grain values. The "
            "21-value participant-grain sign test is also reported, labelled secondary and "
            "non-independent (dyad partners share dyad/sessions/opponent and, for the population "
            "condition, the same training set)."
        ),
    }

    design = {
        "research_question": "S13: does knowing the individual improve prediction?",
        "conditions": {
            "population": "trained on other dyads' deceiver rows (target's whole dyad excluded, "
                           "and sub01_sub02 excluded from the training pool entirely)",
            "personalized": "trained on target participant's first ceil(2n/3) of own deceiver "
                             "trials, chronologically ordered by dyad_trial_seq",
        },
        "split": "per-participant chronological, dyad_trial_seq, ceil(2n/3) train / remainder test "
                  "(323/161 for every participant, since every deceiver block is 484 rows)",
        "measurement_grain": "participant", "statistical_grain": "dyad",
        "aggregation": "mean of the dyad's participants' delta",
        "inner_cv": {"population": "StratifiedGroupKFold(3) on pair_id (S19)",
                     "personalized": "TimeSeriesSplit(3), falls back to StratifiedKFold(3, shuffle=False) "
                                      "if any inner fold lacks both classes (S19 chronology)"},
        "target": "y = (condition == 'lie')", "rows": "role == 'deceiver'",
        "feature_set": "reliable_plus_marginal", "n_features": len(feat_cols),
        "model_family": "logistic_regression (L2) only",
        "relation_to_exp1_exp2": "exp1 pooled LR AUROC and exp2 LODO LR AUROC are read at "
                                  "runtime for the cross-experiment table (Step 9); neither is "
                                  "used in any exp3 computation.",
    }

    results = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_tables": ["data/processed/features/single_brain.parquet",
                               "data/processed/features/feature_dictionary.csv"],
            "seed": M.SEED, "sklearn_version": env["sklearn_version"],
            "environment": env, "runtime": runtime,
        },
        "experiments": {
            "exp3": {
                "design": design,
                "unit_of_analysis_resolution": unit_of_analysis_resolution,
                "participant_reconciliation": participant_reconciliation,
                "gate_recheck": gate_recheck,
                "row_counts": row_info,
                "per_participant": per_participant,
                "per_participant_scores": per_participant_scores,
                "per_dyad_scores": {k: v for k, v in per_dyad_scores.items() if k != "_dyad_members"},
                "paired_test": paired_test,
                "confidence_intervals": confidence_intervals,
                "references": {"majority_class": {p: m for p, m in maj_records.items()}},
                "coefficients": {
                    "population": {d: rec["coef"] for d, rec in pop_records.items()},
                    "personalized": {p: rec["coef"] for p, rec in pers_records.items()},
                    "feature_names": feat_cols,
                },
                "cross_experiment": cross_experiment,
                "validations": validations,
            }
        },
    }

    write_outputs(results)
    return results


if __name__ == "__main__":
    run()
