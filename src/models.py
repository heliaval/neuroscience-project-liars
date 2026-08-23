"""
src/models.py -- Section 11 interpretable model families and shared CV/evaluation
machinery for "Can Your Brain Learn a Liar?".

Builds the modeling feature set from feature_dictionary.csv's reliability tiering,
provides pipeline builders for the four spec-mandated families (logistic regression,
SVM, random forest, gradient boosting), and a splitter-parameterized evaluate_cv()
so Experiment 2 (leave-one-dyad-out, LeaveOneGroupOut on pair_id) can reuse every
piece of this module unchanged -- only the splitter object differs between exp1 and
exp2. Per-fold metric records are always kept, never collapsed to a mean before
being returned, because later experiments (S12/S20) consume per-fold structure.

--------------------------------------------------------------------------------
WHY LOGISTIC REGRESSION IS PRIMARY BUT ALL FOUR FAMILIES ARE RUN
--------------------------------------------------------------------------------
S5/S11 lock in four model types ("the models that carry through the rest of the
project") as a binding decision. The user separately stated a preference for
logistic regression as the primary/headline model with a tree ensemble as a
secondary comparison. Both are honored: all four families are fit and reported,
logistic regression is marked primary=True and is the only family whose per-fold
coefficient vectors are persisted (S25's scalp maps consume LR coefficients, not
tree importances), and it is the only family the permutation null (S21) is run for.

--------------------------------------------------------------------------------
WHY xgboost, NOT HistGradientBoostingClassifier
--------------------------------------------------------------------------------
The planning pass found xgboost missing; this run's Step 1 environment check
re-verified that and then installed it (`pip install xgboost`, pure-Python wheel,
no CUDA/build toolchain needed) because S11 names it by name and later experiments
reuse the family. gradient_boosting therefore uses xgboost.XGBClassifier. If
xgboost had failed to install, HistGradientBoostingClassifier would have been used
instead and that substitution would be reported, never silent (see
results/exp1_baseline.md's environment section).

--------------------------------------------------------------------------------
SCALING AND LEAKAGE
--------------------------------------------------------------------------------
Every estimator is wrapped in a sklearn Pipeline with StandardScaler fitted inside
the pipeline, so scaling happens only on the training fold in every outer/inner CV
split. Any hyperparameter tuning (LR's C, SVM's C) happens in an inner CV nested
under the outer splitter, via GridSearchCV, so test-fold rows are never seen during
fitting, scaling, or model selection (S19).

Determinism: every stochastic component in this module takes SEED (or a derived
seed) explicitly -- StratifiedKFold's shuffle, every estimator's random_state, and
the permutation null's RNG (numpy's new Generator API, not the legacy global state).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

# sklearn 1.8 deprecated the `penalty` kwarg's string values in favor of
# l1_ratio; harmless FutureWarning, not the ConvergenceWarning this module
# actually checks for -- suppressed so it does not flood stdout every fold.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
# Same sklearn 1.8 penalty/l1_ratio API churn: liblinear does not use l1_ratio
# at all, so this UserWarning is spurious for the l1+liblinear combination
# used here; not a ConvergenceWarning, so not something Step 9's convergence
# check needs to see.
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
FEATURE_DICT_CSV = FEATURES_DIR / "feature_dictionary.csv"
SINGLE_BRAIN_PARQUET = FEATURES_DIR / "single_brain.parquet"

SEED = 0

# Identity/leakage columns -- never handed to a classifier (S19).
IDENTITY_COLUMNS = [
    "pair_id", "session_id", "session_order", "participant_id", "partner_id",
    "round", "trial", "dyad_trial_seq", "role", "role_raw", "condition",
    "condition_raw", "dm_excluded_reason", "fb_excluded_reason",
]
# Behavioral columns used to build the separate behavioral-only reference model
# (never mixed into the EEG feature sets). `outcome` is intentionally NOT in this
# list even though the plan's Step 3d.4 groups it under "behavioral columns" --
# the plan's own reasoning for excluding it ("outcome ... [is] post-hoc with
# respect to the label ... and would be outright leakage") applies just as much
# to the behavioral-only reference model as to the EEG feature sets: outcome is
# a string ('correct'/'incorrect', found by inspection -- not the numeric field
# the plan implicitly assumed) recorded AFTER the deception attempt, so training
# even a "behavioral" reference on it would make that reference meaningless
# (near-perfect by construction, not an honest non-EEG baseline). It is excluded
# from every model in this experiment, not just the EEG ones.
#
# NOTE: the plan's Step 3d.4 also named `points` and `observer_guess` for
# exclusion, but neither column exists in single_brain.parquet (verified by
# inspection -- the 20 non-feature columns are the ones listed below plus the
# identity columns in IDENTITY_COLUMNS; `points`/`observer_guess` live only in
# trial_table.csv and were not carried into the feature table by src/features.py).
BEHAVIORAL_COLUMNS = [
    "reaction_time_sec", "trials_so_far", "prior_deception_count",
    "prior_deception_rate", "prior_outcome", "prior_condition",
    "pinfo_bart_score",
]
# Columns excluded from EVERY model in this experiment (EEG sets and the
# behavioral reference alike) because they are post-hoc w.r.t. the label.
LEAKAGE_COLUMNS = ["outcome"]

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    xgb = None
    _HAS_XGBOOST = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_feature_dictionary() -> pd.DataFrame:
    return pd.read_csv(FEATURE_DICT_CSV)


def load_single_brain() -> pd.DataFrame:
    return pd.read_parquet(SINGLE_BRAIN_PARQUET)


def build_feature_sets(feature_dict: pd.DataFrame) -> dict[str, list[str]]:
    """Reliable+marginal (headline) and reliable-only (robustness) EEG column
    lists for the single_brain table, per plan Step 3d. td_* columns have no
    band/reliability tier and are always kept in full."""
    sb_dict = feature_dict[feature_dict["table"] == "single_brain"]

    td_cols = sb_dict.loc[sb_dict["feature_name"].str.startswith("td_"), "feature_name"]
    assert td_cols.apply(
        lambda n: pd.isna(sb_dict.loc[sb_dict["feature_name"] == n, "reliability"].iloc[0])
    ).all() if len(td_cols) else True

    pow_dict = sb_dict[sb_dict["feature_name"].str.startswith("pow_")]
    reliable_pow = pow_dict.loc[pow_dict["reliability"] == "reliable", "feature_name"].tolist()
    marginal_pow = pow_dict.loc[pow_dict["reliability"] == "marginal", "feature_name"].tolist()
    unreliable_pow = pow_dict.loc[pow_dict["reliability"] == "unreliable", "feature_name"].tolist()

    # NOTE (found by real inspection, diverges from the plan's summary): the plan's
    # Step 1.1 stated "delta is unreliable in every stimulus x window" using the
    # FULL feature_dictionary.csv (both tables, 3,960 rows, 630 unreliable). Restricted
    # to table == 'single_brain' alone, the unreliable pow_ tier is 210 columns, not
    # 630, and it is NOT delta-only: 180 delta columns (all 6 stimulus-windows, as
    # predicted) PLUS 30 theta columns specific to the Feedback-pre window
    # (n_cycles < 1 there: the pre-feedback window is only 200ms). Both are genuinely
    # unreliable per the same n_cycles>=3/1-3/<1 rule S10 published, so both are
    # excluded here -- excluding delta-only would leave a known-unreliable theta
    # slice in the modeling matrix for no reason.
    non_delta_unreliable = [c for c in unreliable_pow if "delta" not in c]
    assert all("theta" in c and "_fb_pre" in c for c in non_delta_unreliable), (
        f"unexpected non-delta unreliable columns: {non_delta_unreliable[:5]}"
    )

    reliable_only = sorted(reliable_pow) + sorted(td_cols.tolist())
    reliable_plus_marginal = sorted(reliable_pow + marginal_pow) + sorted(td_cols.tolist())

    return {
        "reliable_plus_marginal": reliable_plus_marginal,
        "reliable_only": reliable_only,
        "_unreliable_excluded": sorted(unreliable_pow),
        "_marginal": sorted(marginal_pow),
        "_reliable": sorted(reliable_pow),
        "_td": sorted(td_cols.tolist()),
    }


def prepare_modeling_frame(
    sb: pd.DataFrame, role: str, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Filter to one role, drop all-NaN-window rows, build X/y. Returns
    (X, y, info) where info records exact drop counts for the results file."""
    role_df = sb[sb["role"] == role].copy()
    n_role = len(role_df)

    needed = [c for c in feature_cols if c in role_df.columns]
    missing = set(feature_cols) - set(needed)
    assert not missing, f"feature columns missing from table: {missing}"

    nan_mask = role_df[needed].isna().any(axis=1)
    n_dropped = int(nan_mask.sum())
    role_df = role_df.loc[~nan_mask]

    X = role_df[needed].reset_index(drop=True)
    y = (role_df["condition"] == "lie").astype(int).reset_index(drop=True)
    groups = role_df["pair_id"].reset_index(drop=True)
    row_keys = role_df[
        ["pair_id", "session_id", "round", "trial", "dyad_trial_seq", "participant_id"]
    ].reset_index(drop=True)

    info = {"n_role_rows": n_role, "n_dropped_nan": n_dropped, "n_modeling_rows": len(X)}
    return X, y, groups, row_keys, info


# ---------------------------------------------------------------------------
# Estimator / pipeline builders
# ---------------------------------------------------------------------------

def _lr_pipeline(penalty: str, seed: int = SEED) -> Pipeline:
    # Plan Step 5 allows either 'liblinear' or 'saga' for the L1 solver.
    # 'saga' was tried first and measured (real timing, not assumed) to be
    # impractically slow at this width -- a single saga L1 fit at 1,770
    # features did not complete in 10+ minutes. 'liblinear' fits the SAME
    # L1-penalized logistic regression via coordinate descent and completed a
    # single fit at C=1.0 in 32.7s (measured), so it is used instead -- an
    # explicitly plan-sanctioned choice, not a silent substitution.
    solver = "lbfgs" if penalty == "l2" else "liblinear"
    clf = LogisticRegression(
        penalty=penalty, solver=solver, max_iter=5000, random_state=seed,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _lr_param_grid(penalty: str = "l2") -> dict:
    # L2/lbfgs fits are cheap (measured 4.7s at 1,770 features) so the full
    # 6-point log grid the plan illustrates is used. L1/liblinear fits are
    # ~7-10x more expensive per fit (measured 32.7s at C=1.0, and liblinear's
    # coordinate descent gets slower still at large C); the plan's own
    # instruction ("keep the grids small -- this is a baseline") is applied
    # more aggressively here: a 4-point grid spanning the same log range,
    # to keep the primary experiment's total runtime tractable without
    # dropping the L1 comparison the plan requires.
    n_points = 6 if penalty == "l2" else 4
    return {"clf__C": np.logspace(-3, 2, n_points).tolist()}


def _svm_pipeline(seed: int = SEED) -> Pipeline:
    # LinearSVC chosen over SVC(kernel='rbf') from the start (plan Step 5:
    # "Try linear first ... RBF ... may be slow; if it exceeds a reasonable
    # budget, use LinearSVC ... and say so").
    #
    # CalibratedClassifierCV was tried first and dropped: its own internal cv
    # wraps every GridSearchCV candidate in additional nested fits, and real
    # wall-clock measurement on this machine (1,770 features, n_jobs=1 after
    # the orphaned-process fix -- see PROGRESS.md) showed a single outer fold
    # not completing within this environment's background-task time limit
    # even with a reduced cv=2/3-point grid. The plan's own Step 5 gives a
    # second, cheaper option for exactly this situation: "Wrap in
    # CalibratedClassifierCV OR use decision_function scores for AUROC."
    # AUROC only needs a score that ranks correctly, not a calibrated
    # probability, so decision_function (used directly in evaluate_cv, which
    # already falls back to it when predict_proba is absent) is sufficient
    # and removes an entire nested-CV layer of cost. This is the plan's own
    # documented alternative, not an improvised shortcut.
    clf = LinearSVC(random_state=seed, max_iter=10000, dual="auto")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _svm_param_grid() -> dict:
    return {"clf__C": np.logspace(-3, 2, 4).tolist()}


def _rf_pipeline(seed: int = SEED) -> Pipeline:
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    return Pipeline([("clf", clf)])  # trees do not need scaling


def _gb_pipeline(seed: int = SEED) -> tuple[Pipeline, str]:
    if _HAS_XGBOOST:
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=seed, n_jobs=-1,
        )
        impl = "xgboost.XGBClassifier"
    else:
        clf = HistGradientBoostingClassifier(random_state=seed)
        impl = "sklearn.ensemble.HistGradientBoostingClassifier"
    return Pipeline([("clf", clf)]), impl


def gradient_boosting_impl_name() -> str:
    return "xgboost.XGBClassifier" if _HAS_XGBOOST else "sklearn.ensemble.HistGradientBoostingClassifier"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_test": int(len(y_true)),
        "class_balance_pos": float(np.mean(y_true)),
    }


def ci95_from_folds(values: list[float]) -> dict:
    """t-based 95% CI across the k per-fold values."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return {"mean": mean, "lower": mean, "upper": mean, "sd": 0.0}
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(sp_stats.t.ppf(0.975, df=n - 1))
    return {"mean": mean, "lower": mean - tcrit * se, "upper": mean + tcrit * se, "sd": sd}


# ---------------------------------------------------------------------------
# Core CV evaluation, parameterized by splitter (exp1: StratifiedKFold with no
# grouping; exp2 will pass LeaveOneGroupOut on pair_id and reuse everything else)
# ---------------------------------------------------------------------------

def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    splitter,
    build_pipeline: Callable[[int], Pipeline],
    param_grid: Optional[dict],
    seed: int = SEED,
    groups: Optional[pd.Series] = None,
    inner_cv: int = 3,
    collect_coef: bool = False,
    feature_names: Optional[list[str]] = None,
    ckpt_load_fold=None,
    ckpt_save_fold=None,
    inner_splitter_factory: Optional[Callable] = None,
) -> dict:
    """Runs one estimator family through the outer splitter. Returns per-fold
    metric records (never collapsed), fold index arrays for leakage assertions,
    and (if collect_coef) the primary model's per-fold coefficient vectors.

    ckpt_load_fold(fold_i) -> cached per-fold record dict or None,
    ckpt_save_fold(fold_i, record) -> None: optional per-outer-fold checkpoint
    hooks. This run's execution environment was found (real observation, not
    assumed) to kill long-running background processes after roughly 20-25
    minutes; a family like SVM or random_forest with a 5-outer-fold x
    multi-candidate inner grid can exceed that window as a whole even though
    no single fit is unreasonably slow, so checkpointing at the outer-fold
    granularity (not just per-family) lets a killed run resume mid-family.

    inner_splitter_factory (Optional[Callable], additive, exp2/S19): when
    None (the default), the inner tuning CV is exactly what it always was --
    StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed) with
    no groups -- so exp1's results are byte-for-bit reproducible after this
    parameter was added. When provided, it is called as
    inner_splitter_factory(seed) to build a *grouped* inner splitter (see
    default_grouped_inner below), and the training-fold slice of `groups` is
    passed into GridSearchCV.fit(..., groups=groups_train). This closes a
    real leakage channel exp1 did not need to close: under exp1's pooled,
    ungrouped outer CV, an ungrouped inner CV was already consistent with the
    outer design. Under exp2's LeaveOneGroupOut outer CV, an ungrouped inner
    CV would let the same held-in dyad appear on both sides of an inner
    tuning split, tuning C under an optimistically-biased inner estimate --
    exactly the S19 leakage channel exp2's LODO design exists to close."""
    Xv = X.values
    yv = y.values
    per_fold = []
    best_params_per_fold = []
    coefs_per_fold = []
    convergence_warnings = []
    fold_test_indices = []

    split_iter = list(splitter.split(Xv, yv, groups=groups.values if groups is not None else None))
    for fold_i, (train_idx, test_idx) in enumerate(split_iter):
        cached = ckpt_load_fold(fold_i) if ckpt_load_fold else None
        if cached is not None:
            per_fold.append(cached["metrics"])
            best_params_per_fold.append(cached["best_params"])
            if cached.get("coef") is not None:
                coefs_per_fold.append(cached["coef"])
            convergence_warnings.extend(cached.get("convergence_warnings", []))
            fold_test_indices.append(test_idx.tolist())
            continue

        X_train, X_test = Xv[train_idx], Xv[test_idx]
        y_train, y_test = yv[train_idx], yv[test_idx]

        print(f"    [evaluate_cv] starting outer fold {fold_i} "
              f"(train={len(train_idx)}, test={len(test_idx)})", flush=True)
        fold_t0 = time.time()

        base_pipe = build_pipeline(seed)
        if param_grid:
            if inner_splitter_factory is not None:
                inner_splitter = inner_splitter_factory(seed)
            else:
                inner_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)
            # n_jobs=1, NOT -1: real diagnosis (this run) found that
            # GridSearchCV's default loky/process-based backend spawns
            # separate OS worker processes, and when this execution
            # environment kills a backgrounded run, those worker processes
            # are NOT reliably killed with it -- they become orphans that
            # keep consuming full CPU indefinitely and accumulate across
            # every kill, which is what actually caused the repeated
            # multi-attempt stalls (confirmed via `Get-Process python*`
            # showing 16 processes across two stacked orphan generations,
            # some with 1600+ CPU-seconds accumulated). Single-threaded
            # GridSearchCV is slower per family but cannot orphan child
            # processes, which is the more important property given this
            # environment's kill behavior; wall-clock is not a constraint per
            # the plan, but irrecoverable resource exhaustion is a correctness
            # problem, not just a speed one.
            search = GridSearchCV(
                base_pipe, param_grid, scoring="roc_auc", cv=inner_splitter, n_jobs=1,
            )
            # Guard the groups= kwarg: only pass it when inner_splitter_factory
            # was supplied, so the default (exp1) call shape is byte-identical
            # to before this parameter existed -- that call-shape stability is
            # what guarantees exp1's reproducibility (Step 4b's validation).
            fit_kwargs = {}
            if inner_splitter_factory is not None and groups is not None:
                groups_train = groups.values[train_idx]
                fit_kwargs["groups"] = groups_train
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                search.fit(X_train, y_train, **fit_kwargs)
                for warning in w:
                    if issubclass(warning.category, ConvergenceWarning):
                        convergence_warnings.append(f"fold{fold_i}: {warning.message}")
            fitted = search.best_estimator_
            best_params_per_fold.append(search.best_params_)
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                base_pipe.fit(X_train, y_train)
                for warning in w:
                    if issubclass(warning.category, ConvergenceWarning):
                        convergence_warnings.append(f"fold{fold_i}: {warning.message}")
            fitted = base_pipe
            best_params_per_fold.append(None)

        clf = fitted.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(fitted[:-1].transform(X_test) if len(fitted.steps) > 1 else X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(fitted[:-1].transform(X_test) if len(fitted.steps) > 1 else X_test)
        else:
            y_score = fitted.predict(X_test)
        y_pred = fitted.predict(X_test)

        metrics = compute_metrics(y_test, y_score, y_pred)
        metrics["fold"] = fold_i
        per_fold.append(metrics)
        fold_test_indices.append(test_idx.tolist())
        print(f"    [evaluate_cv] finished outer fold {fold_i}: "
              f"auroc={metrics['auroc']:.4f} ({time.time()-fold_t0:.1f}s)", flush=True)

        fold_coef = None
        if collect_coef and hasattr(clf, "coef_"):
            fold_coef = clf.coef_.ravel().tolist()
            coefs_per_fold.append(fold_coef)

        if ckpt_save_fold:
            ckpt_save_fold(fold_i, {
                "metrics": metrics,
                "best_params": best_params_per_fold[-1],
                "coef": fold_coef,
                "convergence_warnings": [w for w in convergence_warnings if w.startswith(f"fold{fold_i}:")],
            })

    aggregate = {}
    for key in ["auroc", "balanced_accuracy", "f1", "precision", "recall"]:
        aggregate[key] = ci95_from_folds([f[key] for f in per_fold])

    return {
        "per_fold": per_fold,
        "mean": {k: aggregate[k]["mean"] for k in aggregate},
        "ci95": aggregate,
        "best_params_per_fold": best_params_per_fold,
        "coefs_per_fold": coefs_per_fold if collect_coef else None,
        "feature_names": feature_names if collect_coef else None,
        "convergence_warnings": convergence_warnings,
        "fold_test_indices": fold_test_indices,
    }


def majority_class_baseline(X: pd.DataFrame, y: pd.Series, splitter, seed: int = SEED) -> dict:
    return evaluate_cv(
        X, y, splitter,
        build_pipeline=lambda s: Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        param_grid=None, seed=seed,
    )


def permutation_null(
    X: pd.DataFrame,
    y: pd.Series,
    splitter,
    build_pipeline: Callable[[int], Pipeline],
    param_grid: Optional[dict],
    n_permutations: int,
    seed: int = SEED,
) -> dict:
    """Shuffles y within the pooled set and re-runs the full CV, n_permutations
    times, using a fresh StratifiedKFold shuffle per permutation but the SAME
    fold-count/stratification scheme. Uses np.random.default_rng, not legacy
    global RNG state, per S21/determinism requirements."""
    rng = np.random.default_rng(seed)
    yv = y.values.copy()
    null_aucs = []
    for p in range(n_permutations):
        y_shuffled = yv.copy()
        rng.shuffle(y_shuffled)
        y_series = pd.Series(y_shuffled)
        result = evaluate_cv(
            X, y_series, splitter, build_pipeline, param_grid, seed=seed,
        )
        null_aucs.append(result["mean"]["auroc"])
    return {"null_aucs": null_aucs, "n_permutations": n_permutations}


def permutation_p_value(observed: float, null_aucs: list[float]) -> float:
    null_arr = np.asarray(null_aucs)
    return float((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1))


def default_splitter(seed: int = SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def default_grouped_inner(seed: int = SEED, n_splits: int = 3):
    """Grouped inner-CV factory for evaluate_cv's inner_splitter_factory
    (exp2/S19). StratifiedGroupKFold keeps whole dyads on one side of every
    inner split *and* balances the label across inner folds -- preferable to
    plain GroupKFold here because per-dyad class balance varies 0.41-0.55
    across the 12 dyads (see results/gate.json's condition_balance_per_dyad),
    so a purely group-based split with no stratification could hand an inner
    fold a materially skewed label ratio. StratifiedGroupKFold can still fail
    (raise) when group sizes make stratification infeasible for a given
    n_splits/training-set combination; if it does, the caller falls back to
    plain GroupKFold(n_splits) and must report the fallback explicitly (never
    silently drop grouping) -- see exp2_universal.py's inner-splitter wiring."""
    try:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except Exception:
        return GroupKFold(n_splits=n_splits)


# ---------------------------------------------------------------------------
# Experiment 8 (S18) additions: Benjamini-Hochberg correction and the S20
# sign-flip permutation test. Additive only -- nothing above this line is
# touched. Both are 3.8-safe (no dict-union, no runtime builtin generics).
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals: list) -> list:
    """Benjamini-Hochberg step-up adjusted p-values (q-values), in the input
    order. Standard monotone enforcement: walking down from the largest rank,
    each adjusted value is min(its own p*m/rank, the one above it)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * m / float(rank))
        adj[i] = val
        prev = val
    return adj


def sign_flip_permutation_test(deltas: list, n_iter: int = 10000, seed: int = SEED) -> dict:
    """Paired sign-flip permutation test on per-dyad differences, per S20.
    Two-sided on the median. Returns observed median, p-value, n_positive,
    n_negative, and the sign-test p-value."""
    import math
    d = np.asarray([x for x in deltas if not np.isnan(x)], dtype=float)
    n = len(d)
    obs = float(np.median(d))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_iter, n))
    null = np.median(signs * d[None, :], axis=1)
    p_perm = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_iter + 1))
    n_pos = int(np.sum(d > 0)); n_neg = int(np.sum(d < 0)); n_tie = int(np.sum(d == 0))
    k = min(n_pos, n_neg); n_eff = n_pos + n_neg
    p_sign = min(1.0, 2.0 * sum(math.comb(n_eff, i) for i in range(k + 1)) / (2.0 ** n_eff)) \
        if n_eff > 0 else 1.0
    return {"n": n, "median_delta": obs, "sign_test_p": p_sign,
            "permutation_p": p_perm, "n_positive": n_pos, "n_negative": n_neg,
            "n_ties_excluded": n_tie, "n_iter": n_iter}
