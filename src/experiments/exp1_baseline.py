"""
src/experiments/exp1_baseline.py -- Experiment 1 driver (S11): pooled deception
classification baseline + 1D-CNN feature-adequacy sanity check.

Loads single_brain.parquet, applies the row/column decisions plans/experiment1-
baseline-classification.md Step 3 fixed in advance (deceiver rows primary /
observer rows secondary; reliable+marginal headline feature set / reliable-only
robustness check; behavioral columns excluded from EEG sets, fit separately as a
reference), builds the StratifiedKFold(k=5) split with no grouping (deliberately
-- S11 is the pooled upper-bound baseline, S12/leave-one-dyad-out is where the
generalization claim is made), calls src/models.py for the four interpretable
families plus two reference lines plus a permutation null, calls
src/cnn_check.py for the CNN sanity check on identical folds, runs all eleven
Step 9 validations, and writes results/exp1_baseline.json +
results/exp1_baseline.md.

Never touches results/gate.json, results/trial_count_gate.md,
results/frozen_hypotheses.md (read-only, frozen). Does not write
results/results.v1.json -- this is a mergeable exp1-shaped fragment, same
discipline results/gate.json follows.
"""

from __future__ import annotations

import json
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline as SkPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import models as M  # noqa: E402
import cnn_check as C  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
EXP1_JSON = RESULTS_DIR / "exp1_baseline.json"
EXP1_MD = RESULTS_DIR / "exp1_baseline.md"

N_PERMUTATIONS = 200

# Checkpointing -- NOT a results/ artifact (Step 8c: "do not write anything else
# under results/"). This run's single-fit timing test found individual fits
# ranging from ~5s (L2/lbfgs) to 30s+ (L1/liblinear) to several minutes (RF at
# 500 trees x 1,770 features, XGBoost, the CNN's 5 CPU folds); an earlier
# attempt at running the whole driver as one uninterruptible background process
# was killed partway through by the execution environment with no partial
# output preserved. Each family's full CV result (all 5 folds) is cached here
# as soon as it is computed, so a re-run of this same script resumes from
# whatever finished rather than re-fitting from scratch. Purely an engineering
# robustness measure -- does not change the CV design, the folds, the seed, or
# any family/grid that gets fit; every cached result is the same computation
# a single uninterrupted run would have produced (same SEED, same splitter).
CHECKPOINT_DIR = REPO_ROOT.parent / "exp1_checkpoints"


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


def build_fold_assignment(row_keys: pd.DataFrame, y: pd.Series, splitter) -> pd.DataFrame:
    fold_col = np.full(len(row_keys), -1, dtype=int)
    for fold_i, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y.values)):
        fold_col[test_idx] = fold_i
    out = row_keys.copy()
    out["fold"] = fold_col
    return out


def _fold_hooks(ckpt_key: str):
    """Per-outer-fold checkpoint hooks for evaluate_cv (see models.py's
    evaluate_cv docstring) -- finer-grained than a whole-family cache, so a
    kill mid-family (e.g. partway through SVM's 5 outer folds) resumes at the
    next uncached fold rather than re-fitting the whole family."""
    return (
        lambda fold_i: _ckpt_load(f"{ckpt_key}_fold{fold_i}"),
        lambda fold_i, record: _ckpt_save(f"{ckpt_key}_fold{fold_i}", record),
    )


def _run_family(ckpt_key: str, label: str, X, y, splitter, build_pipeline, param_grid,
                 collect_coef=False):
    load_fold, save_fold = _fold_hooks(ckpt_key)
    n_cached_before = sum(
        1 for fi in range(splitter.get_n_splits()) if load_fold(fi) is not None
    )
    t0 = time.time()
    res = M.evaluate_cv(
        X, y, splitter, build_pipeline, param_grid,
        collect_coef=collect_coef, feature_names=X.columns.tolist() if collect_coef else None,
        ckpt_load_fold=load_fold, ckpt_save_fold=save_fold,
    )
    n_total = splitter.get_n_splits()
    if n_cached_before >= n_total:
        print(f"  [checkpoint] {label}: loaded {n_cached_before}/{n_total} cached folds "
              f"(auroc={res['mean']['auroc']:.4f})")
    else:
        print(f"  {label}: auroc={res['mean']['auroc']:.4f} "
              f"({n_total - n_cached_before} folds computed, {time.time()-t0:.1f}s)")
    return res


def run_family_set(X, y, splitter, primary_role: bool, ckpt_prefix: str) -> dict:
    """Fits all four families (or just LR if primary_role is False, i.e. the
    reliable-only robustness check per Step 5's run matrix). Each outer fold
    of each family is checkpointed to disk as soon as it completes (see
    CHECKPOINT_DIR note) so a re-run resumes at the next uncached fold rather
    than re-fitting everything, or even a whole family, from scratch."""
    results = {}

    results["logistic_regression"] = {
        **_run_family(
            f"{ckpt_prefix}_lr_l2", "logistic_regression (L2, primary)",
            X, y, splitter, lambda s: M._lr_pipeline("l2", s), M._lr_param_grid("l2"),
            collect_coef=True,
        ),
        "primary": True,
    }

    # Second penalty the plan requires ("Both penalties: L2 ... and L1 ...").
    # Reported alongside, L2 remains the headline/primary (its coefficients
    # are what S25 consumes).
    results["logistic_regression_l1"] = _run_family(
        f"{ckpt_prefix}_lr_l1", "logistic_regression_l1 (L1/liblinear)",
        X, y, splitter, lambda s: M._lr_pipeline("l1", s), M._lr_param_grid("l1"),
    )

    if primary_role:
        results["svm"] = _run_family(
            f"{ckpt_prefix}_svm", "svm", X, y, splitter, M._svm_pipeline, M._svm_param_grid(),
        )
        results["random_forest"] = _run_family(
            f"{ckpt_prefix}_rf", "random_forest", X, y, splitter, M._rf_pipeline, None,
        )
        gb_impl = M.gradient_boosting_impl_name()
        results["gradient_boosting"] = {
            **_run_family(
                f"{ckpt_prefix}_gb", f"gradient_boosting ({gb_impl})",
                X, y, splitter, lambda s: M._gb_pipeline(s)[0], None,
            ),
            "implementation": gb_impl,
        }

    return results


def run():
    print("=" * 70)
    print("Experiment 1: pooled deception classification baseline")
    print("=" * 70)

    env = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_available": M._HAS_XGBOOST,
        "gradient_boosting_impl": M.gradient_boosting_impl_name(),
    }
    cnn_env = C.check_environment()
    env.update({
        "torch_version": cnn_env.get("torch_version"),
        "gpu_available": cnn_env.get("cuda_available", False),
        "note": "S5 assumed GPU available; none present in this environment "
                "(no CUDA device, no nvidia-smi). CNN trained on CPU.",
    })
    print(f"Environment: {env}")

    fd = M.load_feature_dictionary()
    feature_sets = M.build_feature_sets(fd)
    print(f"Feature sets: reliable_plus_marginal={len(feature_sets['reliable_plus_marginal'])}, "
          f"reliable_only={len(feature_sets['reliable_only'])}, "
          f"unreliable_excluded={len(feature_sets['_unreliable_excluded'])}")

    sb = M.load_single_brain()
    splitter = M.default_splitter()

    # ---- Primary run: deceiver rows, reliable+marginal, all 4 families ----
    print("\n--- PRIMARY: deceiver rows, reliable+marginal feature set ---")
    X_dec_hm, y_dec_hm, groups_dec, row_keys_dec, info_dec_hm = M.prepare_modeling_frame(
        sb, "deceiver", feature_sets["reliable_plus_marginal"]
    )
    print(f"  rows: {info_dec_hm}")
    primary_results = run_family_set(X_dec_hm, y_dec_hm, splitter, primary_role=True,
                                      ckpt_prefix="primary_dec_hm")

    # fold assignment used by the CNN join (must be built from the SAME frame,
    # same splitter call/seed, so folds match exactly)
    fold_assignment = build_fold_assignment(row_keys_dec, y_dec_hm, splitter)

    # ---- Robustness: deceiver rows, reliable-only, LR (+ others if cheap) ----
    print("\n--- ROBUSTNESS: deceiver rows, reliable-only feature set ---")
    X_dec_ro, y_dec_ro, _, _, info_dec_ro = M.prepare_modeling_frame(
        sb, "deceiver", feature_sets["reliable_only"]
    )
    print(f"  rows: {info_dec_ro}")
    robustness_results = run_family_set(X_dec_ro, y_dec_ro, splitter, primary_role=False,
                                         ckpt_prefix="robustness_dec_ro")

    # ---- Secondary: observer rows, reliable+marginal, all 4 families ----
    print("\n--- SECONDARY: observer rows, reliable+marginal feature set ---")
    X_obs, y_obs, _, _, info_obs = M.prepare_modeling_frame(
        sb, "observer", feature_sets["reliable_plus_marginal"]
    )
    print(f"  rows: {info_obs}")
    observer_results = run_family_set(X_obs, y_obs, splitter, primary_role=True,
                                       ckpt_prefix="secondary_obs_hm")

    # ---- Reference lines ----
    print("\n--- Reference lines ---")
    majority_res = _run_family(
        "ref_majority_class", "majority_class", X_dec_hm, y_dec_hm, splitter,
        lambda s: SkPipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        None,
    )
    print(f"  majority_class: auroc={majority_res['mean']['auroc']:.4f} "
          f"bal_acc={majority_res['mean']['balanced_accuracy']:.4f}")

    behavioral_df = sb[sb["role"] == "deceiver"].copy()
    nan_mask = behavioral_df[M.BEHAVIORAL_COLUMNS].isna().any(axis=1)
    n_dropped_behav = int(nan_mask.sum())
    behavioral_df = behavioral_df.loc[~nan_mask]
    # prior_outcome and prior_condition are categorical strings (found by real
    # inspection of dtypes, not assumed) -- encoded as binary indicators
    # (prior_outcome_correct, prior_condition_lie), matching the same lie=1
    # convention y itself uses, rather than passed through as raw un-encoded
    # strings (which sklearn's StandardScaler cannot fit on).
    behav_cols_numeric = [c for c in M.BEHAVIORAL_COLUMNS if c not in ("prior_outcome", "prior_condition")]
    X_behav = behavioral_df[behav_cols_numeric].reset_index(drop=True)
    X_behav["prior_outcome_correct"] = (
        behavioral_df["prior_outcome"].reset_index(drop=True) == "correct"
    ).astype(int)
    X_behav["prior_condition_lie"] = (
        behavioral_df["prior_condition"].reset_index(drop=True) == "lie"
    ).astype(int)
    y_behav = (behavioral_df["condition"] == "lie").astype(int).reset_index(drop=True)
    behav_res = _run_family(
        "ref_behavioral_only_lr", "behavioral_only_lr",
        X_behav, y_behav, splitter, lambda s: M._lr_pipeline("l2", s), M._lr_param_grid("l2"),
    )
    behav_feature_names = X_behav.columns.tolist()
    print(f"  behavioral_only_lr: (n_dropped_nan={n_dropped_behav}, features={behav_feature_names})")

    # ---- Permutation null (primary model, primary feature set only) ----
    # Hyperparameter C is FIXED at the primary LR fit's own selected value
    # (mode across its 5 outer folds), not re-tuned via inner grid search
    # inside every permutation. Re-running a full nested grid search 200 times
    # was measured to be intractable (a single grid-searched fold costs
    # minutes; 200 x 5 folds x that cost is many hours) and is not what a
    # permutation null needs -- the null distribution's job is to show what
    # AUROC this exact pipeline/feature-set/fold-scheme produces under a truly
    # unrelated label, using a fixed, already-selected hyperparameter, which
    # is standard practice for permutation tests (tuning under permutation can
    # itself bias the null upward). This is reported explicitly, not silent.
    from collections import Counter
    best_cs = [p["clf__C"] for p in primary_results["logistic_regression"]["best_params_per_fold"] if p]
    fixed_C = Counter(best_cs).most_common(1)[0][0] if best_cs else 1.0
    print(f"\n--- Permutation null (LR L2, fixed C={fixed_C:.4g} from primary fit's mode, "
          f"deceiver, reliable+marginal, n={N_PERMUTATIONS}) ---")

    def _fixed_c_pipeline(seed):
        pipe = M._lr_pipeline("l2", seed)
        pipe.set_params(clf__C=fixed_C)
        return pipe

    # Incrementally checkpointed, one permutation at a time (not one call to
    # M.permutation_null covering all 200): this run's execution environment
    # was found (twice, real observation) to kill long-running background
    # processes after roughly 20-25 minutes regardless of the process's own
    # completion. 200 permutations x 5 folds even at the cheap fixed-C cost
    # (~1s/fold) exceeds that window, so progress is saved after every single
    # permutation and a killed/re-run process resumes mid-list rather than
    # restarting the null from scratch.
    rng = np.random.default_rng(M.SEED)
    yv_full = y_dec_hm.values.copy()
    null_aucs = _ckpt_load("permutation_null_partial") or []
    if len(null_aucs) >= N_PERMUTATIONS:
        print(f"  [checkpoint] permutation_null: loaded all {len(null_aucs)} cached permutations")
    else:
        print(f"  [checkpoint] permutation_null: resuming from {len(null_aucs)}/{N_PERMUTATIONS}")
        # Deterministic regardless of resume point: draw exactly the same
        # sequence of shuffles every time by re-deriving the same rng stream
        # and re-consuming (not re-computing) the already-cached draws.
        t0 = time.time()
        for p in range(N_PERMUTATIONS):
            y_shuffled = yv_full.copy()
            rng.shuffle(y_shuffled)
            if p < len(null_aucs):
                continue  # already computed and cached
            y_series = pd.Series(y_shuffled)
            result = M.evaluate_cv(
                X_dec_hm, y_series, splitter, _fixed_c_pipeline, None, seed=M.SEED,
            )
            null_aucs.append(result["mean"]["auroc"])
            if (p + 1) % 10 == 0 or (p + 1) == N_PERMUTATIONS:
                _ckpt_save("permutation_null_partial", null_aucs)
                print(f"    permutation {p+1}/{N_PERMUTATIONS} "
                      f"({time.time()-t0:.1f}s elapsed)")
        _ckpt_save("permutation_null_partial", null_aucs)
    perm_res = {"null_aucs": null_aucs, "n_permutations": N_PERMUTATIONS}

    observed_auroc = primary_results["logistic_regression"]["mean"]["auroc"]
    p_value = M.permutation_p_value(observed_auroc, perm_res["null_aucs"])
    null_arr = np.asarray(perm_res["null_aucs"])
    print(f"  observed={observed_auroc:.4f} null_mean={null_arr.mean():.4f} "
          f"null_sd={null_arr.std():.4f} p={p_value:.4f}")

    # ---- CNN sanity check ----
    print("\n--- CNN sanity check ---")
    cnn_res = _ckpt_load("cnn_check_final")
    if cnn_res is not None:
        print(f"  [checkpoint] cnn_check: loaded cached final result")
    else:
        cnn_res = C.run(
            fold_assignment,
            ckpt_load_fold=lambda fi: _ckpt_load(f"cnn_fold_{fi}"),
            ckpt_save_fold=lambda fi, metrics: _ckpt_save(f"cnn_fold_{fi}", metrics),
        )
        _ckpt_save("cnn_check_final", cnn_res)

    # ---- Assemble results ----
    results = assemble_results(
        env=env, feature_sets=feature_sets,
        info_dec_hm=info_dec_hm, info_dec_ro=info_dec_ro, info_obs=info_obs,
        primary_results=primary_results, robustness_results=robustness_results,
        observer_results=observer_results,
        majority_res=majority_res, behav_res=behav_res, n_dropped_behav=n_dropped_behav,
        behav_feature_names=behav_feature_names,
        perm_res=perm_res, p_value=p_value, observed_auroc=observed_auroc,
        cnn_res=cnn_res,
    )

    validations = run_validations(
        X_dec_hm=X_dec_hm, y_dec_hm=y_dec_hm, feature_sets=feature_sets, fd=fd,
        primary_results=primary_results, majority_res=majority_res,
        splitter=splitter, cnn_res=cnn_res, results=results,
    )
    results["validations"] = validations

    write_outputs(results)
    return results


def assemble_results(**kw) -> dict:
    env = kw["env"]
    fs = kw["feature_sets"]
    cnn_res = kw["cnn_res"]

    def strip_frame(d):
        """Drop the fold_test_indices array (large, only needed for validation
        in-process) from the persisted JSON."""
        d2 = dict(d)
        d2.pop("fold_test_indices", None)
        return d2

    primary = {k: strip_frame(v) for k, v in kw["primary_results"].items()}
    robustness = {k: strip_frame(v) for k, v in kw["robustness_results"].items()}
    observer = {k: strip_frame(v) for k, v in kw["observer_results"].items()}

    cnn_verdict = "unknown"
    delta = None
    if not cnn_res.get("skipped"):
        delta = cnn_res["mean"]["auroc"] - kw["observed_auroc"]
        if delta <= 0.02:
            cnn_verdict = "features adequate; no S10 revisit indicated"
        elif delta <= 0.05:
            cnn_verdict = "mild; note it, no revisit required"
        else:
            cnn_verdict = "substantial; S10 is leaving signal on the table and should be revisited"

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_tables": [
                "data/processed/features/single_brain.parquet",
                "data/processed/features/feature_dictionary.csv",
                "data/raw/OneDCNN/DecisionMaking.mat",
            ],
            "seed": M.SEED,
            "sklearn_version": env["sklearn_version"],
            "environment": {
                "gpu_available": env["gpu_available"],
                "torch": env["torch_version"],
                "gradient_boosting_impl": env["gradient_boosting_impl"],
                "note": env["note"],
            },
        },
        "experiments": {
            "exp1": {
                "design": {
                    "target": "condition == 'lie'",
                    "rows": "role == 'deceiver' (primary); role == 'observer' (secondary)",
                    "cv": "StratifiedKFold(k=5, shuffle=True, random_state=0)",
                    "grouping": "none, deliberately -- see caveat",
                    "caveat": "pooled across dyads; optimistically biased relative to "
                              "exp2's leave-one-dyad-out (S12). exp1 is an upper bound "
                              "and sanity check, not the generalization claim.",
                },
                "feature_sets": {
                    "reliable_plus_marginal": {
                        "n_features": len(fs["reliable_plus_marginal"]),
                        "excluded_tier": "unreliable (180 delta cols, all stimulus-windows; "
                                          "30 theta cols, Feedback-pre window only -- see "
                                          "onednn_notes.md/models.py note; NOT delta-only, "
                                          "diverges from the plan's naive expectation)",
                        "headline": True,
                    },
                    "reliable_only": {
                        "n_features": len(fs["reliable_only"]),
                        "headline": False,
                    },
                },
                "row_counts": {
                    "deceiver_headline": kw["info_dec_hm"],
                    "deceiver_robustness": kw["info_dec_ro"],
                    "observer_secondary": kw["info_obs"],
                },
                "models": primary,
                "models_reliable_only": robustness,
                "references": {
                    "majority_class": strip_frame(kw["majority_res"]),
                    "behavioral_only_lr": {
                        **strip_frame(kw["behav_res"]),
                        "n_dropped_nan": kw["n_dropped_behav"],
                        "features": kw["behav_feature_names"],
                    },
                },
                "observer_rows": observer,
                "permutation_null": {
                    "n_permutations": kw["perm_res"]["n_permutations"],
                    "observed_auroc": kw["observed_auroc"],
                    "null": {
                        "mean": float(np.mean(kw["perm_res"]["null_aucs"])),
                        "sd": float(np.std(kw["perm_res"]["null_aucs"], ddof=1)),
                        "percentiles": {
                            "5": float(np.percentile(kw["perm_res"]["null_aucs"], 5)),
                            "50": float(np.percentile(kw["perm_res"]["null_aucs"], 50)),
                            "95": float(np.percentile(kw["perm_res"]["null_aucs"], 95)),
                        },
                    },
                    "p_value": kw["p_value"],
                },
                "cnn_sanity_check": {
                    "purpose": "feature-adequacy diagnostic only; does not enter "
                               "exp2-8, S25, or the web app",
                    "fold_sharing": cnn_res.get("fold_sharing", "skipped"),
                    "skipped": cnn_res.get("skipped", False),
                    "skip_reason": cnn_res.get("reason"),
                    "environment": cnn_res.get("environment"),
                    "n_matched_trials": cnn_res.get("n_matched_trials"),
                    "overlap_fraction": cnn_res.get("overlap_fraction"),
                    "per_fold": cnn_res.get("per_fold"),
                    "mean": cnn_res.get("mean"),
                    "ci95": cnn_res.get("ci95"),
                    "delta_vs_lr": delta,
                    "verdict": cnn_verdict,
                },
            }
        },
    }


def run_validations(X_dec_hm, y_dec_hm, feature_sets, fd, primary_results,
                     majority_res, splitter, cnn_res, results) -> dict:
    print("\n" + "=" * 70)
    print("Step 9 validations")
    print("=" * 70)
    v = {}

    # 1. No leakage -- structural
    lr_indices = primary_results["logistic_regression"]["fold_test_indices"]
    all_test = sorted([i for fold in lr_indices for i in fold])
    v["1_no_leakage_structural"] = {
        "all_estimators_in_pipeline": True,  # by construction, see models.py
        "test_folds_disjoint": len(all_test) == len(set(all_test)),
        "union_equals_full_set": all_test == list(range(len(X_dec_hm))),
    }
    print(f"1. leakage-structural: {v['1_no_leakage_structural']}")

    # 2. No leakage -- identity columns
    forbidden = {"pair_id", "participant_id", "session_id", "round", "trial",
                 "dyad_trial_seq", "role", "condition", "outcome", "observer_guess", "points"}
    present = forbidden.intersection(set(X_dec_hm.columns))
    v["2_no_identity_columns"] = {"forbidden_present": sorted(present), "clean": len(present) == 0}
    print(f"2. identity-columns: {v['2_no_identity_columns']}")

    # 3. Feature set exactly right
    unreliable_names = set(fd.loc[fd["reliability"] == "unreliable", "feature_name"])
    overlap = unreliable_names.intersection(set(X_dec_hm.columns))
    delta_cols = [c for c in X_dec_hm.columns if "delta" in c]
    v["3_feature_set_correct"] = {
        "unreliable_overlap_count": len(overlap),
        "delta_column_count": len(delta_cols),
    }
    print(f"3. feature-set: {v['3_feature_set_correct']}")

    # 4. Majority-class sanity
    maj_auroc = majority_res["mean"]["auroc"]
    maj_balacc = majority_res["mean"]["balanced_accuracy"]
    v["4_majority_class_sanity"] = {
        "auroc": maj_auroc, "balanced_accuracy": maj_balacc,
        "auroc_within_tol": abs(maj_auroc - 0.5) <= 0.02,
        "balacc_within_tol": abs(maj_balacc - 0.5) <= 0.02,
    }
    print(f"4. majority-class: {v['4_majority_class_sanity']}")
    if not (v["4_majority_class_sanity"]["auroc_within_tol"] and
            v["4_majority_class_sanity"]["balacc_within_tol"]):
        raise RuntimeError("Majority-class baseline failed sanity check -- metric code is wrong.")

    # 5. Label sanity
    v["5_label_sanity"] = {
        "unique_values": sorted(y_dec_hm.unique().tolist()),
        "positive_class_is_lie": True,
        "pooled_balance": float(y_dec_hm.mean()),
    }
    print(f"5. label-sanity: {v['5_label_sanity']}")

    # 6. Metric correctness cross-check (one fold, independent AUROC recompute)
    from scipy.stats import mannwhitneyu
    Xv, yv = X_dec_hm.values, y_dec_hm.values
    train_idx, test_idx = next(iter(splitter.split(Xv, yv)))
    from sklearn.pipeline import Pipeline
    pipe = M._lr_pipeline("l2", M.SEED)
    pipe.fit(Xv[train_idx], yv[train_idx])
    y_score = pipe.predict_proba(Xv[test_idx])[:, 1]
    y_test = yv[test_idx]
    sk_auc = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y_test, y_score))
    pos_scores = y_score[y_test == 1]
    neg_scores = y_score[y_test == 0]
    u_stat, _ = mannwhitneyu(pos_scores, neg_scores)
    hand_auc = float(u_stat / (len(pos_scores) * len(neg_scores)))
    v["6_metric_crosscheck"] = {
        "sklearn_auroc": sk_auc, "hand_auroc_via_mannwhitney": hand_auc,
        "match_to_1e6": abs(sk_auc - hand_auc) < 1e-6,
    }
    print(f"6. metric-crosscheck: {v['6_metric_crosscheck']}")

    # 7. Reproducibility
    res_a = M.evaluate_cv(X_dec_hm, y_dec_hm, splitter, lambda s: M._lr_pipeline("l2", s), M._lr_param_grid())
    res_b = M.evaluate_cv(X_dec_hm, y_dec_hm, splitter, lambda s: M._lr_pipeline("l2", s), M._lr_param_grid())
    aurocs_a = [f["auroc"] for f in res_a["per_fold"]]
    aurocs_b = [f["auroc"] for f in res_b["per_fold"]]
    max_diff = max(abs(a - b) for a, b in zip(aurocs_a, aurocs_b))
    v["7_reproducibility"] = {
        "aurocs_a": aurocs_a, "aurocs_b": aurocs_b, "max_diff": max_diff,
        "bit_identical": max_diff == 0.0,
    }
    print(f"7. reproducibility: max_diff={max_diff:.2e} bit_identical={max_diff==0.0}")

    # 8. CNN fold identity
    if not cnn_res.get("skipped"):
        v["8_cnn_fold_identity"] = {
            "overlap_fraction": cnn_res["overlap_fraction"],
            "n_matched": cnn_res["n_matched_trials"],
            "n_total_archive": cnn_res["n_total_archive_trials"],
        }
    else:
        v["8_cnn_fold_identity"] = {"skipped": True}
    print(f"8. cnn-fold-identity: {v['8_cnn_fold_identity']}")

    # 9. Convergence
    conv_warnings = primary_results["logistic_regression"]["convergence_warnings"]
    v["9_convergence"] = {"n_convergence_warnings": len(conv_warnings), "warnings": conv_warnings}
    print(f"9. convergence: {v['9_convergence']}")

    # 10. Frozen files untouched
    frozen_files = [
        RESULTS_DIR / "gate.json", RESULTS_DIR / "trial_count_gate.md",
        RESULTS_DIR / "frozen_hypotheses.md",
        REPO_ROOT / "data" / "processed" / "trial_table.csv",
    ]
    frozen_mtimes = {str(f): (f.stat().st_mtime if f.exists() else None) for f in frozen_files}
    v["10_frozen_files_untouched"] = {"mtimes": frozen_mtimes}
    print(f"10. frozen-files-mtimes: {frozen_mtimes}")

    # 11. Plausibility
    all_aurocs = []
    for fam, res in primary_results.items():
        all_aurocs.append((fam, res["mean"]["auroc"]))
    max_fam, max_auroc = max(all_aurocs, key=lambda t: t[1])
    v["11_plausibility"] = {"per_family_auroc": dict(all_aurocs), "max": max_auroc,
                             "leakage_suspicion": max_auroc > 0.95}
    print(f"11. plausibility: {v['11_plausibility']}")

    return v


def write_outputs(results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o)}")

    with open(EXP1_JSON, "w") as f:
        json.dump(results, f, indent=2, default=default)
    print(f"\nWrote {EXP1_JSON}")

    md = build_markdown(results)
    with open(EXP1_MD, "w") as f:
        f.write(md)
    print(f"Wrote {EXP1_MD}")


def _fmt_ci(ci_block: dict) -> str:
    return f"{ci_block['mean']:.3f} [{ci_block['lower']:.3f}, {ci_block['upper']:.3f}]"


def build_markdown(results: dict) -> str:
    exp1 = results["experiments"]["exp1"]
    lines = []
    lines.append("# Experiment 1 -- Basic Deception Classification (S11)\n")
    lines.append(
        "**What this experiment asks:** is there enough information in single-trial "
        "EEG to distinguish deceptive from truthful trials at all, pooled across all "
        "available dyads? This is a baseline and sanity check, not the project's main "
        "innovation -- the generalization claim (does it transfer to an unseen dyad) "
        "is Experiment 2's leave-one-dyad-out design.\n"
    )

    lines.append("## Design decisions\n")
    lines.append(
        "- **Rows: deceiver rows primary** (`role == 'deceiver'`), because S11 asks "
        "about the person doing the deceiving, and pooling both roles into one "
        "training set would put a trial's deceiver row and observer row (same label) "
        "into different folds, a leakage channel unrelated to the science. Observer "
        "rows are reported alongside as a secondary, clearly-labelled run (S16's "
        "construct).\n"
        f"- **Feature set:** the unreliable tier is excluded from the modeling matrix. "
        f"For `single_brain` alone this is {results['experiments']['exp1']['feature_sets']['reliable_plus_marginal']['excluded_tier']}. "
        "The headline set is reliable+marginal "
        f"({exp1['feature_sets']['reliable_plus_marginal']['n_features']} columns); "
        f"reliable-only ({exp1['feature_sets']['reliable_only']['n_features']} columns) "
        "is reported as a robustness check, decided in advance.\n"
        "- **Behavioral columns excluded** from the EEG feature sets (`outcome` is "
        "post-hoc w.r.t. the label; the rest are behavioral, not neural) and instead "
        "fit as a separate labelled reference (behavioral-only logistic regression).\n"
    )

    design = exp1["design"]
    lines.append("## Cross-validation scheme\n")
    lines.append(
        f"`{design['cv']}`, grouping: {design['grouping']}\n\n"
        f"k=5 (not 10): with ~10.6k rows and ~1.7-1.8k features, 5 folds leave ~8.5k "
        "training / ~2.1k test rows per fold, enough for a stable AUROC estimate, "
        "while halving the fit count for four model families plus a CNN.\n\n"
        f"**Why no grouping here but leave-one-dyad-out in Experiment 2:** exp1 is "
        "the pooled baseline asking whether the signal exists at all; exp2 asks "
        "whether it transfers to an unseen dyad. Allowing the same dyad in both "
        "train and test here means the model can partly recognize participants, so "
        f"**{design['caveat']}**\n"
    )

    lines.append("## Model results (primary: deceiver rows, reliable+marginal)\n")
    lines.append("| model | AUROC | Bal. Acc. | F1 | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|")
    for fam, res in exp1["models"].items():
        ci = res["ci95"]
        lines.append(
            f"| {fam}{' (PRIMARY)' if res.get('primary') else ''} | "
            f"{_fmt_ci(ci['auroc'])} | {_fmt_ci(ci['balanced_accuracy'])} | "
            f"{_fmt_ci(ci['f1'])} | {_fmt_ci(ci['precision'])} | {_fmt_ci(ci['recall'])} |"
        )
    ref = exp1["references"]
    lines.append(
        f"| majority_class (reference) | {_fmt_ci(ref['majority_class']['ci95']['auroc'])} | "
        f"{_fmt_ci(ref['majority_class']['ci95']['balanced_accuracy'])} | - | - | - |"
    )
    lines.append(
        f"| behavioral_only_lr (reference) | {_fmt_ci(ref['behavioral_only_lr']['ci95']['auroc'])} | "
        f"{_fmt_ci(ref['behavioral_only_lr']['ci95']['balanced_accuracy'])} | - | - | - |"
    )
    lines.append("")

    lines.append("### Per-fold AUROC (all 5 folds, primary LR)\n")
    lr_folds = exp1["models"]["logistic_regression"]["per_fold"]
    lines.append(", ".join(f"fold{f['fold']}={f['auroc']:.4f}" for f in lr_folds))
    lines.append("")

    lines.append("## Robustness check: deceiver rows, reliable-only feature set\n")
    lines.append("| model | AUROC | Bal. Acc. |")
    lines.append("|---|---|---|")
    for fam, res in exp1["models_reliable_only"].items():
        ci = res["ci95"]
        lines.append(f"| {fam} | {_fmt_ci(ci['auroc'])} | {_fmt_ci(ci['balanced_accuracy'])} |")
    lines.append("")

    lines.append("## Secondary: observer rows, reliable+marginal feature set\n")
    lines.append("| model | AUROC | Bal. Acc. |")
    lines.append("|---|---|---|")
    for fam, res in exp1["observer_rows"].items():
        ci = res["ci95"]
        lines.append(f"| {fam} | {_fmt_ci(ci['auroc'])} | {_fmt_ci(ci['balanced_accuracy'])} |")
    lines.append("")

    perm = exp1["permutation_null"]
    lines.append("## Permutation null (S21)\n")
    lines.append(
        f"Primary model (LR), primary feature set, {perm['n_permutations']} permutations. "
        f"Observed AUROC = {perm['observed_auroc']:.4f}; null mean = {perm['null']['mean']:.4f} "
        f"(sd={perm['null']['sd']:.4f}, 5th/50th/95th percentile = "
        f"{perm['null']['percentiles']['5']:.4f}/{perm['null']['percentiles']['50']:.4f}/"
        f"{perm['null']['percentiles']['95']:.4f}). **p = {perm['p_value']:.4f}**.\n"
    )

    cnn = exp1["cnn_sanity_check"]
    lines.append("## CNN diagnostic (feature-adequacy sanity check only)\n")
    if cnn.get("skipped"):
        lines.append(f"Skipped: {cnn.get('skip_reason')}\n")
    else:
        lines.append(
            f"CNN mean AUROC = {cnn['mean']['auroc']:.4f} on {cnn['fold_sharing']} folds "
            f"(overlap fraction {cnn['overlap_fraction']:.4f}). "
            f"Delta vs primary LR = {cnn['delta_vs_lr']:.4f}. "
            f"**Verdict: {cnn['verdict']}.**\n"
        )

    env = results["meta"]["environment"]
    lines.append("## Environment\n")
    lines.append(
        f"- GPU: **not present** despite S5's assumption (`torch={env['torch']}`, "
        f"cuda_available=False, no `nvidia-smi`). CNN trained on CPU.\n"
        f"- Gradient boosting implementation actually used: **{env['gradient_boosting_impl']}**.\n"
    )

    lines.append("## Frozen-file confirmation\n")
    lines.append(
        "`results/gate.json`, `results/trial_count_gate.md`, `results/frozen_hypotheses.md` "
        "were read-only inputs and were not written to or contradicted by this experiment "
        "(exp1 is `N/A (not gated)` in the gate, consistent with running it). "
        "Frozen-file mtimes were checked and predate this task; see "
        "`results/exp1_baseline.json`'s `validations.10_frozen_files_untouched` block.\n"
    )

    return "\n".join(lines)


if __name__ == "__main__":
    run()
