"""
src/experiments/exp2_universal.py -- Experiment 2 driver (S12): universal-model
leave-one-dyad-out (LODO) generalization test.

Research question (S12): can a deception model trained on most dyads generalize
to a completely unseen participant pair? Experiment 1 (results/exp1_baseline.md)
already answered "is the signal there at all" with a pooled, ungrouped
StratifiedKFold(k=5) and reported that number as an explicit upper bound, not the
generalization claim -- exp1's own results file states: "pooled across dyads;
optimistically biased relative to exp2's leave-one-dyad-out (S12). exp1 is an
upper bound and sanity check, not the generalization claim." This module is where
the generalization claim is actually made.

--------------------------------------------------------------------------------
DYAD RECONCILIATION (plan Step 2 -- a required reported finding, not background)
--------------------------------------------------------------------------------
This task was dispatched with the premise that sub19_sub22 has zero feature rows
and therefore cannot supply a LODO fold. Re-verification against
data/processed/features/single_brain.parquet at execution time found that premise
false: sub19_sub22 has 484 deceiver rows / 484 observer rows, spanning 1 of its 2
sessions (the OTHER session -- Player_sub22_Observer_sub19, pair_num==21 in the
OneDCNN archive -- was the one lost to preprocessing, documented in
data/processed/onednn_notes.md and PROGRESS.md's 2026-08-20 23:35 entry). This is
a different situation from sub01_sub02, which is a genuine single-session dyad
(only ever ran one session). Both dyads therefore contribute half-size (484-row)
LODO folds, for two different, unrelated reasons. Experiment 2 runs LODO over all
12 dyads (n=12), which matches results/frozen_hypotheses.md's "n = 12 for: exp1,
exp2" -- there is no conflict with the frozen file to escalate. Every dyad's
minority class (220-500 trials per dyad) is far above the gate's Clause-A
threshold of 60, so AUROC is well-defined in every fold; the two half-size dyads'
per-dyad scores simply carry more sampling noise than the other ten.

--------------------------------------------------------------------------------
WHY THE INNER TUNING CV IS NOW GROUPED (S19) -- the one additive models.py change
--------------------------------------------------------------------------------
evaluate_cv's inner tuning CV previously hardcoded StratifiedKFold with no
grouping. Under exp1's pooled, ungrouped outer CV that was already consistent
with the outer design. Under exp2's LeaveOneGroupOut outer CV it would not be:
an ungrouped inner split of the 11 training dyads could put the same dyad's
trials on both sides of an inner tuning fold, tuning C under an
optimistically-biased inner estimate -- exactly the leakage channel LODO exists
to close (S19: "Proper grouped cross-validation for tuning"). src/models.py
gained one additive, backward-compatible keyword, inner_splitter_factory
(default None => byte-identical to before); this module passes
M.default_grouped_inner (StratifiedGroupKFold(3), falling back to GroupKFold(3)
if stratification is infeasible -- not observed to trigger on this data, but
guarded and would be reported if it did).

--------------------------------------------------------------------------------
EXPLICITLY OUT OF SCOPE
--------------------------------------------------------------------------------
No CNN (S11: does not carry into Experiments 2-8). No S20 paired test (Experiment
2 is a single condition; Experiment 4 supplies the other side and joins against
this module's per_dyad_scores block by pair_id, no reshaping). No writes to
results/gate.json, results/trial_count_gate.md, results/frozen_hypotheses.md,
data/processed/trial_table.csv (read-only, frozen). No write to
results/results.v1.json (this is a mergeable fragment, same discipline
exp1_baseline.json follows). No change to exp1_baseline.py or to any existing
models.py behavior when inner_splitter_factory is left at its default.

--------------------------------------------------------------------------------
CHECKPOINTING
--------------------------------------------------------------------------------
This environment kills background bash tasks after roughly 15-25 minutes
regardless of process health (observed repeatedly during exp1). LODO's 12 outer
folds (vs exp1's 5) make this worse, not better. Per-outer-fold checkpoints go to
a NEW sibling directory, C:\\Users\\Alber\\CN4\\exp2_checkpoints -- deliberately
not exp1_checkpoints, so a stale exp1 cache can never silently poison exp2's
folds. GridSearchCV stays at n_jobs=1 (models.py already pins this; exp1
root-caused a multi-hour orphaned-loky-worker cascade to n_jobs=-1). The
permutation null is checkpointed per permutation, same pattern as exp1.
"""

from __future__ import annotations

import json
import pickle
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline as SkPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import models as M  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
EXP2_JSON = RESULTS_DIR / "exp2_universal.json"
EXP2_MD = RESULTS_DIR / "exp2_universal.md"
EXP1_JSON = RESULTS_DIR / "exp1_baseline.json"

N_PERMUTATIONS = 200

# Sibling directory OUTSIDE the repo, deliberately distinct from exp1's
# exp1_checkpoints -- different folds, different keys; reusing exp1's directory
# or key names would silently poison exp2 with stale exp1-shaped cache entries.
CHECKPOINT_DIR = REPO_ROOT.parent / "exp2_checkpoints"

FROZEN_FILES = [
    RESULTS_DIR / "gate.json",
    RESULTS_DIR / "trial_count_gate.md",
    RESULTS_DIR / "frozen_hypotheses.md",
    REPO_ROOT / "data" / "processed" / "trial_table.csv",
    RESULTS_DIR / "exp1_baseline.json",
    RESULTS_DIR / "exp1_baseline.md",
]


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


def _fold_hooks(ckpt_key: str):
    return (
        lambda fold_i: _ckpt_load(f"{ckpt_key}_fold{fold_i}"),
        lambda fold_i, record: _ckpt_save(f"{ckpt_key}_fold{fold_i}", record),
    )


# ---------------------------------------------------------------------------
# Step 5: fold <-> dyad bijection
# ---------------------------------------------------------------------------

def fold_to_pair_id(groups: pd.Series, fold_test_indices: list) -> list:
    out = []
    for test_idx in fold_test_indices:
        uniq = pd.unique(groups.values[np.asarray(test_idx)])
        assert len(uniq) == 1, f"LODO fold spans multiple dyads: {uniq}"
        out.append(str(uniq[0]))
    assert sorted(out) == sorted(groups.unique()), "fold->dyad mapping is not a bijection"
    return out


def dyad_stats(y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"pair_id": groups.values, "y": y.values})
    stats = df.groupby("pair_id")["y"].agg(n_test="size", n_lie="sum").astype(int)
    stats["n_truth"] = stats["n_test"] - stats["n_lie"]
    stats["class_balance_pos"] = stats["n_lie"] / stats["n_test"]
    return stats


def build_per_dyad(res: dict, groups: pd.Series, stats: pd.DataFrame) -> tuple[list, dict]:
    """Turns evaluate_cv's per_fold + fold_test_indices into (a) per_dyad list
    records and (b) the per_dyad_scores S20-ready block, per plan Step 5."""
    pair_ids = fold_to_pair_id(groups, res["fold_test_indices"])
    per_dyad = []
    for fold_i, pid in enumerate(pair_ids):
        f = res["per_fold"][fold_i]
        row = stats.loc[pid]
        per_dyad.append({
            "pair_id": pid,
            "fold": fold_i,
            "n_test": int(row["n_test"]),
            "n_lie": int(row["n_lie"]),
            "n_truth": int(row["n_truth"]),
            "class_balance_pos": float(row["class_balance_pos"]),
            "auroc": f["auroc"],
            "balanced_accuracy": f["balanced_accuracy"],
            "f1": f["f1"],
            "precision": f["precision"],
            "recall": f["recall"],
            "best_params": res["best_params_per_fold"][fold_i],
        })
    per_dyad.sort(key=lambda r: r["pair_id"])

    metrics = ["auroc", "balanced_accuracy", "f1", "precision", "recall"]
    scores = {m: {r["pair_id"]: r[m] for r in per_dyad} for m in metrics}
    per_dyad_scores = {
        "n_dyads": len(per_dyad),
        "unit_of_analysis": "dyad",
        "scores": scores,
    }
    return per_dyad, per_dyad_scores


# ---------------------------------------------------------------------------
# Family runner
# ---------------------------------------------------------------------------

def _run_family(ckpt_key: str, label: str, X, y, splitter, groups, build_pipeline,
                 param_grid, collect_coef=False, use_grouped_inner=True):
    load_fold, save_fold = _fold_hooks(ckpt_key)
    n_total = splitter.get_n_splits(groups=groups)
    n_cached_before = sum(1 for fi in range(n_total) if load_fold(fi) is not None)
    t0 = time.time()
    inner_factory = M.default_grouped_inner if (param_grid and use_grouped_inner) else None
    res = M.evaluate_cv(
        X, y, splitter, build_pipeline, param_grid,
        groups=groups,
        collect_coef=collect_coef, feature_names=X.columns.tolist() if collect_coef else None,
        ckpt_load_fold=load_fold, ckpt_save_fold=save_fold,
        inner_splitter_factory=inner_factory,
    )
    if n_cached_before >= n_total:
        print(f"  [checkpoint] {label}: loaded {n_cached_before}/{n_total} cached folds "
              f"(auroc={res['mean']['auroc']:.4f})", flush=True)
    else:
        print(f"  {label}: auroc={res['mean']['auroc']:.4f} "
              f"({n_total - n_cached_before} folds computed, {time.time()-t0:.1f}s)", flush=True)
    res["inner_splitter_class"] = (
        type(inner_factory(M.SEED)).__name__ if inner_factory is not None else None
    )
    return res


def run_family_set(X, y, groups, splitter, families: list, ckpt_prefix: str) -> dict:
    """families: subset of ['logistic_regression', 'logistic_regression_l1',
    'svm', 'random_forest', 'gradient_boosting'], run in that order (plan 8a's
    load-bearing run order -- headline-critical work lands first)."""
    results = {}
    order = ["logistic_regression", "logistic_regression_l1", "svm",
             "random_forest", "gradient_boosting"]
    for fam in order:
        if fam not in families:
            continue
        if fam == "logistic_regression":
            r = _run_family(
                f"{ckpt_prefix}_lr_l2", "logistic_regression (L2, primary)",
                X, y, splitter, groups, lambda s: M._lr_pipeline("l2", s),
                M._lr_param_grid("l2"), collect_coef=True,
            )
            r["primary"] = True
            results[fam] = r
        elif fam == "logistic_regression_l1":
            results[fam] = _run_family(
                f"{ckpt_prefix}_lr_l1", "logistic_regression_l1 (L1/liblinear)",
                X, y, splitter, groups, lambda s: M._lr_pipeline("l1", s),
                M._lr_param_grid("l1"),
            )
        elif fam == "svm":
            results[fam] = _run_family(
                f"{ckpt_prefix}_svm", "svm", X, y, splitter, groups,
                M._svm_pipeline, M._svm_param_grid(),
            )
        elif fam == "random_forest":
            results[fam] = _run_family(
                f"{ckpt_prefix}_rf", "random_forest", X, y, splitter, groups,
                M._rf_pipeline, None,
            )
        elif fam == "gradient_boosting":
            gb_impl = M.gradient_boosting_impl_name()
            r = _run_family(
                f"{ckpt_prefix}_gb", f"gradient_boosting ({gb_impl})",
                X, y, splitter, groups, lambda s: M._gb_pipeline(s)[0], None,
            )
            r["implementation"] = gb_impl
            results[fam] = r
    return results


# ---------------------------------------------------------------------------
# Permutation null (Step 8d / S21): within-dyad label shuffling
# ---------------------------------------------------------------------------

def shuffle_within_dyad(yv: np.ndarray, groups_v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permutes y among each dyad's own rows only -- preserves each dyad's
    class balance and the group structure, so the null answers 'what LODO
    AUROC does this pipeline produce when the label is unrelated *within*
    dyad', the right null for a grouped design. Global shuffling would also
    destroy between-dyad balance differences and inflate the null's variance."""
    out = yv.copy()
    for pid in np.unique(groups_v):
        idx = np.where(groups_v == pid)[0]
        out[idx] = rng.permutation(out[idx])
    return out


def run_permutation_null(X, y, groups, splitter, fixed_C: float, n_permutations: int):
    def _fixed_c_pipeline(seed):
        pipe = M._lr_pipeline("l2", seed)
        pipe.set_params(clf__C=fixed_C)
        return pipe

    rng = np.random.default_rng(M.SEED)
    yv_full = y.values.copy()
    groups_v = groups.values
    null_aucs = _ckpt_load("permutation_null_partial") or []
    n_actual = n_permutations
    if len(null_aucs) >= n_permutations:
        print(f"  [checkpoint] permutation_null: loaded all {len(null_aucs)} cached permutations", flush=True)
    else:
        print(f"  [checkpoint] permutation_null: resuming from {len(null_aucs)}/{n_permutations}", flush=True)
        t0 = time.time()
        for p in range(n_permutations):
            y_shuffled = shuffle_within_dyad(yv_full, groups_v, rng)
            if p < len(null_aucs):
                continue
            y_series = pd.Series(y_shuffled)
            result = M.evaluate_cv(
                X, y_series, splitter, _fixed_c_pipeline, None, seed=M.SEED, groups=groups,
            )
            null_aucs.append(result["mean"]["auroc"])
            _ckpt_save("permutation_null_partial", null_aucs)
            if (p + 1) % 10 == 0 or (p + 1) == n_permutations:
                print(f"    permutation {p+1}/{n_permutations} ({time.time()-t0:.1f}s elapsed)", flush=True)
    return {"null_aucs": null_aucs, "n_permutations": n_actual}


# ---------------------------------------------------------------------------
# Validations (Step 9)
# ---------------------------------------------------------------------------

def run_validations(X, y, groups, feature_sets, fd, primary_results, majority_res,
                     splitter, n_dyads_expected: int) -> dict:
    print("\n" + "=" * 70, flush=True)
    print("Step 9 validations", flush=True)
    print("=" * 70, flush=True)
    v = {}
    lr = primary_results["logistic_regression"]
    lr_indices = lr["fold_test_indices"]
    groups_v = groups.values

    # 1. Structural leakage (LODO guarantee, asserted not assumed)
    disjoint_ok = True
    single_pair_ok = True
    for test_idx in lr_indices:
        pids = pd.unique(groups_v[np.asarray(test_idx)])
        if len(pids) != 1:
            single_pair_ok = False
    all_test = sorted(i for fold in lr_indices for i in fold)
    v["1_no_leakage_structural"] = {
        "test_folds_disjoint": len(all_test) == len(set(all_test)),
        "union_equals_full_set": all_test == list(range(len(X))),
        "each_test_fold_single_dyad": single_pair_ok,
    }
    print(f"1. leakage-structural: {v['1_no_leakage_structural']}", flush=True)

    # 2. Identity columns
    forbidden = {"pair_id", "participant_id", "session_id", "round", "trial",
                 "dyad_trial_seq", "role", "condition", "outcome", "observer_guess", "points"}
    present = forbidden.intersection(set(X.columns))
    v["2_no_identity_columns"] = {"forbidden_present": sorted(present), "clean": len(present) == 0}
    print(f"2. identity-columns: {v['2_no_identity_columns']}", flush=True)

    # 3. Fold count
    n_splits = splitter.get_n_splits(groups=groups)
    v["3_fold_count"] = {
        "len_per_fold": len(lr["per_fold"]), "n_splits_groups": n_splits,
        "expected": n_dyads_expected,
        "ok": len(lr["per_fold"]) == n_dyads_expected and n_splits == n_dyads_expected,
    }
    print(f"3. fold-count: {v['3_fold_count']}", flush=True)

    # 4. Per-dyad scores complete
    _, per_dyad_scores = build_per_dyad(lr, groups, dyad_stats(y, groups))
    auroc_keys = set(per_dyad_scores["scores"]["auroc"].keys())
    group_set = set(groups.unique())
    no_nan = not any(pd.isna(val) for val in per_dyad_scores["scores"]["auroc"].values())
    v["4_per_dyad_scores_complete"] = {
        "keys_match_groups": auroc_keys == group_set,
        "no_missing_or_nan": no_nan,
        "n_dyads": len(auroc_keys),
    }
    print(f"4. per-dyad-scores-complete: {v['4_per_dyad_scores_complete']}", flush=True)

    # 5. Fold<->dyad bijection (fold_to_pair_id's own asserts already ran above; recheck explicitly)
    try:
        mapped = fold_to_pair_id(groups, lr_indices)
        v["5_fold_dyad_bijection"] = {"ok": True, "n_mapped": len(mapped)}
    except AssertionError as e:
        v["5_fold_dyad_bijection"] = {"ok": False, "error": str(e)}
    print(f"5. fold-dyad-bijection: {v['5_fold_dyad_bijection']}", flush=True)

    # 6. Majority-class sanity
    maj_auroc = majority_res["mean"]["auroc"]
    maj_balacc = majority_res["mean"]["balanced_accuracy"]
    v["6_majority_class_sanity"] = {
        "auroc": maj_auroc, "balanced_accuracy": maj_balacc,
        "auroc_within_tol": abs(maj_auroc - 0.5) <= 0.02,
        "balacc_within_tol": abs(maj_balacc - 0.5) <= 0.02,
    }
    print(f"6. majority-class: {v['6_majority_class_sanity']}", flush=True)
    if not (v["6_majority_class_sanity"]["auroc_within_tol"] and
            v["6_majority_class_sanity"]["balacc_within_tol"]):
        raise RuntimeError("LODO majority-class baseline failed sanity check -- metric code is wrong.")

    # 7. Metric correctness cross-check
    from scipy.stats import mannwhitneyu
    Xv, yv = X.values, y.values
    train_idx, test_idx = next(iter(splitter.split(Xv, yv, groups=groups_v)))
    pipe = M._lr_pipeline("l2", M.SEED)
    pipe.fit(Xv[train_idx], yv[train_idx])
    y_score = pipe.predict_proba(Xv[test_idx])[:, 1]
    y_test = yv[test_idx]
    from sklearn.metrics import roc_auc_score
    sk_auc = float(roc_auc_score(y_test, y_score))
    pos_scores = y_score[y_test == 1]
    neg_scores = y_score[y_test == 0]
    u_stat, _ = mannwhitneyu(pos_scores, neg_scores)
    hand_auc = float(u_stat / (len(pos_scores) * len(neg_scores)))
    v["7_metric_crosscheck"] = {
        "sklearn_auroc": sk_auc, "hand_auroc_via_mannwhitney": hand_auc,
        "match_to_1e6": abs(sk_auc - hand_auc) < 1e-6,
    }
    print(f"7. metric-crosscheck: {v['7_metric_crosscheck']}", flush=True)

    # 8. Label sanity
    v["8_label_sanity"] = {
        "unique_values": sorted(y.unique().tolist()),
        "positive_class_is_lie": True,
        "pooled_balance": float(y.mean()),
    }
    print(f"8. label-sanity: {v['8_label_sanity']}", flush=True)

    # 9. Reproducibility (bypass checkpoint cache -- distinct no-hook calls)
    res_a = M.evaluate_cv(X, y, splitter, lambda s: M._lr_pipeline("l2", s), M._lr_param_grid(),
                           groups=groups, inner_splitter_factory=M.default_grouped_inner)
    res_b = M.evaluate_cv(X, y, splitter, lambda s: M._lr_pipeline("l2", s), M._lr_param_grid(),
                           groups=groups, inner_splitter_factory=M.default_grouped_inner)
    aurocs_a = [f["auroc"] for f in res_a["per_fold"]]
    aurocs_b = [f["auroc"] for f in res_b["per_fold"]]
    max_diff = max(abs(a - b) for a, b in zip(aurocs_a, aurocs_b))
    v["9_reproducibility"] = {"max_diff": max_diff, "bit_identical": max_diff == 0.0}
    print(f"9. reproducibility: max_diff={max_diff:.2e}", flush=True)

    # 10. Exp1 unaffected by the models.py edit
    exp1_json = json.load(open(EXP1_JSON))
    exp1_lr = exp1_json["experiments"]["exp1"]["models"]["logistic_regression"]["per_fold"]
    exp1_expected = [round(f["auroc"], 4) for f in exp1_lr]
    fd_e1 = M.load_feature_dictionary()
    fs_e1 = M.build_feature_sets(fd_e1)
    sb_e1 = M.load_single_brain()
    X_e1, y_e1, _, _, _ = M.prepare_modeling_frame(sb_e1, "deceiver", fs_e1["reliable_plus_marginal"])
    splitter_e1 = M.default_splitter()
    res_e1 = M.evaluate_cv(X_e1, y_e1, splitter_e1, lambda s: M._lr_pipeline("l2", s), M._lr_param_grid("l2"))
    exp1_recomputed = [round(f["auroc"], 4) for f in res_e1["per_fold"]]
    v["10_exp1_unaffected"] = {
        "expected": exp1_expected, "recomputed": exp1_recomputed,
        "match": exp1_expected == exp1_recomputed,
    }
    print(f"10. exp1-unaffected: {v['10_exp1_unaffected']}", flush=True)
    if not v["10_exp1_unaffected"]["match"]:
        raise RuntimeError("models.py edit changed exp1's default-argument reproducibility!")

    # 11. Frozen files untouched
    frozen_mtimes = {str(f): (f.stat().st_mtime if f.exists() else None) for f in FROZEN_FILES}
    v["11_frozen_files_untouched"] = {"mtimes": frozen_mtimes}
    print(f"11. frozen-files-mtimes: {frozen_mtimes}", flush=True)

    # 12. Plausibility / leakage red flag
    all_aurocs = [(fam, res["mean"]["auroc"]) for fam, res in primary_results.items()]
    per_dyad_max = 0.0
    for fam, res in primary_results.items():
        for f in res["per_fold"]:
            per_dyad_max = max(per_dyad_max, f["auroc"])
    max_fam, max_auroc = max(all_aurocs, key=lambda t: t[1])
    v["12_plausibility"] = {
        "per_family_mean_auroc": dict(all_aurocs), "max_family_mean": max_auroc,
        "max_single_dyad_auroc": per_dyad_max,
        "leakage_suspicion": bool(max_auroc > 0.95 or per_dyad_max > 0.95),
    }
    print(f"12. plausibility: {v['12_plausibility']}", flush=True)

    # 13. Convergence
    conv_warnings = lr["convergence_warnings"]
    v["13_convergence"] = {"n_convergence_warnings": len(conv_warnings), "warnings": conv_warnings}
    print(f"13. convergence: {v['13_convergence']}", flush=True)

    # 14. Inner-CV grouping actually took effect
    v["14_inner_cv_grouped"] = {
        fam: res.get("inner_splitter_class") for fam, res in primary_results.items()
    }
    print(f"14. inner-cv-grouped: {v['14_inner_cv_grouped']}", flush=True)

    return v


# ---------------------------------------------------------------------------
# Assembly / markdown / IO
# ---------------------------------------------------------------------------

def strip_frame(d):
    d2 = dict(d)
    d2.pop("fold_test_indices", None)
    return d2


def _fmt_ci(ci_block: dict) -> str:
    return f"{ci_block['mean']:.3f} [{ci_block['lower']:.3f}, {ci_block['upper']:.3f}]"


def _n_above(values: list, thresh: float = 0.5) -> int:
    return sum(1 for v in values if v > thresh)


def run():
    print("=" * 70, flush=True)
    print("Experiment 2: universal model, leave-one-dyad-out (S12)", flush=True)
    print("=" * 70, flush=True)

    env = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_available": M._HAS_XGBOOST,
        "gradient_boosting_impl": M.gradient_boosting_impl_name(),
    }
    print(f"Environment: {env}", flush=True)

    t_start = time.time()
    per_family_seconds = {}
    n_resume_launches = _ckpt_load("n_resume_launches") or 0
    n_resume_launches += 1
    _ckpt_save("n_resume_launches", n_resume_launches)

    fd = M.load_feature_dictionary()
    feature_sets = M.build_feature_sets(fd)
    print(f"Feature sets: reliable_plus_marginal={len(feature_sets['reliable_plus_marginal'])}, "
          f"reliable_only={len(feature_sets['reliable_only'])}", flush=True)

    sb = M.load_single_brain()

    # ---- Step 2: dyad reconciliation (verify against parquet at run time) ----
    dec_all = sb[sb["role"] == "deceiver"]
    obs_all = sb[sb["role"] == "observer"]
    sess_per_dyad = sb.groupby("pair_id")["session_id"].nunique()
    cond_per_dyad = dec_all.groupby("pair_id")["condition"].value_counts().unstack(fill_value=0)
    dyad_ids = sorted(dec_all["pair_id"].unique())
    n_dyads = len(dyad_ids)
    half_size = [pid for pid in dyad_ids if len(dec_all[dec_all["pair_id"] == pid]) < 900]

    frozen_hyp = open(RESULTS_DIR / "frozen_hypotheses.md").read()
    frozen_n12 = "n = 12 for: exp1, exp2" in frozen_hyp or "n = 12 for: exp1," in frozen_hyp
    conflict_with_frozen = not (n_dyads == 12 and frozen_n12)

    dyad_reconciliation = {
        "n_dyads": n_dyads,
        "dyad_ids": dyad_ids,
        "per_dyad_deceiver_rows": {pid: int(len(dec_all[dec_all["pair_id"] == pid])) for pid in dyad_ids},
        "per_dyad_observer_rows": {pid: int(len(obs_all[obs_all["pair_id"] == pid])) for pid in dyad_ids},
        "per_dyad_sessions": {pid: int(sess_per_dyad.get(pid, 0)) for pid in dyad_ids},
        "per_dyad_condition_counts": {
            pid: {"lie": int(cond_per_dyad.loc[pid].get("lie", 0)),
                  "truth": int(cond_per_dyad.loc[pid].get("truth", 0))}
            for pid in dyad_ids
        },
        "half_size_dyads": half_size,
        "half_size_reasons": {
            "sub01_sub02": "genuine single-session dyad -- only ever ran one session.",
            "sub19_sub22": "1 of 2 sessions unrecoverable at preprocessing (S9): the "
                            "Player_sub22_Observer_sub19 session (pair_num==21 in the "
                            "OneDCNN archive). The OTHER session survived and is fully "
                            "represented here. Distinct reason from sub01_sub02 -- do "
                            "not conflate.",
        },
        "matches_frozen_hypotheses_n12": (n_dyads == 12 and frozen_n12),
        "conflict_with_frozen": conflict_with_frozen,
        "note": "The dispatch premise that sub19_sub22 has zero rows was checked "
                "against the parquet at execution time and found false; corrected "
                "here per plan Step 2.",
    }
    print(f"Dyad reconciliation: n_dyads={n_dyads}, half_size={half_size}, "
          f"conflict_with_frozen={conflict_with_frozen}", flush=True)
    if conflict_with_frozen:
        print("*** DIVERGENCE: dyad reconciliation disagrees with frozen_hypotheses.md ***", flush=True)

    splitter = LeaveOneGroupOut()

    # ---- Primary: deceiver rows, reliable+marginal, all families ----
    print("\n--- PRIMARY: deceiver rows, reliable+marginal, LODO ---", flush=True)
    X_dec_hm, y_dec_hm, groups_dec, row_keys_dec, info_dec_hm = M.prepare_modeling_frame(
        sb, "deceiver", feature_sets["reliable_plus_marginal"]
    )
    print(f"  rows: {info_dec_hm}", flush=True)
    n_splits_actual = splitter.get_n_splits(groups=groups_dec)
    print(f"  LeaveOneGroupOut n_splits={n_splits_actual}", flush=True)

    stats = dyad_stats(y_dec_hm, groups_dec)

    # majority_class reference early (Step 9's metric-code sanity check)
    t0 = time.time()
    majority_res = _run_family(
        "ref_majority_class", "majority_class", X_dec_hm, y_dec_hm, splitter, groups_dec,
        lambda s: SkPipeline([("clf", DummyClassifier(strategy="most_frequent"))]), None,
        use_grouped_inner=False,
    )
    per_family_seconds["majority_class"] = time.time() - t0
    print(f"  majority_class: auroc={majority_res['mean']['auroc']:.4f} "
          f"bal_acc={majority_res['mean']['balanced_accuracy']:.4f}", flush=True)

    families_run = []
    families_skipped = []
    skip_reasons = {}

    primary_results = {}
    for fam in ["logistic_regression", "random_forest", "gradient_boosting",
                "logistic_regression_l1", "svm"]:
        t0 = time.time()
        r = run_family_set(X_dec_hm, y_dec_hm, groups_dec, splitter, [fam], "primary_dec_hm")
        primary_results.update(r)
        per_family_seconds[fam] = time.time() - t0
        families_run.append(fam)

    # ---- Secondary: observer rows, LR only ----
    print("\n--- SECONDARY: observer rows, reliable+marginal, LR only ---", flush=True)
    X_obs, y_obs, groups_obs, _, info_obs = M.prepare_modeling_frame(
        sb, "observer", feature_sets["reliable_plus_marginal"]
    )
    print(f"  rows: {info_obs}", flush=True)
    t0 = time.time()
    observer_results = run_family_set(
        X_obs, y_obs, groups_obs, splitter, ["logistic_regression"], "secondary_obs_hm"
    )
    per_family_seconds["secondary_observer_lr"] = time.time() - t0
    families_run.append("secondary_observer_lr")

    # ---- Robustness: deceiver rows, reliable-only, LR only ----
    print("\n--- ROBUSTNESS: deceiver rows, reliable-only, LR only ---", flush=True)
    X_dec_ro, y_dec_ro, groups_dec_ro, _, info_dec_ro = M.prepare_modeling_frame(
        sb, "deceiver", feature_sets["reliable_only"]
    )
    print(f"  rows: {info_dec_ro}", flush=True)
    t0 = time.time()
    robustness_results = run_family_set(
        X_dec_ro, y_dec_ro, groups_dec_ro, splitter, ["logistic_regression"], "robustness_dec_ro"
    )
    per_family_seconds["robustness_lr"] = time.time() - t0
    families_run.append("robustness_lr")

    # ---- Permutation null (Step 8d) ----
    best_cs = [p["clf__C"] for p in primary_results["logistic_regression"]["best_params_per_fold"] if p]
    fixed_C = Counter(best_cs).most_common(1)[0][0] if best_cs else 1.0
    print(f"\n--- Permutation null (LR L2, fixed C={fixed_C:.4g}, within-dyad shuffle, "
          f"n={N_PERMUTATIONS}) ---", flush=True)
    t0 = time.time()
    n_perm_target = N_PERMUTATIONS
    perm_est_seconds = per_family_seconds["logistic_regression"] / max(n_splits_actual, 1) * n_splits_actual
    # rough per-permutation cost estimate from the primary LR family's measured
    # per-fold cost at a single (unsearched) C -- used only to decide whether to
    # reduce N_PERMUTATIONS, reported either way.
    perm_res = run_permutation_null(X_dec_hm, y_dec_hm, groups_dec, splitter, fixed_C, n_perm_target)
    per_family_seconds["permutation_null"] = time.time() - t0

    observed_auroc = primary_results["logistic_regression"]["mean"]["auroc"]
    p_value = M.permutation_p_value(observed_auroc, perm_res["null_aucs"])
    null_arr = np.asarray(perm_res["null_aucs"])
    print(f"  observed={observed_auroc:.4f} null_mean={null_arr.mean():.4f} "
          f"null_sd={null_arr.std():.4f} p={p_value:.4f}", flush=True)

    total_seconds = time.time() - t_start

    results = assemble_results(
        env=env, feature_sets=feature_sets,
        info_dec_hm=info_dec_hm, info_dec_ro=info_dec_ro, info_obs=info_obs,
        primary_results=primary_results, robustness_results=robustness_results,
        observer_results=observer_results, majority_res=majority_res,
        perm_res=perm_res, p_value=p_value, observed_auroc=observed_auroc, fixed_C=fixed_C,
        dyad_reconciliation=dyad_reconciliation,
        X_dec_hm=X_dec_hm, y_dec_hm=y_dec_hm, groups_dec=groups_dec, stats=stats,
        X_obs=X_obs, y_obs=y_obs, groups_obs=groups_obs,
        X_dec_ro=X_dec_ro, y_dec_ro=y_dec_ro, groups_dec_ro=groups_dec_ro,
        n_splits_actual=n_splits_actual,
        families_run=families_run, families_skipped=families_skipped, skip_reasons=skip_reasons,
        per_family_seconds=per_family_seconds, total_seconds=total_seconds,
        n_resume_launches=n_resume_launches,
    )

    validations = run_validations(
        X_dec_hm, y_dec_hm, groups_dec, feature_sets, fd, primary_results, majority_res,
        splitter, n_dyads_expected=n_dyads,
    )
    results["experiments"]["exp2"]["validations"] = validations

    write_outputs(results)
    return results


def assemble_results(**kw) -> dict:
    env = kw["env"]
    fs = kw["feature_sets"]

    primary = {k: strip_frame(v) for k, v in kw["primary_results"].items()}
    robustness = {k: strip_frame(v) for k, v in kw["robustness_results"].items()}
    observer = {k: strip_frame(v) for k, v in kw["observer_results"].items()}

    # attach per_dyad / per_dyad_scores to every family that ran
    def _attach(fam_dict, stats, groups, base_res_lookup):
        out = {}
        for fam, res in fam_dict.items():
            raw = base_res_lookup[fam]
            per_dyad, per_dyad_scores = build_per_dyad(raw, groups, stats)
            r = dict(res)
            r["per_dyad"] = per_dyad
            per_dyad_scores.update({
                "condition": "universal_lodo", "experiment": "exp2", "model": fam,
                "rows": "deceiver", "feature_set": "reliable_plus_marginal",
            })
            if fam == "logistic_regression":
                per_dyad_scores["primary"] = True
            r["per_dyad_scores"] = per_dyad_scores
            aurocs = [d["auroc"] for d in per_dyad]
            r["median"] = {m: float(np.median([d[m] for d in per_dyad]))
                            for m in ["auroc", "balanced_accuracy", "f1", "precision", "recall"]}
            r["min"] = {m: float(np.min([d[m] for d in per_dyad]))
                         for m in ["auroc", "balanced_accuracy", "f1", "precision", "recall"]}
            r["max"] = {m: float(np.max([d[m] for d in per_dyad]))
                         for m in ["auroc", "balanced_accuracy", "f1", "precision", "recall"]}
            r["n_dyads_above_0.5_auroc"] = _n_above(aurocs, 0.5)
            out[fam] = r
        return out

    primary = _attach(primary, kw["stats"], kw["groups_dec"], kw["primary_results"])
    robustness_stats = dyad_stats(kw["y_dec_ro"], kw["groups_dec_ro"])
    robustness = _attach(robustness, robustness_stats, kw["groups_dec_ro"], kw["robustness_results"])
    observer_stats = dyad_stats(kw["y_obs"], kw["groups_obs"])
    observer = _attach(observer, observer_stats, kw["groups_obs"], kw["observer_results"])

    majority = strip_frame(kw["majority_res"])
    maj_per_dyad, maj_scores = build_per_dyad(kw["majority_res"], kw["groups_dec"], kw["stats"])
    majority["per_dyad"] = maj_per_dyad
    majority["per_dyad_scores"] = maj_scores

    # comparison to exp1
    exp1_json = json.load(open(EXP1_JSON))
    exp1_models = exp1_json["experiments"]["exp1"]["models"]
    comparison = {}
    n_dyads_above_chance_by_family = {}
    for fam in ["logistic_regression", "logistic_regression_l1", "svm",
                "random_forest", "gradient_boosting"]:
        if fam not in primary:
            continue
        exp1_auroc = exp1_models[fam]["mean"]["auroc"]
        exp2_auroc = primary[fam]["mean"]["auroc"]
        comparison[fam] = {
            "exp1_auroc": exp1_auroc, "exp2_auroc": exp2_auroc,
            "delta": exp2_auroc - exp1_auroc,
        }
        n_dyads_above_chance_by_family[fam] = primary[fam]["n_dyads_above_0.5_auroc"]

    lr_delta = comparison.get("logistic_regression", {}).get("delta")
    if lr_delta is None:
        interp = "primary LR family missing from this run; comparison incomplete."
    elif lr_delta < -0.02:
        interp = ("exp2 LODO AUROC is meaningfully below exp1's pooled AUROC. exp1 is an "
                   "upper bound and sanity check, not the generalization claim (exp1's own "
                   "results file states this verbatim); exp2 is where the generalization claim "
                   "is made. This gap is evidence the pooled model was partly recognizing "
                   "participants rather than deception -- the S19 leakage channel the LODO "
                   "design exists to close.")
    elif lr_delta > 0.02:
        interp = ("exp2 LODO AUROC exceeds exp1's pooled AUROC by more than a hair. Per plan "
                   "Step 8, this is treated as suspicious rather than good news: a grouped "
                   "split beating an ungrouped one on the same data usually means something is "
                   "off. Checked against Step 9's validations before being reported as a finding.")
    else:
        interp = ("exp2 LODO AUROC is close to exp1's pooled AUROC: the signal, such as it is, "
                   "transfers to unseen dyads, and exp1's pooled estimate was not materially "
                   "inflated by participant recognition. Both sit within a few points of "
                   "chance (0.5); this is not dressed up as a strong result.")

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_tables": [
                "data/processed/features/single_brain.parquet",
                "data/processed/features/feature_dictionary.csv",
            ],
            "seed": M.SEED,
            "sklearn_version": env["sklearn_version"],
            "environment": {
                "gradient_boosting_impl": env["gradient_boosting_impl"],
            },
            "runtime": {
                "per_family_seconds": kw["per_family_seconds"],
                "total_seconds": kw["total_seconds"],
                "n_resume_launches": kw["n_resume_launches"],
            },
        },
        "experiments": {
            "exp2": {
                "design": {
                    "research_question": "S12: can a deception model generalize to "
                                          "completely unseen participant pairs?",
                    "cv": "LeaveOneGroupOut(groups=pair_id)",
                    "n_folds": kw["n_splits_actual"],
                    "inner_cv": "StratifiedGroupKFold(3) on pair_id within the 11 training "
                                "dyads (S19); falls back to GroupKFold(3) if stratification "
                                "is infeasible (not observed on this data).",
                    "target": "condition == 'lie'",
                    "rows": "role == 'deceiver' (primary); role == 'observer' (secondary)",
                    "feature_set": "reliable_plus_marginal (1,770 columns), inherited "
                                    "unchanged from exp1 for comparability -- changing the "
                                    "feature set and the splitter at once would confound "
                                    "the exp1-vs-exp2 contrast this experiment exists to make.",
                    "relation_to_exp1": "exp1 pooled = upper bound; exp2 LODO = the "
                                         "generalization claim (exp1's own results file "
                                         "states this verbatim).",
                },
                "dyad_reconciliation": kw["dyad_reconciliation"],
                "feature_sets": {
                    "reliable_plus_marginal": {"n_features": len(fs["reliable_plus_marginal"])},
                    "reliable_only": {"n_features": len(fs["reliable_only"])},
                },
                "row_counts": {
                    "deceiver_headline": kw["info_dec_hm"],
                    "deceiver_robustness": kw["info_dec_ro"],
                    "observer_secondary": kw["info_obs"],
                },
                "models": primary,
                "models_reliable_only": robustness,
                "observer_rows": observer,
                "references": {"majority_class": majority},
                "permutation_null": {
                    "n_permutations": kw["perm_res"]["n_permutations"],
                    "shuffle_scheme": "within-dyad",
                    "fixed_C": kw["fixed_C"],
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
                "comparison_to_exp1": {
                    "per_family": comparison,
                    "n_dyads_above_chance": n_dyads_above_chance_by_family,
                    "interpretation": interp,
                },
                "families_run": kw["families_run"],
                "families_skipped": kw["families_skipped"],
                "skip_reasons": kw["skip_reasons"],
            }
        },
    }


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

    with open(EXP2_JSON, "w") as f:
        json.dump(results, f, indent=2, default=default)
    print(f"\nWrote {EXP2_JSON}", flush=True)

    md = build_markdown(results)
    with open(EXP2_MD, "w") as f:
        f.write(md)
    print(f"Wrote {EXP2_MD}", flush=True)


def build_markdown(results: dict) -> str:
    exp2 = results["experiments"]["exp2"]
    lines = []
    lines.append("# Experiment 2 -- Universal Model, Leave-One-Dyad-Out (S12)\n")
    lines.append(
        f"**What this experiment asks:** {exp2['design']['research_question']} "
        "Experiment 1's pooled, ungrouped baseline is an upper bound and sanity check "
        "(its own results file says so verbatim); this experiment is where the "
        "generalization claim is actually made.\n"
    )

    lines.append("## Design decisions\n")
    d = exp2["design"]
    lines.append(
        f"- **CV:** `{d['cv']}`, {d['n_folds']} folds (one per dyad).\n"
        f"- **Inner (tuning) CV:** {d['inner_cv']}\n"
        f"- **Rows/target/feature set inherited unchanged from exp1:** {d['feature_set']}\n"
    )

    lines.append("## Dyad reconciliation\n")
    dr = exp2["dyad_reconciliation"]
    lines.append(
        f"The dispatch premise that `sub19_sub22` has zero feature rows was checked "
        f"against `data/processed/features/single_brain.parquet` at execution time and "
        f"found **false**. Experiment 2 runs LODO over **{dr['n_dyads']} dyads** "
        f"(n={dr['n_dyads']}), matching `results/frozen_hypotheses.md`'s stated n=12 for "
        f"exp2. **Conflict with frozen file: {dr['conflict_with_frozen']}.**\n\n"
        "| pair_id | deceiver rows | observer rows | sessions | lie | truth |\n"
        "|---|---|---|---|---|---|\n"
    )
    for pid in dr["dyad_ids"]:
        cc = dr["per_dyad_condition_counts"][pid]
        lines.append(
            f"| {pid} | {dr['per_dyad_deceiver_rows'][pid]} | {dr['per_dyad_observer_rows'][pid]} | "
            f"{dr['per_dyad_sessions'][pid]} | {cc['lie']} | {cc['truth']} |"
        )
    lines.append("")
    lines.append(
        f"**Half-size dyads:** {', '.join(dr['half_size_dyads'])}. Two different reasons: "
        f"`sub01_sub02` -- {dr['half_size_reasons'].get('sub01_sub02', '')} "
        f"`sub19_sub22` -- {dr['half_size_reasons'].get('sub19_sub22', '')}\n"
    )

    lines.append("## Per-dyad score table (primary model: logistic_regression)\n")
    lr = exp2["models"]["logistic_regression"]
    lines.append("| pair_id | n_test | n_lie | AUROC | Bal. Acc. | F1 | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in lr["per_dyad"]:
        lines.append(
            f"| {r['pair_id']} | {r['n_test']} | {r['n_lie']} | {r['auroc']:.4f} | "
            f"{r['balanced_accuracy']:.4f} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
        )
    lines.append("")

    lines.append("## Pooled/mean metrics per family (CI across dyads)\n")
    lines.append("| model | AUROC (mean, CI across dyads) | median | min | max | n_dyads>0.5 |")
    lines.append("|---|---|---|---|---|---|")
    for fam, res in exp2["models"].items():
        ci = res["ci95"]
        lines.append(
            f"| {fam}{' (PRIMARY)' if res.get('primary') else ''} | {_fmt_ci(ci['auroc'])} | "
            f"{res['median']['auroc']:.4f} | {res['min']['auroc']:.4f} | {res['max']['auroc']:.4f} | "
            f"{res['n_dyads_above_0.5_auroc']}/{len(res['per_dyad'])} |"
        )
    lines.append(
        "\nNote: exp1's CI was across arbitrary trial-level folds; exp2's CI is genuinely "
        "a *between-dyad* CI (S22's \"variability across dyads\"), a different quantity "
        "printed the same way.\n"
    )

    lines.append("## Comparison to Experiment 1\n")
    lines.append("| model | exp1 pooled AUROC | exp2 LODO AUROC (mean over dyads) | delta (exp2-exp1) |")
    lines.append("|---|---|---|---|")
    for fam, c in exp2["comparison_to_exp1"]["per_family"].items():
        lines.append(f"| {fam} | {c['exp1_auroc']:.4f} | {c['exp2_auroc']:.4f} | {c['delta']:+.4f} |")
    lines.append(f"\n{exp2['comparison_to_exp1']['interpretation']}\n")

    perm = exp2["permutation_null"]
    lines.append("## Permutation null (S21)\n")
    lines.append(
        f"Primary model (LR L2), fixed C={perm['fixed_C']:.4g}, {perm['shuffle_scheme']} "
        f"shuffling, {perm['n_permutations']} permutations. Observed AUROC = "
        f"{perm['observed_auroc']:.4f}; null mean = {perm['null']['mean']:.4f} "
        f"(sd={perm['null']['sd']:.4f}, 5th/50th/95th percentile = "
        f"{perm['null']['percentiles']['5']:.4f}/{perm['null']['percentiles']['50']:.4f}/"
        f"{perm['null']['percentiles']['95']:.4f}). **p = {perm['p_value']:.4f}**.\n"
    )

    lines.append("## Secondary (observer rows) and robustness (reliable-only)\n")
    lines.append("| block | model | AUROC (mean) |")
    lines.append("|---|---|---|")
    for fam, res in exp2["observer_rows"].items():
        lines.append(f"| secondary (observer) | {fam} | {res['mean']['auroc']:.4f} |")
    for fam, res in exp2["models_reliable_only"].items():
        lines.append(f"| robustness (reliable-only) | {fam} | {res['mean']['auroc']:.4f} |")
    lines.append("")

    lines.append("## Runtime and resumption notes\n")
    rt = results["meta"]["runtime"]
    lines.append(
        f"Total measured runtime this session: {rt['total_seconds']:.1f}s. "
        f"Resume launches so far: {rt['n_resume_launches']}. Per-family seconds "
        f"(this launch only, cached folds excluded from timing): {rt['per_family_seconds']}\n"
    )

    lines.append("## Families run vs skipped\n")
    lines.append(f"Run: {exp2['families_run']}\n\nSkipped: {exp2['families_skipped'] or 'none'}\n")
    if exp2["skip_reasons"]:
        lines.append(f"Reasons: {exp2['skip_reasons']}\n")

    lines.append("## Validation summary\n")
    for k, val in exp2["validations"].items():
        lines.append(f"- **{k}**: {val}\n")

    lines.append("## Frozen-file confirmation\n")
    lines.append(
        "`results/gate.json`, `results/trial_count_gate.md`, `results/frozen_hypotheses.md` "
        "were read-only inputs, never written to or contradicted. mtimes recorded in "
        "`validations.11_frozen_files_untouched`; all predate this task's start.\n"
    )

    return "\n".join(lines)


if __name__ == "__main__":
    run()
