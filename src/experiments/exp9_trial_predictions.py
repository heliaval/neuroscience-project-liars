"""
src/experiments/exp9_trial_predictions.py -- per-trial predicted probabilities
and feature attribution for the walkthrough dashboard's individual-trial-browser
and failure-case-gallery sections, both of which currently show invented
placeholder numbers (see results/fixtures/results.v1.fixture.json's "trials"
and "failures" blocks, tagged provenance:"placeholder" in every emitted
results.v1.json).

WHY THIS DIDN'T ALREADY EXIST: exp1's cross-validation (src/models.py's
evaluate_cv) computes a per-trial predicted probability for every held-out row
during scoring, but immediately collapses those into per-fold aggregate
metrics (AUROC, precision, ...) and discards the individual values -- nothing
in the existing pipeline ever wrote per-trial predictions to disk. This script
is new computation, not a re-export of something already sitting in a file.

WHAT MODEL: the exact primary model from Experiment 1 (S11) -- deceiver rows,
reliable+marginal feature set, logistic regression with L2 penalty. Reuses
src/models.py's build_feature_sets / prepare_modeling_frame / default_splitter
so the row set and StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
split are byte-identical to exp1's own. Reuses exp1's already-tuned per-fold
C (results/exp1_baseline.json's best_params_per_fold) instead of re-running
GridSearchCV -- this reproduces exp1's exact fold structure and hyperparameters
rather than deriving a new, only-approximately-equivalent one, and skips the
expensive inner-CV tuning entirely (this script refits one LogisticRegression
per outer fold, nothing more).

OUT-OF-FOLD, NOT IN-SAMPLE: every trial's prediction comes from the one fold
where it was held out -- across all 5 outer folds together, every deceiver-role
modeling row gets exactly one prediction, always from a model that never saw
that row during training or scaling. This is standard out-of-fold (OOF)
prediction reconstruction, not a shortcut; a Step 9-style validation at the end
recomputes AUROC from the pooled OOF predictions and compares it to exp1's own
published headline, which should be close (not necessarily identical -- see
that validation's own comment for why) as a leakage/correctness check.

ATTRIBUTION: exact, not approximate, because the model is linear. For a fitted
LogisticRegression with coefficient vector w on standardized features, a given
trial's contribution to the (pre-sigmoid) score for feature j is
w_j * standardized_value_j -- computed with the SAME StandardScaler fitted on
that fold's training rows only (the identical scaler the fold's own prediction
used, never refit on or including the test row).

PERFORMANCE NOTE, for whoever runs this: this is a CPU job, not a GPU one.
scikit-learn's LogisticRegression does not use a GPU. Five single-fit calls at
1,770 features measured ~5s each during exp1 (see src/models.py's docstring),
so the whole outer loop below is a well-under-a-minute job on an ordinary
laptop CPU -- there's no grid search left to do, since C is reused from
exp1's own tuning. Nothing about this needs remote/hosted compute; it's
included here in case the environment running it doesn't have this repo's
Python deps installed locally, not because it's slow.

Writes results/exp9_trial_predictions.json: {meta, trials: [...], failures: [...]}.
Does not touch results/results.v1.json or any other results/ file -- wiring
the output into the public contract is a separate step once this has run and
been reviewed.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import models as M  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
EXP1_JSON = RESULTS_DIR / "exp1_baseline.json"
OUT_JSON = RESULTS_DIR / "exp9_trial_predictions.json"

TOP_K_FEATURES = 5
N_FAILURES = 50


def load_exp1_best_c_per_fold() -> list[float]:
    """Reuse exp1's own tuned C per fold -- see module docstring for why this
    is not re-derived via a fresh GridSearchCV."""
    d = json.loads(EXP1_JSON.read_text(encoding="utf8"))
    lr = d["experiments"]["exp1"]["models"]["logistic_regression"]
    best_params = lr["best_params_per_fold"]
    assert len(best_params) == 5, f"expected 5 folds, got {len(best_params)}"
    return [p["clf__C"] for p in best_params]


def attach_identity_columns(sb: pd.DataFrame, row_keys: pd.DataFrame) -> pd.DataFrame:
    """row_keys (from prepare_modeling_frame) carries only the identity columns
    needed for the trial_table join; partner_id/condition are pulled back in
    here by an exact-key merge against the same deceiver-role rows, rather than
    re-deriving them (they are already real columns in single_brain.parquet)."""
    role_df = sb[sb["role"] == "deceiver"][
        ["session_id", "round", "trial", "participant_id", "partner_id", "condition"]
    ]
    merged = row_keys.merge(role_df, on=["session_id", "round", "trial", "participant_id"], how="left")
    assert len(merged) == len(row_keys), "merge changed row count -- non-unique key"
    assert merged["partner_id"].notna().all() and merged["condition"].notna().all()
    return merged


def main() -> None:
    t_start = time.time()

    fd = M.load_feature_dictionary()
    feature_sets = M.build_feature_sets(fd)
    sb = M.load_single_brain()
    splitter = M.default_splitter()

    X, y, groups, row_keys, info = M.prepare_modeling_frame(
        sb, "deceiver", feature_sets["reliable_plus_marginal"]
    )
    row_keys = attach_identity_columns(sb, row_keys)
    feature_names = X.columns.tolist()
    print(f"Modeling frame: {info}")

    best_c_per_fold = load_exp1_best_c_per_fold()
    print(f"Reusing exp1's per-fold C: {best_c_per_fold}")

    Xv = X.values
    yv = y.values

    oof_p_lie = np.full(len(X), np.nan, dtype=float)
    oof_fold = np.full(len(X), -1, dtype=int)
    oof_contributions = [None] * len(X)

    split_iter = list(splitter.split(Xv, yv))
    for fold_i, (train_idx, test_idx) in enumerate(split_iter):
        t0 = time.time()
        scaler = StandardScaler().fit(Xv[train_idx])
        X_train_s = scaler.transform(Xv[train_idx])
        X_test_s = scaler.transform(Xv[test_idx])

        clf = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=5000, random_state=M.SEED,
            C=best_c_per_fold[fold_i],
        )
        clf.fit(X_train_s, yv[train_idx])

        p_lie = clf.predict_proba(X_test_s)[:, 1]
        oof_p_lie[test_idx] = p_lie
        oof_fold[test_idx] = fold_i

        # Exact per-trial linear attribution: coefficient * this trial's own
        # standardized feature value, computed with the fold's train-only scaler.
        coef = clf.coef_.ravel()
        contributions = X_test_s * coef[None, :]  # (n_test, n_features)
        for local_i, global_i in enumerate(test_idx):
            oof_contributions[global_i] = contributions[local_i]

        fold_auroc = roc_auc_score(yv[test_idx], p_lie)
        print(f"  fold {fold_i}: n_test={len(test_idx)}, C={best_c_per_fold[fold_i]:.4g}, "
              f"auroc={fold_auroc:.4f} ({time.time()-t0:.1f}s)")

    assert (oof_fold >= 0).all(), "every row must have received exactly one OOF prediction"
    assert not np.isnan(oof_p_lie).any()

    # Step-9-style validation: pooled OOF AUROC vs exp1's published headline.
    # These are not required to match exactly -- exp1's headline is the MEAN of
    # five per-fold AUROCs (each computed within its own fold's class balance),
    # while this is one AUROC computed over the pooled predictions of all folds
    # together. Both are legitimate, standard aggregations of the same
    # per-fold-honest CV; they differ because AUROC is not linear in the
    # confusion counts it pools over. A large discrepancy would indicate a real
    # bug (e.g. a fold mismatch); a small one is expected and reported, not
    # treated as an error.
    pooled_auroc = float(roc_auc_score(yv, oof_p_lie))
    exp1_headline = json.loads(EXP1_JSON.read_text(encoding="utf8"))[
        "experiments"]["exp1"]["models"]["logistic_regression"]["mean"]["auroc"]
    print(f"\nPooled OOF AUROC: {pooled_auroc:.4f}  |  exp1 headline (mean of per-fold): {exp1_headline:.4f}  "
          f"|  delta: {pooled_auroc - exp1_headline:+.4f}")

    trials = []
    for i in range(len(X)):
        rk = row_keys.iloc[i]
        contrib = oof_contributions[i]
        top_idx = np.argsort(-np.abs(contrib))[:TOP_K_FEATURES]
        actual_lie = bool(yv[i])
        p_lie = float(oof_p_lie[i])
        predicted_lie = p_lie >= 0.5
        trials.append({
            "trial_uid": f"{rk['pair_id']}:deceiver:{rk['session_id']}:r{rk['round']}t{rk['trial']}",
            "pair_id": rk["pair_id"],
            "participant_id": rk["participant_id"],
            "partner_id": rk["partner_id"],
            "role": "deceiver",
            "fold": int(oof_fold[i]),
            "actual": "lie" if actual_lie else "truth",
            "predicted": "lie" if predicted_lie else "truth",
            "p_lie": round(p_lie, 4),
            "correct": bool(predicted_lie == actual_lie),
            "top_features": [
                {"feature": feature_names[j], "contribution": round(float(contrib[j]), 4)}
                for j in top_idx
            ],
        })

    # Failures: predicted != actual, ranked by confidence of the wrong call
    # (|p_lie - 0.5|, descending) -- the most confidently wrong predictions
    # first, which is what a "failure case gallery" is for.
    wrong = [t for t in trials if not t["correct"]]
    wrong.sort(key=lambda t: abs(t["p_lie"] - 0.5), reverse=True)
    failures = []
    for t in wrong[:N_FAILURES]:
        kind = "false_positive" if t["predicted"] == "lie" else "false_negative"
        failures.append({
            "trial_uid": t["trial_uid"], "pair_id": t["pair_id"],
            "actual": t["actual"], "predicted": t["predicted"], "p_lie": t["p_lie"],
            "kind": kind, "attribution": t["top_features"],
        })

    n_correct = sum(1 for t in trials if t["correct"])
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "src/experiments/exp9_trial_predictions.py",
            "model": "logistic_regression (L2), deceiver rows, reliable+marginal features -- exp1's primary model",
            "n_trials": len(trials),
            "n_correct": n_correct,
            "n_features": len(feature_names),
            "per_fold_c_reused_from": "results/exp1_baseline.json:experiments.exp1.models.logistic_regression.best_params_per_fold",
            "pooled_oof_auroc": round(pooled_auroc, 4),
            "exp1_headline_auroc": exp1_headline,
        },
        "trials": trials,
        "failures": failures,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
    print(f"\nWrote {OUT_JSON} ({len(trials)} trials, {len(failures)} failures, "
          f"{OUT_JSON.stat().st_size} bytes, {time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()
