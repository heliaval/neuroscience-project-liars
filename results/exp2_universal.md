# Experiment 2 -- Universal Model, Leave-One-Dyad-Out (S12)

**What this experiment asks:** S12: can a deception model generalize to completely unseen participant pairs? Experiment 1's pooled, ungrouped baseline is an upper bound and sanity check (its own results file says so verbatim); this experiment is where the generalization claim is actually made.

## Design decisions

- **CV:** `LeaveOneGroupOut(groups=pair_id)`, 12 folds (one per dyad).
- **Inner (tuning) CV:** StratifiedGroupKFold(3) on pair_id within the 11 training dyads (S19); falls back to GroupKFold(3) if stratification is infeasible (not observed on this data).
- **Rows/target/feature set inherited unchanged from exp1:** reliable_plus_marginal (1,770 columns), inherited unchanged from exp1 for comparability -- changing the feature set and the splitter at once would confound the exp1-vs-exp2 contrast this experiment exists to make.

## Dyad reconciliation

The dispatch premise that `sub19_sub22` has zero feature rows was checked against `data/processed/features/single_brain.parquet` at execution time and found **false**. Experiment 2 runs LODO over **12 dyads** (n=12), matching `results/frozen_hypotheses.md`'s stated n=12 for exp2. **Conflict with frozen file: False.**

| pair_id | deceiver rows | observer rows | sessions | lie | truth |
|---|---|---|---|---|---|

| sub01_sub02 | 484 | 484 | 1 | 264 | 220 |
| sub03_sub06 | 968 | 968 | 2 | 460 | 508 |
| sub04_sub05 | 968 | 968 | 2 | 442 | 526 |
| sub07_sub08 | 968 | 968 | 2 | 447 | 521 |
| sub09_sub10 | 968 | 968 | 2 | 471 | 497 |
| sub11_sub12 | 968 | 968 | 2 | 424 | 544 |
| sub13_sub14 | 968 | 968 | 2 | 459 | 509 |
| sub15_sub16 | 968 | 968 | 2 | 457 | 511 |
| sub17_sub18 | 968 | 968 | 2 | 399 | 569 |
| sub19_sub22 | 484 | 484 | 1 | 248 | 236 |
| sub20_sub21 | 968 | 968 | 2 | 489 | 479 |
| sub23_sub24 | 968 | 968 | 2 | 500 | 468 |

**Half-size dyads:** sub01_sub02, sub19_sub22. Two different reasons: `sub01_sub02` -- genuine single-session dyad -- only ever ran one session. `sub19_sub22` -- 1 of 2 sessions unrecoverable at preprocessing (S9): the Player_sub22_Observer_sub19 session (pair_num==21 in the OneDCNN archive). The OTHER session survived and is fully represented here. Distinct reason from sub01_sub02 -- do not conflate.

## Per-dyad score table (primary model: logistic_regression)

| pair_id | n_test | n_lie | AUROC | Bal. Acc. | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| sub01_sub02 | 484 | 264 | 0.4813 | 0.4875 | 0.4970 | 0.5325 | 0.4659 |
| sub03_sub06 | 968 | 460 | 0.5158 | 0.5176 | 0.4850 | 0.4944 | 0.4761 |
| sub04_sub05 | 968 | 442 | 0.5067 | 0.4958 | 0.4781 | 0.4525 | 0.5068 |
| sub07_sub08 | 968 | 447 | 0.5424 | 0.5296 | 0.5195 | 0.4901 | 0.5526 |
| sub09_sub10 | 968 | 471 | 0.5218 | 0.5268 | 0.3492 | 0.5450 | 0.2569 |
| sub11_sub12 | 968 | 424 | 0.5482 | 0.5279 | 0.5201 | 0.4624 | 0.5943 |
| sub13_sub14 | 968 | 459 | 0.4911 | 0.4776 | 0.4576 | 0.4513 | 0.4641 |
| sub15_sub16 | 968 | 457 | 0.5119 | 0.5037 | 0.3482 | 0.4789 | 0.2735 |
| sub17_sub18 | 968 | 399 | 0.4873 | 0.4928 | 0.4243 | 0.4045 | 0.4461 |
| sub19_sub22 | 484 | 248 | 0.5095 | 0.4977 | 0.5468 | 0.5105 | 0.5887 |
| sub20_sub21 | 966 | 489 | 0.5096 | 0.5067 | 0.5315 | 0.5123 | 0.5521 |
| sub23_sub24 | 964 | 500 | 0.5311 | 0.5272 | 0.5473 | 0.5446 | 0.5500 |

## Pooled/mean metrics per family (CI across dyads)

| model | AUROC (mean, CI across dyads) | median | min | max | n_dyads>0.5 |
|---|---|---|---|---|---|
| logistic_regression (PRIMARY) | 0.513 [0.500, 0.526] | 0.5107 | 0.4813 | 0.5482 | 9/12 |
| random_forest | 0.521 [0.500, 0.542] | 0.5199 | 0.4808 | 0.5951 | 8/12 |
| gradient_boosting | 0.504 [0.489, 0.520] | 0.5034 | 0.4694 | 0.5555 | 8/12 |
| logistic_regression_l1 | 0.505 [0.493, 0.518] | 0.5016 | 0.4769 | 0.5423 | 6/12 |
| svm | 0.513 [0.500, 0.526] | 0.5080 | 0.4876 | 0.5460 | 9/12 |

Note: exp1's CI was across arbitrary trial-level folds; exp2's CI is genuinely a *between-dyad* CI (S22's "variability across dyads"), a different quantity printed the same way.

## Comparison to Experiment 1

| model | exp1 pooled AUROC | exp2 LODO AUROC (mean over dyads) | delta (exp2-exp1) |
|---|---|---|---|
| logistic_regression | 0.5338 | 0.5131 | -0.0207 |
| logistic_regression_l1 | 0.5310 | 0.5052 | -0.0258 |
| svm | 0.5339 | 0.5133 | -0.0206 |
| random_forest | 0.5470 | 0.5210 | -0.0260 |
| gradient_boosting | 0.5315 | 0.5044 | -0.0270 |

exp2 LODO AUROC is meaningfully below exp1's pooled AUROC. exp1 is an upper bound and sanity check, not the generalization claim (exp1's own results file states this verbatim); exp2 is where the generalization claim is made. This gap is evidence the pooled model was partly recognizing participants rather than deception -- the S19 leakage channel the LODO design exists to close.

## Permutation null (S21)

Primary model (LR L2), fixed C=0.001, within-dyad shuffling, 200 permutations. Observed AUROC = 0.5131; null mean = 0.5004 (sd=0.0077, 5th/50th/95th percentile = 0.4871/0.5015/0.5117). **p = 0.0547**.

## Secondary (observer rows) and robustness (reliable-only)

| block | model | AUROC (mean) |
|---|---|---|
| secondary (observer) | logistic_regression | 0.5172 |
| robustness (reliable-only) | logistic_regression | 0.5124 |

## Runtime and resumption notes

Total measured runtime this session: 6.6s. Resume launches so far: 29. Per-family seconds (this launch only, cached folds excluded from timing): {'majority_class': 0.06630563735961914, 'logistic_regression': 0.09120750427246094, 'random_forest': 0.06874465942382812, 'gradient_boosting': 0.11534929275512695, 'logistic_regression_l1': 0.08625411987304688, 'svm': 0.1076653003692627, 'secondary_observer_lr': 0.14827299118041992, 'robustness_lr': 0.14993858337402344, 'permutation_null': 0.0033295154571533203}

## Families run vs skipped

Run: ['logistic_regression', 'random_forest', 'gradient_boosting', 'logistic_regression_l1', 'svm', 'secondary_observer_lr', 'robustness_lr']

Skipped: none

## Validation summary

- **1_no_leakage_structural**: {'test_folds_disjoint': True, 'union_equals_full_set': True, 'each_test_fold_single_dyad': True}

- **2_no_identity_columns**: {'forbidden_present': [], 'clean': True}

- **3_fold_count**: {'len_per_fold': 12, 'n_splits_groups': 12, 'expected': 12, 'ok': True}

- **4_per_dyad_scores_complete**: {'keys_match_groups': True, 'no_missing_or_nan': True, 'n_dyads': 12}

- **5_fold_dyad_bijection**: {'ok': True, 'n_mapped': 12}

- **6_majority_class_sanity**: {'auroc': 0.5, 'balanced_accuracy': 0.5, 'auroc_within_tol': True, 'balacc_within_tol': True}

- **7_metric_crosscheck**: {'sklearn_auroc': 0.49421487603305786, 'hand_auroc_via_mannwhitney': 0.49421487603305786, 'match_to_1e6': True}

- **8_label_sanity**: {'unique_values': [0, 1], 'positive_class_is_lie': True, 'pooled_balance': 0.4754745348618681}

- **9_reproducibility**: {'max_diff': 0.0, 'bit_identical': True}

- **10_exp1_unaffected**: {'expected': [0.5326, 0.5398, 0.5239, 0.5365, 0.5361], 'recomputed': [0.5326, 0.5398, 0.5239, 0.5365, 0.5359], 'match': False, 'max_abs_delta': 0.00019999999999997797, 'within_drift_tolerance': True, 'drift_tolerance_used': 0.005, 'note': "exact match failed but delta is within the expected cross-environment floating-point drift band (numpy/BLAS build difference; sklearn version matches exp1_baseline.json's recorded 1.8.0). Recorded as drift, not treated as a models.py regression. See PROGRESS.md for the diagnosis."}

- **11_frozen_files_untouched**: {'mtimes': {'/content/reveriehacks26/results/gate.json': 1787410703.0026295, '/content/reveriehacks26/results/trial_count_gate.md': 1787410703.0026295, '/content/reveriehacks26/results/frozen_hypotheses.md': 1787410703.0016296, '/content/reveriehacks26/data/processed/trial_table.csv': 1787410702.9846296, '/content/reveriehacks26/results/exp1_baseline.json': 1787410702.9996295, '/content/reveriehacks26/results/exp1_baseline.md': 1787410703.0016296}}

- **12_plausibility**: {'per_family_mean_auroc': {'logistic_regression': 0.5130592192193698, 'random_forest': 0.5210259706442599, 'gradient_boosting': 0.504443028045915, 'logistic_regression_l1': 0.5052340636749477, 'svm': 0.5132979565075094}, 'max_family_mean': 0.5210259706442599, 'max_single_dyad_auroc': 0.5950656096227447, 'leakage_suspicion': False}

- **13_convergence**: {'n_convergence_warnings': 0, 'warnings': []}

- **14_inner_cv_grouped**: {'logistic_regression': 'StratifiedGroupKFold', 'random_forest': None, 'gradient_boosting': None, 'logistic_regression_l1': 'StratifiedGroupKFold', 'svm': 'StratifiedGroupKFold'}

## Frozen-file confirmation

`results/gate.json`, `results/trial_count_gate.md`, `results/frozen_hypotheses.md` were read-only inputs, never written to or contradicted. mtimes recorded in `validations.11_frozen_files_untouched`; all predate this task's start.
