# Experiment 3 -- Personalized Model, Population vs Person-Specific (S13)

**What this experiment asks:** S13: does knowing the individual improve prediction?

## Unit of analysis

S13 instructs recording each participant's score individually and computing the aggregate via S20, not by pooling trials across participants. S20 and results/frozen_hypotheses.md (frozen before any model was fit) fix the unit of analysis as the dyad, n=11 for exp3. These are read as consistent, not competing: S13's 'individually' governs what is measured (forbidding trial-pooling across participants into one AUROC), not what is statistically tested. So measurement happens at participant grain (21 participants, all five metrics, both conditions); the dyad-grain delta is the arithmetic mean of the dyad's participants' deltas (the only aggregation that uses both participants when present, is symmetric in them, and reduces to the lone participant's delta for sub19_sub22); S20's sign test and paired sign-flip permutation test run on those 11 dyad-grain values. The 21-value participant-grain sign test is also reported, labelled secondary and non-independent (dyad partners share dyad/sessions/opponent and, for the population condition, the same training set).

## Participant reconciliation

21 participants used, over 11 dyads. Excluded: `sub01` (dyad sub01_sub02 excluded from exp3 by frozen_hypotheses.md (n=11 for exp3); sub01 additionally excluded from population training pool (Step 4b).) `sub02` (no deceiver-role session was ever recorded for sub01_sub02 (archive gap); sub02 has zero deceiver rows in single_brain.parquet.) `sub22` (the Player_sub22_Observer_sub19 session was unrecoverable in S9 EEG preprocessing (see data/processed/onednn_notes.md); sub22 has zero deceiver rows in the feature table even though the dyad survives via sub19.)

`sub19_sub22` contributes a single participant (n_participants_per_dyad=1). Conflict with frozen file: False.

## Design

- Split: per-participant chronological, dyad_trial_seq, ceil(2n/3) train / remainder test (323/161 for every participant, since every deceiver block is 484 rows)
- Population: trained on other dyads' deceiver rows (target's whole dyad excluded, and sub01_sub02 excluded from the training pool entirely)
- Personalized: trained on target participant's first ceil(2n/3) of own deceiver trials, chronologically ordered by dyad_trial_seq
- Inner CV: population = StratifiedGroupKFold(3) on pair_id (S19); personalized = TimeSeriesSplit(3), falls back to StratifiedKFold(3, shuffle=False) if any inner fold lacks both classes (S19 chronology)
- Feature set: reliable_plus_marginal (1770 cols). Model family: logistic_regression (L2) only

## Gate status

Gate verdict for exp3: CONFIRMATORY (`results/frozen_hypotheses.md`). Participant-grain recheck (never seen by the original dyad-grain gate): 1 participant(s) below THRESHOLD_M=60 -- ['sub24']. Dyads passing Clause A: 10 of 11. Clause B satisfied: True. Verdict unchanged: CONFIRMATORY.

## Per-participant score table

| participant | dyad | n_test | minority | population AUROC | personalized AUROC | delta |
|---|---|---|---|---|---|---|
| sub03 | sub03_sub06 | 161 | 62 | 0.5256 | 0.6831 | +0.1575 |
| sub04 | sub04_sub05 | 161 | 70 | 0.5091 | 0.5852 | +0.0761 |
| sub05 | sub04_sub05 | 161 | 68 | 0.5773 | 0.4883 | -0.0890 |
| sub06 | sub03_sub06 | 161 | 76 | 0.4735 | 0.4430 | -0.0305 |
| sub07 | sub07_sub08 | 161 | 79 | 0.5190 | 0.5644 | +0.0454 |
| sub08 | sub07_sub08 | 161 | 69 | 0.5525 | 0.4370 | -0.1155 |
| sub09 | sub09_sub10 | 161 | 78 | 0.4285 | 0.4921 | +0.0636 |
| sub10 | sub09_sub10 | 161 | 68 | 0.5577 | 0.5952 | +0.0375 |
| sub11 | sub11_sub12 | 161 | 78 | 0.6526 | 0.5732 | -0.0794 |
| sub12 | sub11_sub12 | 161 | 67 | 0.5373 | 0.5592 | +0.0219 |
| sub13 | sub13_sub14 | 161 | 80 | 0.5492 | 0.5540 | +0.0048 |
| sub14 | sub13_sub14 | 161 | 75 | 0.4344 | 0.5509 | +0.1165 |
| sub15 | sub15_sub16 | 161 | 69 | 0.4820 | 0.6931 | +0.2111 |
| sub16 | sub15_sub16 | 161 | 79 | 0.4481 | 0.5502 | +0.1020 |
| sub17 | sub17_sub18 | 161 | 63 | 0.4258 | 0.4381 | +0.0123 |
| sub18 | sub17_sub18 | 161 | 72 | 0.5133 | 0.5649 | +0.0517 |
| sub19 | sub19_sub22 | 161 | 80 | 0.5971 | 0.6080 | +0.0110 |
| sub20 | sub20_sub21 | 161 | 71 | 0.5480 | 0.5219 | -0.0261 |
| sub21 | sub20_sub21 | 160 | 77 | 0.4840 | 0.5814 | +0.0975 |
| sub23 | sub23_sub24 | 160 | 72 | 0.4943 | 0.5649 | +0.0705 |
| sub24 | sub23_sub24 | 161 | 53 | 0.5377 | 0.4890 | -0.0487 |

## Per-dyad delta table and S20 test (primary, n=11)

| pair_id | n_participants | delta AUROC |
|---|---|---|
| sub03_sub06 | 2 | +0.0635 |
| sub04_sub05 | 2 | -0.0064 |
| sub07_sub08 | 2 | -0.0350 |
| sub09_sub10 | 2 | +0.0506 |
| sub11_sub12 | 2 | -0.0287 |
| sub13_sub14 | 2 | +0.0606 |
| sub15_sub16 | 2 | +0.1566 |
| sub17_sub18 | 2 | +0.0320 |
| sub19_sub22 | 1 | +0.0110 |
| sub20_sub21 | 2 | +0.0357 |
| sub23_sub24 | 2 | +0.0109 |

**Median delta = +0.0320**, 8/11 positive, sign-test p = 0.2266, sign-flip permutation p = 0.1233 (n_signflip=10000).

## Sensitivity (n=10) and participant-grain secondary test

n=10 (sub19_sub22 dropped): median delta = +0.0338, 7/10 positive, sign-test p = 0.3438, permutation p = 0.0485.

Participant-grain (n=21, non-independent -- Non-independent: participants within a dyad share the same dyad, sessions, opponent, and (population condition) the same training set. Reported for consistency (S23), not as the primary inferential test.): median delta = +0.0375, 15/21 positive, sign-test p = 0.07835, permutation p = 0.1204.

## Confidence intervals (S22)

Between-dyad 95% CI on delta AUROC: 0.0319 [-0.0038, 0.0676]. sub24's held-out block (53 minority < 60) carries a visibly wider Hanley-McNeil SE than the other participants (see confidence_intervals.per_participant_hanley_mcneil_se in the JSON).

## Cross-experiment reading

| quantity | value |
|---|---|
| exp1 pooled LR AUROC | 0.533789809045845 |
| exp2 LODO LR AUROC | not available at run time |
| exp3 population mean AUROC | 0.5165 |
| exp3 personalized mean AUROC | 0.5494 |

Personalized mean AUROC exceeds population mean AUROC. This supports H2 (person-specific signature) descriptively; the S20 paired test at dyad grain (n=11) is the actual inferential claim, not this mean comparison.

exp3's population number is a near-neighbour of exp2's LODO number but is not the same quantity: exp2 scores whole held-out dyads, exp3 scores only the last third of one participant's deceiver trials. Everything in this project sits within a few points of chance (exp1's headline was 0.534); a 0.53-ish number is not dressed up as a strong result. With 11 dyads, small differences are not overinterpreted (S22).

## Runtime and resumption notes

Total measured runtime this session: 2.3s. Resume launches so far: 3.

## Limitations

- Personalized fits use only 323 rows against 1,770 features -- a severely underdetermined fit by design (this is the honest operationalization of S13's question, not a flaw).
- The ~30x training-size asymmetry (population ~9.2-9.7k rows vs personalized 323 rows) is intrinsic to the question and was not corrected by subsampling.
- `sub24`'s held-out block has only 53 minority-class trials, below THRESHOLD_M=60; its CI is visibly wider than the other participants'.
- `sub19_sub22` contributes a single participant (sub19 only) to the dyad-grain test; its delta is a one-participant mean and carries the same weight as every other dyad in the equal-weight paired test.
- Everything in this project sits within a few points of chance (exp1's headline pooled AUROC was 0.534).

## Validation summary (all sixteen)

- **1_chronological_personalized**: {'ok': np.True_, 'per_participant': {'sub03': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub04': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub05': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub06': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub07': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub08': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub09': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub10': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub11': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub12': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub13': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub14': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub15': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub16': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub17': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub18': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub19': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub20': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub21': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub23': {'disjoint': True, 'union_full': True, 'chronological': np.True_}, 'sub24': {'disjoint': True, 'union_full': True, 'chronological': np.True_}}}

- **2_chronological_population**: {'ok': True, 'per_dyad': {'sub03_sub06': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}, 'sub04_sub05': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}, 'sub07_sub08': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}, 'sub09_sub10': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}, 'sub11_sub12': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}, 'sub13_sub14': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}, 'sub15_sub16': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}, 'sub17_sub18': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}, 'sub19_sub22': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}, 'sub20_sub21': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}, 'sub23_sub24': {'target_dyad_absent_from_train': True, 'n_train_pair_ids': 10}}}

- **3_identical_held_out_block**: {'ok': True, 'test_block_length_per_participant': {'sub03': 161, 'sub04': 161, 'sub05': 161, 'sub06': 161, 'sub07': 161, 'sub08': 161, 'sub09': 161, 'sub10': 161, 'sub11': 161, 'sub12': 161, 'sub13': 161, 'sub14': 161, 'sub15': 161, 'sub16': 161, 'sub17': 161, 'sub18': 161, 'sub19': 161, 'sub20': 161, 'sub21': 160, 'sub23': 160, 'sub24': 161}}

- **4_no_identity_columns**: {'forbidden_set': ['condition', 'condition_raw', 'dm_excluded_reason', 'dyad_trial_seq', 'fb_excluded_reason', 'observer_guess', 'pair_id', 'participant_id', 'partner_id', 'points', 'role', 'role_raw', 'round', 'session_id', 'session_order', 'trial'], 'checked_at': 'prepare_modeling_frame call site in run()'}

- **5_participant_pool_complete**: {'n_participants': 21, 'expected': 21, 'n_dyads': 11, 'expected_dyads': 11, 'sets_equal': True, 'sub02_absent': True, 'sub22_absent': True, 'sub01_excluded': True, 'no_nan_scores': True}

- **6_aggregation_correctness**: {'ok': True, 'per_dyad': {'sub03_sub06': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}, 'sub04_sub05': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}, 'sub07_sub08': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}, 'sub09_sub10': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}, 'sub11_sub12': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}, 'sub13_sub14': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}, 'sub15_sub16': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}, 'sub17_sub18': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}, 'sub19_sub22': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}, 'sub20_sub21': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}, 'sub23_sub24': {'mean_of_deltas_matches_stored': True, 'delta_of_means_matches_mean_of_deltas': True}}}

- **7_majority_class_sanity**: {'mean_auroc': 0.5, 'mean_balanced_accuracy': 0.5, 'ok': True}

- **8_metric_crosscheck**: {'participant': 'sub03', 'sklearn_auroc': 0.6831215379602477, 'hand_auroc': 0.6831215379602477, 'match_to_1e6': True}

- **9_label_sanity**: {'unique_values': [0, 1], 'positive_class_is_lie': True, 'every_test_block_has_both_classes': True}

- **10_reproducibility**: {'max_diff': 0.0, 'per_participant': {'sub03': 0.0, 'sub04': 0.0, 'sub05': 0.0}, 'note': 'TimeSeriesSplit has no random_state; deterministic by construction.'}

- **11_inner_cv_scheme**: {'population_inner_classes': {'sub03_sub06': 'StratifiedGroupKFold', 'sub04_sub05': 'StratifiedGroupKFold', 'sub07_sub08': 'StratifiedGroupKFold', 'sub09_sub10': 'StratifiedGroupKFold', 'sub11_sub12': 'StratifiedGroupKFold', 'sub13_sub14': 'StratifiedGroupKFold', 'sub15_sub16': 'StratifiedGroupKFold', 'sub17_sub18': 'StratifiedGroupKFold', 'sub19_sub22': 'StratifiedGroupKFold', 'sub20_sub21': 'StratifiedGroupKFold', 'sub23_sub24': 'StratifiedGroupKFold'}, 'personalized_inner_classes': {'sub03': 'TimeSeriesSplit', 'sub04': 'TimeSeriesSplit', 'sub05': 'TimeSeriesSplit', 'sub06': 'TimeSeriesSplit', 'sub07': 'TimeSeriesSplit', 'sub08': 'TimeSeriesSplit', 'sub09': 'TimeSeriesSplit', 'sub10': 'TimeSeriesSplit', 'sub11': 'TimeSeriesSplit', 'sub12': 'TimeSeriesSplit', 'sub13': 'TimeSeriesSplit', 'sub14': 'TimeSeriesSplit', 'sub15': 'TimeSeriesSplit', 'sub16': 'TimeSeriesSplit', 'sub17': 'TimeSeriesSplit', 'sub18': 'TimeSeriesSplit', 'sub19': 'TimeSeriesSplit', 'sub20': 'TimeSeriesSplit', 'sub21': 'TimeSeriesSplit', 'sub23': 'TimeSeriesSplit', 'sub24': 'TimeSeriesSplit'}, 'personalized_fallbacks': {'sub03': False, 'sub04': False, 'sub05': False, 'sub06': False, 'sub07': False, 'sub08': False, 'sub09': False, 'sub10': False, 'sub11': False, 'sub12': False, 'sub13': False, 'sub14': False, 'sub15': False, 'sub16': False, 'sub17': False, 'sub18': False, 'sub19': False, 'sub20': False, 'sub21': False, 'sub23': False, 'sub24': False}, 'n_fallbacks': 0, 'grid_min_C': 0.001, 'grid_satisfies_le_0.01': True, 'grid_used': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

- **12_models_py_unchanged**: {'models_py_modified': False}

- **13_gate_verdict_recorded**: {'frozen_hypotheses_classifies_exp3_confirmatory': True, 'gate_json_exp3_row': {'smallest_fold_total': 161, 'smallest_fold_minority': 70, 'clause_a': True, 'clause_b': True, 'n_dyads_passing_clause_a': 11, 'n_dyads_considered': 11, 'verdict': 'CONFIRMATORY', 'reason': 'Smallest fold minority (70) clears threshold (60); 11/11 included dyads individually clear it (needed >= 10).', 'smallest_fold_dyad': 'sub01_sub02', 'assumption': "The spec's own example ('train rounds 1-30, test rounds 31-40') does not map onto this dataset's 'round' field, which only spans 1-11 per session (44 trials/round) -- not the 30+ rounds the example implies. Standing in for the held-out block: the late tercile of the dyad's own dyad_trial_seq (2e), i.e. train = early+middle, test = late. This is stated as an assumption, not a fact pinned down by the spec."}, 'participant_grain_recheck': 'see gate_recheck block', 'verdict_unchanged_by_exp3': True}

- **14_frozen_files_untouched**: {'mtimes': {'C:\\Users\\Alber\\CN4\\reveriehacks26\\results\\gate.json': 1787242126.1023006, 'C:\\Users\\Alber\\CN4\\reveriehacks26\\results\\trial_count_gate.md': 1787242125.9957113, 'C:\\Users\\Alber\\CN4\\reveriehacks26\\results\\frozen_hypotheses.md': 1787242126.1033032, 'C:\\Users\\Alber\\CN4\\reveriehacks26\\data\\processed\\trial_table.csv': 1787241055.2587442, 'C:\\Users\\Alber\\CN4\\reveriehacks26\\results\\exp1_baseline.json': 1787342489.955807, 'C:\\Users\\Alber\\CN4\\reveriehacks26\\results\\exp1_baseline.md': 1787342489.9608066, 'C:\\Users\\Alber\\CN4\\reveriehacks26\\results\\exp2_universal.json': None, 'C:\\Users\\Alber\\CN4\\reveriehacks26\\results\\exp2_universal.md': None}}

- **15_plausibility**: {'max_auroc': 0.6931316950220541, 'mean_population_auroc': 0.516530325283956, 'mean_personalized_auroc': 0.5493972301285905, 'leakage_suspicion': False, 'participants_flagged_large_delta_or_high_auroc': ['sub15']}

- **16_convergence**: {'n_convergence_warnings_population': 0, 'n_convergence_warnings_personalized': 0}

## Frozen-file confirmation

`results/gate.json`, `results/trial_count_gate.md`, `results/frozen_hypotheses.md` were read-only inputs, never written to or contradicted. mtimes recorded in `validations.14_frozen_files_untouched`; all predate this task's start.
