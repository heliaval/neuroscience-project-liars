# Experiment 4 -- Dyad-Specific Model (S14)

**What this experiment asks:** S14: does the partner's own prior interaction history help predict deception beyond the target's own history and the wider population?

## Finding A and Finding B

Every deceiver-role participant is the deceiver in exactly one session, and the two sessions occupy disjoint contiguous dyad_trial_seq blocks (1-484, 485-968). Holding out each participant's own last third (exp3's scheme) is fatal for Dyad-Specific: a session-1 deceiver's partner history lives entirely in session 2, chronologically after any block confined to session 1. Exp4 instead holds out the dyad's own late tercile (seq 647-968) as a single dyad-grain block, all belonging to the session-2 deceiver; the session-1 deceiver's entire block is then legally prior. Universal and Person-Specific are recomputed on this new split and are not exp3's numbers.

sub19_sub22 has only sub19 as a deceiver-role participant (sub22 was lost in S9 preprocessing), and sub19 is the session-1 deceiver, so this dyad's late tercile contains zero deceiver rows -- structurally untestable at dyad grain, discovered from data structure before any exp4 model was fit. results/frozen_hypotheses.md Amendment 1 (2026-08-22, pre-results) reduces exp4's n from 11 to 10 accordingly. sub19_sub22's 484 rows remain in the Universal training pool.

## Design

- n = 10 dyads (amended from pre-registered n=11; see `results/frozen_hypotheses.md` Amendment 1).
- Held-out block: dyad's own late tercile of dyad_trial_seq, [647,968], 322 rows -- all belong to the session-2 deceiver (B)
- Feature set: reliable_plus_marginal (1770 cols). Model: logistic_regression (L2) only

| condition | training rows |
|---|---|

| Universal | all deceiver rows from every other dyad (sub19_sub22 in the pool, sub01_sub02 excluded); n_train ~9,196; StratifiedGroupKFold(3) on pair_id |

| Person-Specific | B's own prior rows, seq [485,646], 162 rows; TimeSeriesSplit(3) |

| Dyad-Specific | person_specific UNION A's full session-1 block seq [1,484] (484 rows) = 646 rows; TimeSeriesSplit(3) on seq-sorted union -- PRIMARY (H3) |

| Dyad-Specific (N-matched) | 81 most-recent B rows UNION 81 most-recent A rows = 162 rows (volume control) |

| Person + Other-Dyad (volume-matched) | person_specific UNION 484 label-stratified rows drawn from the Universal pool (seeded per dyad) = 646 rows (source control) |


## Per-dyad score table (AUROC)

| pair_id | tested | partner | Universal | Person-Specific | Dyad-Specific | Dyad-Specific (N-matched) | Person + Other-Dyad (volume-matched) | majority |
|---|---|---|---|---|---|---|---|---|
| sub03_sub06 | sub06 | sub03 | 0.4635 | 0.5465 | 0.5037 | 0.5307 | 0.5231 | 0.5000 |
| sub04_sub05 | sub05 | sub04 | 0.5377 | 0.4656 | 0.4582 | 0.4793 | 0.5015 | 0.5000 |
| sub07_sub08 | sub07 | sub08 | 0.5499 | 0.5768 | 0.5705 | 0.5352 | 0.6258 | 0.5000 |
| sub09_sub10 | sub10 | sub09 | 0.5409 | 0.5345 | 0.5270 | 0.4940 | 0.4991 | 0.5000 |
| sub11_sub12 | sub11 | sub12 | 0.5393 | 0.5527 | 0.5231 | 0.4807 | 0.5660 | 0.5000 |
| sub13_sub14 | sub14 | sub13 | 0.4643 | 0.5121 | 0.4826 | 0.5594 | 0.4491 | 0.5000 |
| sub15_sub16 | sub16 | sub15 | 0.4973 | 0.5157 | 0.4803 | 0.5199 | 0.4775 | 0.5000 |
| sub17_sub18 | sub18 | sub17 | 0.5301 | 0.5218 | 0.4924 | 0.5208 | 0.5334 | 0.5000 |
| sub20_sub21 | sub20 | sub21 | 0.5417 | 0.4881 | 0.5003 | 0.5176 | 0.5071 | 0.5000 |
| sub23_sub24 | sub24 | sub23 | 0.5457 | 0.5192 | 0.5293 | 0.5017 | 0.5253 | 0.5000 |

## Primary claim -- DyadGain (H3)

Median DyadGain = -0.0185, 2/10 dyads positive, sign-test p = 0.1094, sign-flip permutation p = 0.05179 (n_signflip=10000). 95% CI: [-0.0304, -0.0028].

## Secondary claims

- **person_gain**: median = +0.0035, 5/10 positive, sign-test p = 1, permutation p = 0.8665

- **nfi**: median = -0.0163, 3/10 positive, sign-test p = 0.3438, permutation p = 0.5083

## Tertiary -- volume/source controls

- **dyad_gain_volume_controlled**: median = -0.0131, 4/10 positive, sign-test p = 0.7539, permutation p = 0.3231

- **dyad_gain_matched**: median = -0.0084, 4/10 positive, sign-test p = 0.7539, permutation p = 0.5482


DyadGain (H3) is the single pre-registered primary inferential claim; PersonGain and NFI are pre-specified secondaries; the two volume/source controls are tertiary. No correction is applied to a single primary. This declaration is written before the numbers below and is not revised after seeing them.

## NFI distribution

Median NFI = -0.0163, 3/10 positive, sign-test p = 0.3438.

## Pooled descriptive aggregate (NOT the test, S20)

- Universal pooled mean AUROC: 0.5210

- Person-Specific pooled mean AUROC: 0.5233

- Dyad-Specific pooled mean AUROC: 0.5067

- Dyad-Specific (N-matched) pooled mean AUROC: 0.5139

- Person + Other-Dyad (volume-matched) pooled mean AUROC: 0.5208

## Environment and drift check

Remote: Python 3.8.10, sklearn 1.3.2, numpy 1.24.4. Laptop (exp1/exp3): Python 3.13.14, sklearn 1.8.0.

Drift check (3 sampled exp3 participants, refit in this environment vs laptop): max|delta AUROC| = 0.0149. Measurement of environment comparability only (Python 3.8/sklearn 1.3.2 remote vs Python 3.13/sklearn 1.8.0 laptop). Never a source of exp4's own numbers.

## Validation summary

- **1_chronological_person_specific**: {'ok': True, 'per_dyad': {'sub03_sub06': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}, 'sub04_sub05': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}, 'sub07_sub08': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}, 'sub09_sub10': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}, 'sub11_sub12': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}, 'sub13_sub14': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}, 'sub15_sub16': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}, 'sub17_sub18': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}, 'sub20_sub21': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}, 'sub23_sub24': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}}

- **2_chronological_dyad_specific**: {'ok': True, 'per_dyad': {'sub03_sub06': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True, 'only_this_dyad': True}, 'sub04_sub05': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True, 'only_this_dyad': True}, 'sub07_sub08': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True, 'only_this_dyad': True}, 'sub09_sub10': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True, 'only_this_dyad': True}, 'sub11_sub12': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True, 'only_this_dyad': True}, 'sub13_sub14': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True, 'only_this_dyad': True}, 'sub15_sub16': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True, 'only_this_dyad': True}, 'sub17_sub18': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True, 'only_this_dyad': True}, 'sub20_sub21': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True, 'only_this_dyad': True}, 'sub23_sub24': {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True, 'only_this_dyad': True}}}

- **3_chronological_controls**: {'ok': True, 'per_dyad': {'sub03_sub06': {'n_matched_chronological': True, 'n_matched_only_this_dyad': True, 'person_other_dyad_b_part_chronological': True, 'person_other_dyad_draw_excludes_target_and_sub01_sub02': True}, 'sub04_sub05': {'n_matched_chronological': True, 'n_matched_only_this_dyad': True, 'person_other_dyad_b_part_chronological': True, 'person_other_dyad_draw_excludes_target_and_sub01_sub02': True}, 'sub07_sub08': {'n_matched_chronological': True, 'n_matched_only_this_dyad': True, 'person_other_dyad_b_part_chronological': True, 'person_other_dyad_draw_excludes_target_and_sub01_sub02': True}, 'sub09_sub10': {'n_matched_chronological': True, 'n_matched_only_this_dyad': True, 'person_other_dyad_b_part_chronological': True, 'person_other_dyad_draw_excludes_target_and_sub01_sub02': True}, 'sub11_sub12': {'n_matched_chronological': True, 'n_matched_only_this_dyad': True, 'person_other_dyad_b_part_chronological': True, 'person_other_dyad_draw_excludes_target_and_sub01_sub02': True}, 'sub13_sub14': {'n_matched_chronological': True, 'n_matched_only_this_dyad': True, 'person_other_dyad_b_part_chronological': True, 'person_other_dyad_draw_excludes_target_and_sub01_sub02': True}, 'sub15_sub16': {'n_matched_chronological': True, 'n_matched_only_this_dyad': True, 'person_other_dyad_b_part_chronological': True, 'person_other_dyad_draw_excludes_target_and_sub01_sub02': True}, 'sub17_sub18': {'n_matched_chronological': True, 'n_matched_only_this_dyad': True, 'person_other_dyad_b_part_chronological': True, 'person_other_dyad_draw_excludes_target_and_sub01_sub02': True}, 'sub20_sub21': {'n_matched_chronological': True, 'n_matched_only_this_dyad': True, 'person_other_dyad_b_part_chronological': True, 'person_other_dyad_draw_excludes_target_and_sub01_sub02': True}, 'sub23_sub24': {'n_matched_chronological': True, 'n_matched_only_this_dyad': True, 'person_other_dyad_b_part_chronological': True, 'person_other_dyad_draw_excludes_target_and_sub01_sub02': True}}, 'note': "Person+Other-Dyad's cross-dyad draw is checked for target/sub01_sub02 exclusion, not seq-ordering -- dyad_trial_seq is dyad-local, so a different dyad's seq values carry no chronological relationship to d's timeline (see module docstring)."}

- **4_identical_held_out_block**: {'ok': True, 'note': "True by construction -- every condition's _fit_predict call is passed blocks[d]['held_out_idx'] directly."}

- **5_no_identity_columns**: {'forbidden_set': ['condition', 'condition_raw', 'dm_excluded_reason', 'dyad_trial_seq', 'fb_excluded_reason', 'observer_guess', 'outcome', 'pair_id', 'participant_id', 'partner_id', 'points', 'role', 'role_raw', 'round', 'session_id', 'session_order', 'trial'], 'checked_at': 'run() before .values'}

- **6_dyad_pool_reconciliation**: {'ok': True, 'n_tested_dyads': 10, 'sub19_sub22_present_in_frame_not_tested': True, 'sub01_sub02_present_in_frame_not_tested': True, 'no_nan_scores': True}

- **7_universal_pool_exclusion**: {'ok': True}

- **8_volume_bookkeeping**: {'ok': True, 'per_dyad': {'sub03_sub06': {'n_universal': 9190, 'n_person_specific': 162, 'n_dyad_specific': 646, 'n_matched': 162, 'n_person_other_dyad': 646, 'checks': {'pers_near_162': True, 'dyad_near_646': True, 'nmatch_near_162': True, 'pother_near_646': True, 'dyad_eq_pother': True, 'pers_eq_nmatch': True, 'univ_near_9196': True}}, 'sub04_sub05': {'n_universal': 9190, 'n_person_specific': 162, 'n_dyad_specific': 646, 'n_matched': 162, 'n_person_other_dyad': 646, 'checks': {'pers_near_162': True, 'dyad_near_646': True, 'nmatch_near_162': True, 'pother_near_646': True, 'dyad_eq_pother': True, 'pers_eq_nmatch': True, 'univ_near_9196': True}}, 'sub07_sub08': {'n_universal': 9190, 'n_person_specific': 162, 'n_dyad_specific': 646, 'n_matched': 162, 'n_person_other_dyad': 646, 'checks': {'pers_near_162': True, 'dyad_near_646': True, 'nmatch_near_162': True, 'pother_near_646': True, 'dyad_eq_pother': True, 'pers_eq_nmatch': True, 'univ_near_9196': True}}, 'sub09_sub10': {'n_universal': 9190, 'n_person_specific': 162, 'n_dyad_specific': 646, 'n_matched': 162, 'n_person_other_dyad': 646, 'checks': {'pers_near_162': True, 'dyad_near_646': True, 'nmatch_near_162': True, 'pother_near_646': True, 'dyad_eq_pother': True, 'pers_eq_nmatch': True, 'univ_near_9196': True}}, 'sub11_sub12': {'n_universal': 9190, 'n_person_specific': 162, 'n_dyad_specific': 646, 'n_matched': 162, 'n_person_other_dyad': 646, 'checks': {'pers_near_162': True, 'dyad_near_646': True, 'nmatch_near_162': True, 'pother_near_646': True, 'dyad_eq_pother': True, 'pers_eq_nmatch': True, 'univ_near_9196': True}}, 'sub13_sub14': {'n_universal': 9190, 'n_person_specific': 162, 'n_dyad_specific': 646, 'n_matched': 162, 'n_person_other_dyad': 646, 'checks': {'pers_near_162': True, 'dyad_near_646': True, 'nmatch_near_162': True, 'pother_near_646': True, 'dyad_eq_pother': True, 'pers_eq_nmatch': True, 'univ_near_9196': True}}, 'sub15_sub16': {'n_universal': 9190, 'n_person_specific': 162, 'n_dyad_specific': 646, 'n_matched': 162, 'n_person_other_dyad': 646, 'checks': {'pers_near_162': True, 'dyad_near_646': True, 'nmatch_near_162': True, 'pother_near_646': True, 'dyad_eq_pother': True, 'pers_eq_nmatch': True, 'univ_near_9196': True}}, 'sub17_sub18': {'n_universal': 9190, 'n_person_specific': 162, 'n_dyad_specific': 646, 'n_matched': 162, 'n_person_other_dyad': 646, 'checks': {'pers_near_162': True, 'dyad_near_646': True, 'nmatch_near_162': True, 'pother_near_646': True, 'dyad_eq_pother': True, 'pers_eq_nmatch': True, 'univ_near_9196': True}}, 'sub20_sub21': {'n_universal': 9192, 'n_person_specific': 162, 'n_dyad_specific': 644, 'n_matched': 162, 'n_person_other_dyad': 644, 'checks': {'pers_near_162': True, 'dyad_near_646': True, 'nmatch_near_162': True, 'pother_near_646': True, 'dyad_eq_pother': True, 'pers_eq_nmatch': True, 'univ_near_9196': True}}, 'sub23_sub24': {'n_universal': 9194, 'n_person_specific': 162, 'n_dyad_specific': 642, 'n_matched': 162, 'n_person_other_dyad': 642, 'checks': {'pers_near_162': True, 'dyad_near_646': True, 'nmatch_near_162': True, 'pother_near_646': True, 'dyad_eq_pother': True, 'pers_eq_nmatch': True, 'univ_near_9196': True}}}}

- **9_majority_class_sanity**: {'mean_auroc': 0.5, 'mean_balanced_accuracy': 0.5, 'ok': True}

- **10_reproducibility**: {'max_diff': 0.0, 'per_unit': {'pers_sub03_sub06': 0.0, 'dyad_sub04_sub05': 0.0, 'univ_sub07_sub08': 0.0}}

- **11_grid_n_jobs_invariance**: {'dyad_sampled': 'sub03_sub06', 'n_jobs_1_seconds': -0.8134438991546631, 'n_jobs_chosen_seconds': 1.1415894031524658, 'n_jobs_chosen': 6, 'metrics_identical': True, 'best_params_n1': {'clf__C': 1.0}, 'best_params_nchosen': {'clf__C': 1.0}}

- **12_metric_crosscheck**: {'dyad': 'sub03_sub06', 'sklearn_auroc': 0.5465035504785427, 'hand_auroc': 0.5465035504785427, 'match_to_1e6': True}

- **13_upstream_files_hashed**: {'hashes': {'models.py': '117ad942f6500edc839774f7f4d07cf4d68a179f18cc25b4de157bb45e43820e', 'gate.json': '900ffddb6b6546e4681549d6d2c5744d596f21bb498691f99d33b0125164dc5d', 'frozen_hypotheses.md': 'f16a32495d437f1dcd1959e57a47668ef93d20a297f2d9475e421fe2983498bd', 'exp3_personalized.json': '6877d2a493b68d6a02188c106edf630a88f776806c0ce53e18777c991be83d7b', 'feature_dictionary.csv': 'd57e93b8d6389b2ea5ff3b271b298dc47d3fdc9a3864cd5ea69d4eb48f6a9ddb'}, 'frozen_hypotheses_matches_amended_hash': True}

- **14_data_integrity**: {'row_count_10648': True, 'col_count_matches_manifest': True, 'total_frame_hash_verified': 'asserted in load_deceiver_frame() -- would have raised if mismatched'}

- **15_gate_verdict_recorded**: {'gate_json_exp4_entry': {'smallest_fold_total': 161, 'smallest_fold_minority': 70, 'clause_a': True, 'clause_b': True, 'n_dyads_passing_clause_a': 11, 'n_dyads_considered': 11, 'verdict': 'CONFIRMATORY', 'reason': 'Smallest fold minority (70) clears threshold (60); 11/11 included dyads individually clear it (needed >= 10).', 'smallest_fold_dyad': 'sub01_sub02', 'assumption': 'Same late-tercile stand-in as Exp3, at dyad grain rather than participant grain.'}, 'minority_counts_per_dyad': {'sub03_sub06': 158, 'sub04_sub05': 141, 'sub07_sub08': 160, 'sub09_sub10': 136, 'sub11_sub12': 141, 'sub13_sub14': 150, 'sub15_sub16': 152, 'sub17_sub18': 134, 'sub20_sub21': 150, 'sub23_sub24': 153}, 'threshold_m': 60, 'all_dyads_clear_threshold': True}

- **16_amendment_recorded**: {'amendment_1_present': True, 'n_dyads': 10, 'mismatch_vs_original_n11_stated': True, 'original_n11_line_present': True}

- **17_plausibility**: {'max_auroc': 0.6257716049382717, 'mean_dyad_specific_auroc': 0.5067209787889294, 'exp1_pooled_auroc_reference': 0.5338, 'implausibly_above_exp1': False, 'dyads_flagged': [], 'note': 'Flagging is for inspection, not automatic exclusion (exp3 precedent: sub15 was flagged, inspected, and kept).'}

- **18_convergence**: {'n_convergence_warnings_per_condition': {'universal': 0, 'person_specific': 0, 'dyad_specific': 0, 'n_matched': 0, 'person_other_dyad': 0}}

- **drift_check**: {'sampled_participants': ['sub03', 'sub04', 'sub05'], 'per_participant': {'sub03': {'remote_auroc': 0.6831215379602477, 'laptop_auroc': 0.6831215379602477, 'abs_delta': 0.0}, 'sub04': {'remote_auroc': 0.585243328100471, 'laptop_auroc': 0.585243328100471, 'abs_delta': 0.0}, 'sub05': {'remote_auroc': 0.5031625553447185, 'laptop_auroc': 0.4882985452245414, 'abs_delta': 0.014864010120177129}}, 'max_abs_delta_auroc': 0.014864010120177129, 'note': "Measurement of environment comparability only (Python 3.8/sklearn 1.3.2 remote vs Python 3.13/sklearn 1.8.0 laptop). Never a source of exp4's own numbers."}

## Limitations

- n=10, not the pre-registered n=11 (Amendment 1; sub19_sub22 structurally untestable at dyad grain).
- All five conditions fit in one Python-3.8/sklearn-1.3.2 remote environment; not directly comparable point-to-point to exp1/exp3's laptop numbers (see drift check above).
- Person-Specific trains on only 162 rows against 1,770 features -- severely underdetermined by design, the honest operationalization of the question.
