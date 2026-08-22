# Experiment 5 -- Interaction-History Model (S15)

**What this experiment asks:** S15: does deception become more distinguishable after participants have interacted for longer?

## Design

- n = 10 dyads (amended from pre-registered n=11; see `results/frozen_hypotheses.md` Amendment 2).
- Held-out block: dyad's own late tercile of dyad_trial_seq, [647,968], 322 rows -- identical to exp4's, all belong to the session-2 deceiver (B)
- Ladder rungs: [81, 162, 322, 484, 646], confirmatory rungs: [322, 646]
- Feature set: reliable_plus_marginal (1770 cols). Model: logistic_regression (L2) only

Every participant is the deceiver in exactly one session, occupying disjoint contiguous dyad_trial_seq blocks (1-484, 485-968). Analysis 1A's dyad-grain terciles are therefore confounded with participant identity -- the early tercile is entirely deceiver A's trials, the late tercile entirely deceiver B's. This is published as data (per-bin participant ids and A/B row counts), not smoothed over.

sub19_sub22's only deceiver-role participant (sub19) is the session-1 deceiver, so its late tercile (exp5's held-out block) contains zero deceiver rows -- structurally untestable. results/frozen_hypotheses.md Amendment 2 reduces exp5's n from 11 to 10 accordingly; sub19_sub22's rows remain in the other-dyad training pool.

## Gate re-check (Amendment 2)

THRESHOLD_M = 60 (copied unchanged from the original gate).


| analysis | test-fold grain | minority | Clause A | Clause B | designation |
|---|---|---|---|---|---|

| 1A dyad-grain positional | 322/324/322 | 130 | PASS | 10/10 PASS | confirmatory_on_power_confounded |

| 1B participant-grain positional | 162/161/161 | 45 | FAIL | 8/10 FAIL | exploratory_underpowered |

| 2 learning curve | 322 (late tercile) | 134 | PASS | 10/10 PASS | confirmatory_at_frozen_rungs |

| 3 volume control | 322 (late tercile) | 134 | PASS | 10/10 PASS | confirmatory_at_k646 |


Failing bins: {'sub24': 45, 'sub17': 59}


## Per-dyad curve table (AUROC, same-dyad TimeSeriesSplit)

| pair_id | k=81 | k=162 | k=322 | k=484 | k=646 |

|---|---|---|---|---|---|

| sub03_sub06 | 0.4920 | 0.4828 | 0.4944 | 0.5141 | 0.5037 |

| sub04_sub05 | 0.5054 | 0.4751 | 0.4572 | 0.4762 | 0.4582 |

| sub07_sub08 | 0.4921 | 0.4724 | 0.4859 | 0.5273 | 0.5705 |

| sub09_sub10 | 0.4847 | 0.5268 | 0.5090 | 0.4935 | 0.5270 |

| sub11_sub12 | 0.4453 | 0.4431 | 0.4368 | 0.4512 | 0.5231 |

| sub13_sub14 | 0.5077 | 0.5372 | 0.4420 | 0.4767 | 0.4826 |

| sub15_sub16 | 0.5039 | 0.4821 | 0.4715 | 0.4763 | 0.4803 |

| sub17_sub18 | 0.5087 | 0.4899 | 0.5071 | 0.5161 | 0.4924 |

| sub20_sub21 | 0.4625 | 0.5277 | 0.5583 | 0.5529 | 0.5003 |

| sub23_sub24 | 0.5042 | 0.4754 | 0.5054 | 0.5151 | 0.5293 |


## Primary claim -- history_gain

Median = +0.0136, 8/10 dyads positive, sign-test p = 0.1094, permutation p = 0.1718. 95% CI: [-0.0110, 0.0510].


## Secondary claims

- **same_vs_other_dyad_at_646**: median = -0.0173, 4/10 positive, sign-test p = 0.7539, permutation p = 0.5188

- **positional_late_minus_early**: median = -0.0061, 5/10 positive, sign-test p = 1, permutation p = 1


## Control -- same-dyad vs other-dyad volume

Reported inside secondary claims (`same_vs_other_dyad_at_646`) and the exploratory per-rung breakdown below.


## Exploratory (Benjamini-Hochberg corrected)

- **within_person_late_minus_early**: median = -0.0114, 5/10 positive, sign-test p = 1 (BH-adjusted: 1)

- **same_vs_other_dyad_at_81**: median = -0.0015, 4/10 positive, sign-test p = 0.7539 (BH-adjusted: 1)

- **same_vs_other_dyad_at_162**: median = -0.0261, 3/10 positive, sign-test p = 0.3438 (BH-adjusted: 1)

- **same_vs_other_dyad_at_322**: median = -0.0156, 5/10 positive, sign-test p = 1 (BH-adjusted: 1)

- **same_vs_other_dyad_at_484**: median = +0.0025, 6/10 positive, sign-test p = 0.7539 (BH-adjusted: 1)

- **history_gain_volume_controlled**: median = +0.0148, 6/10 positive, sign-test p = 0.7539 (BH-adjusted: 1)

- **adjacent_rung_gain_81_to_162**: median = -0.0140, 3/10 positive, sign-test p = 0.3438 (BH-adjusted: 1)

- **adjacent_rung_gain_162_to_322**: median = +0.0026, 5/10 positive, sign-test p = 1 (BH-adjusted: 1)

- **adjacent_rung_gain_322_to_484**: median = +0.0121, 8/10 positive, sign-test p = 0.1094 (BH-adjusted: 1)

- **adjacent_rung_gain_484_to_646**: median = +0.0049, 6/10 positive, sign-test p = 0.7539 (BH-adjusted: 1)

- **slope_test**: median = -0.0069, 4/10 positive, sign-test p = 0.7539 (BH-adjusted: 1)


history_gain (k=646 vs k=322, same-dyad) is the single pre-registered primary claim, no correction. same_vs_other_dyad_at_646 and positional_late_minus_early are pre-registered secondaries, no correction. Every other rung comparison, curve slope, the difference-in-differences (history_gain_volume_controlled), and within_person_late_minus_early are exploratory, Benjamini-Hochberg corrected at alpha=0.05 within that family. Declared before any number below was computed, per results/frozen_hypotheses.md Amendment 2.


## Aggregate (NOT the test, S20)

- k=81: same-dyad median AUROC = 0.4980, other-dyad median AUROC = 0.5001

- k=162: same-dyad median AUROC = 0.4825, other-dyad median AUROC = 0.5116

- k=322: same-dyad median AUROC = 0.4902, other-dyad median AUROC = 0.5162

- k=484: same-dyad median AUROC = 0.5038, other-dyad median AUROC = 0.5099

- k=646: same-dyad median AUROC = 0.5020, other-dyad median AUROC = 0.5199


## Analysis 1B -- within-participant early/middle/late (EXPLORATORY, UNDERPOWERED)

Fails the exp5-specific gate re-check (Clause A on sub24=45, Clause B at 8/10). Reported with CIs per the frozen file's own rule, never dropped.

Median = -0.0114, 5/10 positive, sign-test p = 1. 95% CI: [-0.0366, 0.0366].


## Environment

Remote: Python 3.8.10, sklearn 1.3.2. Laptop: Python 3.13.14, sklearn 1.8.0.


## Validation summary

- **1_chronological_ladder**: {'ok': True, 'per_dyad': {'sub03_sub06': {81: {'train_max_seq': 81, 'test_min_seq': 647, 'chronological': True}, 162: {'train_max_seq': 162, 'test_min_seq': 647, 'chronological': True}, 322: {'train_max_seq': 322, 'test_min_seq': 647, 'chronological': True}, 484: {'train_max_seq': 484, 'test_min_seq': 647, 'chronological': True}, 646: {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}, 'sub04_sub05': {81: {'train_max_seq': 81, 'test_min_seq': 647, 'chronological': True}, 162: {'train_max_seq': 162, 'test_min_seq': 647, 'chronological': True}, 322: {'train_max_seq': 322, 'test_min_seq': 647, 'chronological': True}, 484: {'train_max_seq': 484, 'test_min_seq': 647, 'chronological': True}, 646: {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}, 'sub07_sub08': {81: {'train_max_seq': 81, 'test_min_seq': 647, 'chronological': True}, 162: {'train_max_seq': 162, 'test_min_seq': 647, 'chronological': True}, 322: {'train_max_seq': 322, 'test_min_seq': 647, 'chronological': True}, 484: {'train_max_seq': 484, 'test_min_seq': 647, 'chronological': True}, 646: {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}, 'sub09_sub10': {81: {'train_max_seq': 81, 'test_min_seq': 647, 'chronological': True}, 162: {'train_max_seq': 162, 'test_min_seq': 647, 'chronological': True}, 322: {'train_max_seq': 322, 'test_min_seq': 647, 'chronological': True}, 484: {'train_max_seq': 484, 'test_min_seq': 647, 'chronological': True}, 646: {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}, 'sub11_sub12': {81: {'train_max_seq': 81, 'test_min_seq': 647, 'chronological': True}, 162: {'train_max_seq': 162, 'test_min_seq': 647, 'chronological': True}, 322: {'train_max_seq': 322, 'test_min_seq': 647, 'chronological': True}, 484: {'train_max_seq': 484, 'test_min_seq': 647, 'chronological': True}, 646: {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}, 'sub13_sub14': {81: {'train_max_seq': 81, 'test_min_seq': 647, 'chronological': True}, 162: {'train_max_seq': 162, 'test_min_seq': 647, 'chronological': True}, 322: {'train_max_seq': 322, 'test_min_seq': 647, 'chronological': True}, 484: {'train_max_seq': 484, 'test_min_seq': 647, 'chronological': True}, 646: {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}, 'sub15_sub16': {81: {'train_max_seq': 81, 'test_min_seq': 647, 'chronological': True}, 162: {'train_max_seq': 162, 'test_min_seq': 647, 'chronological': True}, 322: {'train_max_seq': 322, 'test_min_seq': 647, 'chronological': True}, 484: {'train_max_seq': 484, 'test_min_seq': 647, 'chronological': True}, 646: {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}, 'sub17_sub18': {81: {'train_max_seq': 81, 'test_min_seq': 647, 'chronological': True}, 162: {'train_max_seq': 162, 'test_min_seq': 647, 'chronological': True}, 322: {'train_max_seq': 322, 'test_min_seq': 647, 'chronological': True}, 484: {'train_max_seq': 484, 'test_min_seq': 647, 'chronological': True}, 646: {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}, 'sub20_sub21': {81: {'train_max_seq': 81, 'test_min_seq': 647, 'chronological': True}, 162: {'train_max_seq': 162, 'test_min_seq': 647, 'chronological': True}, 322: {'train_max_seq': 322, 'test_min_seq': 647, 'chronological': True}, 484: {'train_max_seq': 484, 'test_min_seq': 647, 'chronological': True}, 646: {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}, 'sub23_sub24': {81: {'train_max_seq': 81, 'test_min_seq': 647, 'chronological': True}, 162: {'train_max_seq': 162, 'test_min_seq': 647, 'chronological': True}, 322: {'train_max_seq': 322, 'test_min_seq': 647, 'chronological': True}, 484: {'train_max_seq': 484, 'test_min_seq': 647, 'chronological': True}, 646: {'train_max_seq': 646, 'test_min_seq': 647, 'chronological': True}}}}

- **2_nesting**: {'ok': True, 'per_dyad': {'sub03_sub06': {'sizes': [81, 162, 322, 484, 646], 'monotone': True, 'strict_subset': True}, 'sub04_sub05': {'sizes': [81, 162, 322, 484, 646], 'monotone': True, 'strict_subset': True}, 'sub07_sub08': {'sizes': [81, 162, 322, 484, 646], 'monotone': True, 'strict_subset': True}, 'sub09_sub10': {'sizes': [81, 162, 322, 484, 646], 'monotone': True, 'strict_subset': True}, 'sub11_sub12': {'sizes': [81, 162, 322, 484, 646], 'monotone': True, 'strict_subset': True}, 'sub13_sub14': {'sizes': [81, 162, 322, 484, 646], 'monotone': True, 'strict_subset': True}, 'sub15_sub16': {'sizes': [81, 162, 322, 484, 646], 'monotone': True, 'strict_subset': True}, 'sub17_sub18': {'sizes': [81, 162, 322, 484, 646], 'monotone': True, 'strict_subset': True}, 'sub20_sub21': {'sizes': [81, 162, 320, 482, 644], 'monotone': True, 'strict_subset': True}, 'sub23_sub24': {'sizes': [81, 162, 320, 480, 642], 'monotone': True, 'strict_subset': True}}}

- **3_identical_held_out_block**: {'ok': True, 'note': "True by construction -- run_ladder_tss/run_ladder_skf/run_ladder_other/run_universal_and_score_bins/run_majority all pass blocks[d]['held_out_idx'] directly, never a per-condition copy."}

- **4_bin_integrity**: {'ok': True, 'dyad_grain': {'sub03_sub06': {'early': {'n': 323, 'n_minority': 148}, 'middle': {'n': 323, 'n_minority': 148}, 'late': {'n': 322, 'n_minority': 158}}, 'sub04_sub05': {'early': {'n': 323, 'n_minority': 156}, 'middle': {'n': 323, 'n_minority': 145}, 'late': {'n': 322, 'n_minority': 141}}, 'sub07_sub08': {'early': {'n': 323, 'n_minority': 146}, 'middle': {'n': 323, 'n_minority': 139}, 'late': {'n': 322, 'n_minority': 160}}, 'sub09_sub10': {'early': {'n': 323, 'n_minority': 161}, 'middle': {'n': 323, 'n_minority': 150}, 'late': {'n': 322, 'n_minority': 136}}, 'sub11_sub12': {'early': {'n': 323, 'n_minority': 147}, 'middle': {'n': 323, 'n_minority': 136}, 'late': {'n': 322, 'n_minority': 141}}, 'sub13_sub14': {'early': {'n': 323, 'n_minority': 154}, 'middle': {'n': 323, 'n_minority': 155}, 'late': {'n': 322, 'n_minority': 150}}, 'sub15_sub16': {'early': {'n': 323, 'n_minority': 140}, 'middle': {'n': 323, 'n_minority': 147}, 'late': {'n': 322, 'n_minority': 152}}, 'sub17_sub18': {'early': {'n': 323, 'n_minority': 130}, 'middle': {'n': 323, 'n_minority': 135}, 'late': {'n': 322, 'n_minority': 134}}, 'sub20_sub21': {'early': {'n': 322, 'n_minority': 160}, 'middle': {'n': 322, 'n_minority': 145}, 'late': {'n': 322, 'n_minority': 150}}, 'sub23_sub24': {'early': {'n': 322, 'n_minority': 151}, 'middle': {'n': 321, 'n_minority': 140}, 'late': {'n': 321, 'n_minority': 153}}}, 'person_grain': {'sub03_sub06': {'A_early': {'n': 162, 'n_minority': 80}, 'A_middle': {'n': 161, 'n_minority': 68}, 'A_late': {'n': 161, 'n_minority': 62}, 'B_early': {'n': 162, 'n_minority': 76}, 'B_middle': {'n': 161, 'n_minority': 73}, 'B_late': {'n': 161, 'n_minority': 76}}, 'sub04_sub05': {'A_early': {'n': 162, 'n_minority': 75}, 'A_middle': {'n': 161, 'n_minority': 80}, 'A_late': {'n': 161, 'n_minority': 70}, 'B_early': {'n': 162, 'n_minority': 75}, 'B_middle': {'n': 161, 'n_minority': 73}, 'B_late': {'n': 161, 'n_minority': 68}}, 'sub07_sub08': {'A_early': {'n': 162, 'n_minority': 70}, 'A_middle': {'n': 161, 'n_minority': 76}, 'A_late': {'n': 161, 'n_minority': 69}, 'B_early': {'n': 162, 'n_minority': 70}, 'B_middle': {'n': 161, 'n_minority': 80}, 'B_late': {'n': 161, 'n_minority': 79}}, 'sub09_sub10': {'A_early': {'n': 162, 'n_minority': 79}, 'A_middle': {'n': 161, 'n_minority': 79}, 'A_late': {'n': 161, 'n_minority': 78}, 'B_early': {'n': 162, 'n_minority': 72}, 'B_middle': {'n': 161, 'n_minority': 68}, 'B_late': {'n': 161, 'n_minority': 68}}, 'sub11_sub12': {'A_early': {'n': 162, 'n_minority': 77}, 'A_middle': {'n': 161, 'n_minority': 70}, 'A_late': {'n': 161, 'n_minority': 67}, 'B_early': {'n': 162, 'n_minority': 69}, 'B_middle': {'n': 161, 'n_minority': 63}, 'B_late': {'n': 161, 'n_minority': 78}}, 'sub13_sub14': {'A_early': {'n': 162, 'n_minority': 71}, 'A_middle': {'n': 161, 'n_minority': 78}, 'A_late': {'n': 161, 'n_minority': 80}, 'B_early': {'n': 162, 'n_minority': 74}, 'B_middle': {'n': 161, 'n_minority': 75}, 'B_late': {'n': 161, 'n_minority': 75}}, 'sub15_sub16': {'A_early': {'n': 162, 'n_minority': 66}, 'A_middle': {'n': 161, 'n_minority': 74}, 'A_late': {'n': 161, 'n_minority': 69}, 'B_early': {'n': 162, 'n_minority': 78}, 'B_middle': {'n': 161, 'n_minority': 73}, 'B_late': {'n': 161, 'n_minority': 79}}, 'sub17_sub18': {'A_early': {'n': 162, 'n_minority': 71}, 'A_middle': {'n': 161, 'n_minority': 59}, 'A_late': {'n': 161, 'n_minority': 63}, 'B_early': {'n': 162, 'n_minority': 72}, 'B_middle': {'n': 161, 'n_minority': 62}, 'B_late': {'n': 161, 'n_minority': 72}}, 'sub20_sub21': {'A_early': {'n': 161, 'n_minority': 75}, 'A_middle': {'n': 161, 'n_minority': 76}, 'A_late': {'n': 160, 'n_minority': 77}, 'B_early': {'n': 162, 'n_minority': 68}, 'B_middle': {'n': 161, 'n_minority': 79}, 'B_late': {'n': 161, 'n_minority': 71}}, 'sub23_sub24': {'A_early': {'n': 160, 'n_minority': 70}, 'A_middle': {'n': 160, 'n_minority': 80}, 'A_late': {'n': 160, 'n_minority': 72}, 'B_early': {'n': 162, 'n_minority': 53}, 'B_middle': {'n': 161, 'n_minority': 45}, 'B_late': {'n': 161, 'n_minority': 53}}}, 'min_minority_dyad_grain': 130, 'min_minority_person_grain': 45, 'reproduces_fact3_130': True, 'reproduces_fact3_45': True}

- **5_volume_matching**: {'ok': True, 'sample': {'sub03_sub06_k81': {'n_same': 81, 'n_other': 81, 'n_lie_same': 46, 'n_lie_other': 46, 'ok': True}, 'sub03_sub06_k162': {'n_same': 162, 'n_other': 162, 'n_lie_same': 80, 'n_lie_other': 80, 'ok': True}, 'sub03_sub06_k322': {'n_same': 322, 'n_other': 322, 'n_lie_same': 147, 'n_lie_other': 147, 'ok': True}, 'sub03_sub06_k484': {'n_same': 484, 'n_other': 484, 'n_lie_same': 210, 'n_lie_other': 210, 'ok': True}, 'sub03_sub06_k646': {'n_same': 646, 'n_other': 646, 'n_lie_same': 296, 'n_lie_other': 296, 'ok': True}}}

- **6_other_dyad_pool_exclusion**: {'ok': True, 'contributing_pair_ids': ['sub03_sub06', 'sub04_sub05', 'sub07_sub08', 'sub09_sub10', 'sub11_sub12', 'sub13_sub14', 'sub15_sub16', 'sub17_sub18', 'sub19_sub22', 'sub20_sub21', 'sub23_sub24']}

- **7_inner_cv_sensitivity**: {'max_abs_diff': 0.06578115117014538, 'exceeds_0.05': True, 'note': 'Reported measurement, not a pass/fail gate.'}

- **8_no_identity_columns**: {'forbidden_set': ['condition', 'condition_raw', 'dm_excluded_reason', 'dyad_trial_seq', 'fb_excluded_reason', 'observer_guess', 'outcome', 'pair_id', 'participant_id', 'partner_id', 'points', 'role', 'role_raw', 'round', 'session_id', 'session_order', 'trial'], 'checked_at': 'run() before .values'}

- **9_majority_class_sanity**: {'mean_auroc': 0.5, 'mean_balanced_accuracy': 0.5, 'ok': True}

- **10_reproducibility**: {'max_diff': 0.0, 'per_unit': {'tss_sub03_sub06_k322': 0.0, 'tss_sub04_sub05_k646': 0.0, 'skf_sub07_sub08_k646': 0.0}}

- **11_grid_n_jobs_invariance**: {'dyad_sampled': 'sub03_sub06', 'n_jobs_1_seconds': 0.5979394912719727, 'n_jobs_chosen_seconds': 1.3068397045135498, 'n_jobs_chosen': 6, 'metrics_identical': True, 'best_params_n1': {'clf__C': 10.0}, 'best_params_nchosen': {'clf__C': 10.0}}

- **12_metric_crosscheck**: {'dyad': 'sub03_sub06', 'sklearn_auroc': 0.4944041370793454, 'hand_auroc': 0.4944041370793455, 'match_to_1e6': True}

- **13_upstream_files_hashed**: {'hashes': {'models.py': '117ad942f6500edc839774f7f4d07cf4d68a179f18cc25b4de157bb45e43820e', 'gate.json': '900ffddb6b6546e4681549d6d2c5744d596f21bb498691f99d33b0125164dc5d', 'shard_manifest.json': '97966bd3a575b285b3476ef8c06568574b508ac1fb5e1059e7f3d0080bf20f6d', 'frozen_hypotheses.md': 'e8f7235601abc6dc1f6b4441af24959728598b78bd6792c346fb29aad4ebda31'}, 'frozen_hypotheses_matches_amended_hash': True}

- **14_data_integrity**: {'row_count_10648_before_nan_drop': True, 'total_frame_hash_verified': 'asserted in load_deceiver_frame() -- would have raised if mismatched'}

- **15_gate_verdicts_recorded**: {'designations_present': {'1A': 'confirmatory_on_power_confounded', '1B': 'exploratory_underpowered', '2_frozen_rungs': 'confirmatory', '2_other_rungs': 'exploratory', '3_k646': 'confirmatory', '3_other_rungs': 'exploratory'}, 'no_test_labelled_confirmatory_outside_hierarchy': True}

- **16_amendment_recorded**: {'amendment_2_present': True, 'n_dyads': 10, 'sub19_sub22_absent_from_tested_dyads': True}

- **17_plausibility**: {'max_auroc': 0.5704861111111111, 'dyads_flagged': [], 'leakage_suspicion': False, 'note': 'Flagging is for inspection, not automatic exclusion (exp3/exp4 precedent).'}

- **18_convergence**: {'n_convergence_warnings': {'ladder_tss': 0, 'ladder_skf': 0, 'ladder_other': 0, 'universal': 0}}


## Cross-experiment comparison (exp4)

{'per_dyad_diff_exp5_k646_minus_exp4_dyad_specific': {'sub03_sub06': 0.0, 'sub04_sub05': 0.0, 'sub07_sub08': 0.0, 'sub09_sub10': 0.0, 'sub11_sub12': 0.0, 'sub13_sub14': 0.0, 'sub15_sub16': 0.0, 'sub17_sub18': 0.0, 'sub20_sub21': 0.0, 'sub23_sub24': 0.0}, 'max_abs_diff': 0.0, 'note': "exp5's k=646 same-dyad rung is fit on the identical 646 rows and scored on the identical T_d as exp4's dyad_specific condition, in the same environment. Expected ~0 up to the inner-CV scheme (both use TimeSeriesSplit here); any systematic gap is worth reporting, not automatically a bug."}


## Limitations

- n=10, not the pre-registered n=11 (Amendment 2; sub19_sub22 structurally untestable).
- Analysis 1A (dyad-grain positional split) is confounded with participant identity: the early tercile is entirely deceiver A's trials, the late tercile entirely deceiver B's -- 'late is more distinguishable' cannot be told apart from 'B is more distinguishable than A' at this grain.
- Analysis 1B (participant-grain, unconfounded) is underpowered and exploratory -- reported with CIs, not dropped.
- The ladder's rung 5 (k=646) introduces the tested participant's own rows for the first time; a rise from rung 4 to rung 5 conflates relationship learning with the arrival of own-person data -- Analysis 3's control is the mitigation, not a full resolution.
- A most-recent-k 'recency ladder' (train on the k most recent rows before the held-out block, rather than a growing prefix) would separate 'more history' from 'more recent history' -- a genuinely different question S15 does not ask. Deliberately out of scope for this pass.
- All conditions fit in one Python-3.8/sklearn-1.3.2 remote environment; not directly comparable point-to-point to laptop-fit experiments.
