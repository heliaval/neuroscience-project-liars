# Experiment 6 -- Observer-Only Prediction (S16)

## Findings

Above-chance decoding of a partner's condition from an observer's EEG is consistent with the observer's brain responding differently to lies than to truths; it is NOT evidence the observer knows, senses, or acts on that difference. It is also consistent with mundane alternatives that are not detection at all: the deceiver behaving differently on lie trials (longer pauses, altered prosody, different button-press timing) and the observer's EEG tracking that observable behaviour; task-structure differences between lie and truth trials; or shared temporal structure between the two brains' recordings. exp6 cannot distinguish these.


The observer's brain did not become measurably more informative about the partner's deception across the interaction (median delta = +0.0238, 6 of 10 dyads positive, sign p = 0.7539, permutation p = 0.4043).


## Design

- n = 10 dyads (amended from pre-registered n=11; see `results/frozen_hypotheses.md` Amendment 3).
- Role: observer. Target: trial ground-truth condition; NOT observer_guess
- Feature set: reliable_plus_marginal (1770 cols). Model: logistic_regression (L2) only

The observer of seq[1,484] is the session-2 deceiver and the observer of seq[485,968] is the session-1 deceiver -- the reverse of exp5's identity assignment on the same seq bins (Fact 2). This partially breaks the participant-identity confound across the two experiments but does not remove positional confounds (session order, fatigue, electrode drift, task familiarity) within exp6 alone. Published as data (per-bin participant ids), not smoothed over.

sub19_sub22's observer rows span seq [1,484] only (its only session), so its late tercile (exp6's positional 'late' bin) contains zero observer rows -- structurally untestable. results/frozen_hypotheses.md Amendment 3 reduces exp6's n from 11 to 10 accordingly; sub19_sub22's rows remain in the other-dyad training pool.

exp1's already-run observer secondary put observer-only decodability at mean AUROC 0.5368 [0.5285, 0.5451] -- real but tiny, comparable to the deceiver-side signal (pooled AUROC 0.5338). The early/late comparison is therefore a comparison of two near-chance numbers; the S21 absolute-decodability null is run and reported first for this reason.


## Gate re-check (Amendment 3)

THRESHOLD_M = 60 (copied unchanged from the original gate).


| analysis | test-fold grain | minority | Clause A | Clause B | designation |
|---|---|---|---|---|---|

| 1A dyad-grain positional | 320-324 | 130 | PASS | 10/10 PASS | confirmatory_on_power_confounded |

| 1B participant-grain positional | 162/161/161 | 45 | FAIL | 8/10 FAIL | exploratory_underpowered |

| 2 S21 absolute-decodability null | same bins as 1A | 130 | PASS | 10/10 PASS | confirmatory |


Failing bins: {'sub23': 45, 'sub18': 59}


## Absolute decodability (S21) -- reported before the early/late comparison

Pooled across all ten dyads' positional bins, the fitted observer model's AUROC was 0.5054 against a label-shuffled null with mean 0.4999 (sd 0.0059), p = 0.1767.


## Positional early/middle/late per dyad (dyad-grain, AUROC)

| pair_id | early | middle | late | delta(late-early) |
|---|---|---|---|---|

| sub03_sub06 | 0.5022 | 0.5591 | 0.4855 | -0.0167 |

| sub04_sub05 | 0.5471 | 0.5357 | 0.4807 | -0.0665 |

| sub07_sub08 | 0.4870 | 0.5099 | 0.5208 | +0.0338 |

| sub09_sub10 | 0.4948 | 0.5451 | 0.5864 | +0.0916 |

| sub11_sub12 | 0.4928 | 0.5412 | 0.4753 | -0.0175 |

| sub13_sub14 | 0.4615 | 0.4702 | 0.5397 | +0.0783 |

| sub15_sub16 | 0.4871 | 0.5072 | 0.5009 | +0.0138 |

| sub17_sub18 | 0.5028 | 0.5009 | 0.5583 | +0.0554 |

| sub20_sub21 | 0.4839 | 0.5332 | 0.5920 | +0.1081 |

| sub23_sub24 | 0.4881 | 0.5317 | 0.4319 | -0.0562 |


## Primary claim (S20)

Median = +0.0238, 6/10 dyads positive, sign-test p = 0.7539, permutation p = 0.4043. 95% CI: [-0.0214, 0.0662]. supported = False.


## Exploratory (Benjamini-Hochberg corrected)

- **observer_within_person_late_minus_early**: median = +0.0127, 6/10 positive, sign-test p = 0.7539 (BH-adjusted: 0.8835)

- **observer_positional_middle_minus_early**: median = +0.0333, 8/10 positive, sign-test p = 0.1094 (BH-adjusted: 0.4114)

- **observer_positional_late_minus_middle**: median = +0.0023, 5/10 positive, sign-test p = 1 (BH-adjusted: 1)


observer_positional_late_minus_early (S20 paired, dyad-grain terciles) is the single pre-registered primary claim, no correction. observer_decodability_pooled (S21 permutation null) is a pre-registered secondary, no correction. observer_within_person_late_minus_early, observer_positional_middle_minus_early, observer_positional_late_minus_middle, and the 30 per-bin S21 nulls (dyad-grain) are exploratory, Benjamini-Hochberg corrected together at alpha=0.05 within that family (family size 33). Declared before any number below was computed, per results/frozen_hypotheses.md Amendment 3.


## Within-observer grain (1B, EXPLORATORY, UNDERPOWERED)

Fails the exp6-specific gate re-check (Clause A on sub23=45, Clause B at 8/10). Reported with CIs per the frozen file's own rule, never dropped.

Median = +0.0127, 6/10 positive, sign-test p = 0.7539. 95% CI: [-0.0334, 0.0602].


## Aggregate (NOT the test, S20)

- early: median AUROC = 0.4905
- middle: median AUROC = 0.5325
- late: median AUROC = 0.5108
- pooled observer decodability (S21): observed = 0.5054, p = 0.1767


## Cross-experiment vs exp5 (descriptive, not a test)

descriptive corroboration, not a test -- exp7 (S17) owns the paired observer-vs-deceiver comparison. exp5 scores the deceiver's own brain on the same trials/seq bins; exp6 scores the observer's brain. The identity-swap table (Fact 2) shows the two experiments' 'early' and 'late' brains belong to opposite participants.


## Environment

Remote: Python 3.8.10, sklearn 1.3.2. Laptop: Python 3.13.14, sklearn 1.8.0.


## Validation summary

- **1_role_purity**: {'ok': True, 'note': "asserted (full['role']==ROLE).all() in load_observer_frame()"}

- **2_no_self_dyad_leakage**: {'ok': True, 'per_dyad': {'sub03_sub06': {'n_train': 9190, 'pids': ['sub04_sub05', 'sub07_sub08', 'sub09_sub10', 'sub11_sub12', 'sub13_sub14', 'sub15_sub16', 'sub17_sub18', 'sub19_sub22', 'sub20_sub21', 'sub23_sub24'], 'clean': True}, 'sub04_sub05': {'n_train': 9190, 'pids': ['sub03_sub06', 'sub07_sub08', 'sub09_sub10', 'sub11_sub12', 'sub13_sub14', 'sub15_sub16', 'sub17_sub18', 'sub19_sub22', 'sub20_sub21', 'sub23_sub24'], 'clean': True}, 'sub07_sub08': {'n_train': 9190, 'pids': ['sub03_sub06', 'sub04_sub05', 'sub09_sub10', 'sub11_sub12', 'sub13_sub14', 'sub15_sub16', 'sub17_sub18', 'sub19_sub22', 'sub20_sub21', 'sub23_sub24'], 'clean': True}, 'sub09_sub10': {'n_train': 9190, 'pids': ['sub03_sub06', 'sub04_sub05', 'sub07_sub08', 'sub11_sub12', 'sub13_sub14', 'sub15_sub16', 'sub17_sub18', 'sub19_sub22', 'sub20_sub21', 'sub23_sub24'], 'clean': True}, 'sub11_sub12': {'n_train': 9190, 'pids': ['sub03_sub06', 'sub04_sub05', 'sub07_sub08', 'sub09_sub10', 'sub13_sub14', 'sub15_sub16', 'sub17_sub18', 'sub19_sub22', 'sub20_sub21', 'sub23_sub24'], 'clean': True}, 'sub13_sub14': {'n_train': 9190, 'pids': ['sub03_sub06', 'sub04_sub05', 'sub07_sub08', 'sub09_sub10', 'sub11_sub12', 'sub15_sub16', 'sub17_sub18', 'sub19_sub22', 'sub20_sub21', 'sub23_sub24'], 'clean': True}, 'sub15_sub16': {'n_train': 9190, 'pids': ['sub03_sub06', 'sub04_sub05', 'sub07_sub08', 'sub09_sub10', 'sub11_sub12', 'sub13_sub14', 'sub17_sub18', 'sub19_sub22', 'sub20_sub21', 'sub23_sub24'], 'clean': True}, 'sub17_sub18': {'n_train': 9190, 'pids': ['sub03_sub06', 'sub04_sub05', 'sub07_sub08', 'sub09_sub10', 'sub11_sub12', 'sub13_sub14', 'sub15_sub16', 'sub19_sub22', 'sub20_sub21', 'sub23_sub24'], 'clean': True}, 'sub20_sub21': {'n_train': 9192, 'pids': ['sub03_sub06', 'sub04_sub05', 'sub07_sub08', 'sub09_sub10', 'sub11_sub12', 'sub13_sub14', 'sub15_sub16', 'sub17_sub18', 'sub19_sub22', 'sub23_sub24'], 'clean': True}, 'sub23_sub24': {'n_train': 9194, 'pids': ['sub03_sub06', 'sub04_sub05', 'sub07_sub08', 'sub09_sub10', 'sub11_sub12', 'sub13_sub14', 'sub15_sub16', 'sub17_sub18', 'sub19_sub22', 'sub20_sub21'], 'clean': True}}}

- **3_identical_bins_across_arms**: {'ok': True, 'note': "True by construction -- run_universal_and_score_bins/run_majority both index blocks[d]['dyad_tercile_bins']/['person_tercile_bins'] directly; the S21 null reuses the Universal model's own cached y_true/y_score for the identical bin, never a separate scoring pass."}

- **4_bin_integrity_and_mirror**: {'ok': False, 'dyad_grain': {'sub03_sub06': {'early': {'n': 323, 'n_minority': 148}, 'middle': {'n': 323, 'n_minority': 148}, 'late': {'n': 322, 'n_minority': 158}}, 'sub04_sub05': {'early': {'n': 323, 'n_minority': 156}, 'middle': {'n': 323, 'n_minority': 145}, 'late': {'n': 322, 'n_minority': 141}}, 'sub07_sub08': {'early': {'n': 323, 'n_minority': 146}, 'middle': {'n': 323, 'n_minority': 139}, 'late': {'n': 322, 'n_minority': 160}}, 'sub09_sub10': {'early': {'n': 323, 'n_minority': 161}, 'middle': {'n': 323, 'n_minority': 150}, 'late': {'n': 322, 'n_minority': 136}}, 'sub11_sub12': {'early': {'n': 323, 'n_minority': 147}, 'middle': {'n': 323, 'n_minority': 136}, 'late': {'n': 322, 'n_minority': 141}}, 'sub13_sub14': {'early': {'n': 323, 'n_minority': 154}, 'middle': {'n': 323, 'n_minority': 155}, 'late': {'n': 322, 'n_minority': 150}}, 'sub15_sub16': {'early': {'n': 323, 'n_minority': 140}, 'middle': {'n': 323, 'n_minority': 147}, 'late': {'n': 322, 'n_minority': 152}}, 'sub17_sub18': {'early': {'n': 323, 'n_minority': 130}, 'middle': {'n': 323, 'n_minority': 135}, 'late': {'n': 322, 'n_minority': 134}}, 'sub20_sub21': {'early': {'n': 322, 'n_minority': 160}, 'middle': {'n': 322, 'n_minority': 145}, 'late': {'n': 322, 'n_minority': 150}}, 'sub23_sub24': {'early': {'n': 322, 'n_minority': 151}, 'middle': {'n': 321, 'n_minority': 140}, 'late': {'n': 321, 'n_minority': 153}}}, 'person_grain': {'sub03_sub06': {'sub06_early': {'n': 162, 'n_minority': 80}, 'sub06_middle': {'n': 161, 'n_minority': 68}, 'sub06_late': {'n': 161, 'n_minority': 62}, 'sub03_early': {'n': 162, 'n_minority': 76}, 'sub03_middle': {'n': 161, 'n_minority': 73}, 'sub03_late': {'n': 161, 'n_minority': 76}}, 'sub04_sub05': {'sub05_early': {'n': 162, 'n_minority': 75}, 'sub05_middle': {'n': 161, 'n_minority': 80}, 'sub05_late': {'n': 161, 'n_minority': 70}, 'sub04_early': {'n': 162, 'n_minority': 75}, 'sub04_middle': {'n': 161, 'n_minority': 73}, 'sub04_late': {'n': 161, 'n_minority': 68}}, 'sub07_sub08': {'sub07_early': {'n': 162, 'n_minority': 70}, 'sub07_middle': {'n': 161, 'n_minority': 76}, 'sub07_late': {'n': 161, 'n_minority': 69}, 'sub08_early': {'n': 162, 'n_minority': 70}, 'sub08_middle': {'n': 161, 'n_minority': 80}, 'sub08_late': {'n': 161, 'n_minority': 79}}, 'sub09_sub10': {'sub10_early': {'n': 162, 'n_minority': 79}, 'sub10_middle': {'n': 161, 'n_minority': 79}, 'sub10_late': {'n': 161, 'n_minority': 78}, 'sub09_early': {'n': 162, 'n_minority': 72}, 'sub09_middle': {'n': 161, 'n_minority': 68}, 'sub09_late': {'n': 161, 'n_minority': 68}}, 'sub11_sub12': {'sub11_early': {'n': 162, 'n_minority': 77}, 'sub11_middle': {'n': 161, 'n_minority': 70}, 'sub11_late': {'n': 161, 'n_minority': 67}, 'sub12_early': {'n': 162, 'n_minority': 69}, 'sub12_middle': {'n': 161, 'n_minority': 63}, 'sub12_late': {'n': 161, 'n_minority': 78}}, 'sub13_sub14': {'sub14_early': {'n': 162, 'n_minority': 71}, 'sub14_middle': {'n': 161, 'n_minority': 78}, 'sub14_late': {'n': 161, 'n_minority': 80}, 'sub13_early': {'n': 162, 'n_minority': 74}, 'sub13_middle': {'n': 161, 'n_minority': 75}, 'sub13_late': {'n': 161, 'n_minority': 75}}, 'sub15_sub16': {'sub16_early': {'n': 162, 'n_minority': 66}, 'sub16_middle': {'n': 161, 'n_minority': 74}, 'sub16_late': {'n': 161, 'n_minority': 69}, 'sub15_early': {'n': 162, 'n_minority': 78}, 'sub15_middle': {'n': 161, 'n_minority': 73}, 'sub15_late': {'n': 161, 'n_minority': 79}}, 'sub17_sub18': {'sub18_early': {'n': 162, 'n_minority': 71}, 'sub18_middle': {'n': 161, 'n_minority': 59}, 'sub18_late': {'n': 161, 'n_minority': 63}, 'sub17_early': {'n': 162, 'n_minority': 72}, 'sub17_middle': {'n': 161, 'n_minority': 62}, 'sub17_late': {'n': 161, 'n_minority': 72}}, 'sub20_sub21': {'sub20_early': {'n': 161, 'n_minority': 75}, 'sub20_middle': {'n': 161, 'n_minority': 76}, 'sub20_late': {'n': 160, 'n_minority': 77}, 'sub21_early': {'n': 162, 'n_minority': 68}, 'sub21_middle': {'n': 161, 'n_minority': 79}, 'sub21_late': {'n': 161, 'n_minority': 71}}, 'sub23_sub24': {'sub24_early': {'n': 160, 'n_minority': 70}, 'sub24_middle': {'n': 160, 'n_minority': 80}, 'sub24_late': {'n': 160, 'n_minority': 72}, 'sub23_early': {'n': 162, 'n_minority': 53}, 'sub23_middle': {'n': 161, 'n_minority': 45}, 'sub23_late': {'n': 161, 'n_minority': 53}}}, 'min_minority_dyad_grain': 130, 'min_minority_person_grain': 45, 'reproduces_fact4_130': True, 'reproduces_fact4_45': True, 'mirror_check_vs_deceiver_rows_ok': False, 'mirror_detail_sample': {'sub03_sub06_early': {'observer_n': 323, 'observer_n_minority': 148, 'deceiver_n': 322, 'deceiver_n_minority': 147, 'identical': False}, 'sub03_sub06_middle': {'observer_n': 323, 'observer_n_minority': 148, 'deceiver_n': 324, 'deceiver_n_minority': 149, 'identical': False}, 'sub03_sub06_late': {'observer_n': 322, 'observer_n_minority': 158, 'deceiver_n': 322, 'deceiver_n_minority': 158, 'identical': True}, 'sub04_sub05_early': {'observer_n': 323, 'observer_n_minority': 156, 'deceiver_n': 322, 'deceiver_n_minority': 155, 'identical': False}, 'sub04_sub05_middle': {'observer_n': 323, 'observer_n_minority': 145, 'deceiver_n': 324, 'deceiver_n_minority': 146, 'identical': False}, 'sub04_sub05_late': {'observer_n': 322, 'observer_n_minority': 141, 'deceiver_n': 322, 'deceiver_n_minority': 141, 'identical': True}}}

- **5_target_correctness**: {'match_rate': 1.0, 'n_sampled': 200, 'ok': True}

- **6_nan_drop**: {'note': 'checked and recorded at run() call site (n_dropped_nan block)'}

- **7_identity_swap_table**: {'ok': True, 'per_dyad': {'sub03_sub06': {'observer_early': 'sub06', 'observer_late': 'sub03', 'deceiver_early': ['sub03'], 'deceiver_late': ['sub06'], 'swapped': True}, 'sub04_sub05': {'observer_early': 'sub05', 'observer_late': 'sub04', 'deceiver_early': ['sub04'], 'deceiver_late': ['sub05'], 'swapped': True}, 'sub07_sub08': {'observer_early': 'sub07', 'observer_late': 'sub08', 'deceiver_early': ['sub08'], 'deceiver_late': ['sub07'], 'swapped': True}, 'sub09_sub10': {'observer_early': 'sub10', 'observer_late': 'sub09', 'deceiver_early': ['sub09'], 'deceiver_late': ['sub10'], 'swapped': True}, 'sub11_sub12': {'observer_early': 'sub11', 'observer_late': 'sub12', 'deceiver_early': ['sub12'], 'deceiver_late': ['sub11'], 'swapped': True}, 'sub13_sub14': {'observer_early': 'sub14', 'observer_late': 'sub13', 'deceiver_early': ['sub13'], 'deceiver_late': ['sub14'], 'swapped': True}, 'sub15_sub16': {'observer_early': 'sub16', 'observer_late': 'sub15', 'deceiver_early': ['sub15'], 'deceiver_late': ['sub16'], 'swapped': True}, 'sub17_sub18': {'observer_early': 'sub18', 'observer_late': 'sub17', 'deceiver_early': ['sub17'], 'deceiver_late': ['sub18'], 'swapped': True}, 'sub20_sub21': {'observer_early': 'sub20', 'observer_late': 'sub21', 'deceiver_early': ['sub21'], 'deceiver_late': ['sub20'], 'swapped': True}, 'sub23_sub24': {'observer_early': 'sub24', 'observer_late': 'sub23', 'deceiver_early': ['sub23'], 'deceiver_late': ['sub24'], 'swapped': True}}}

- **8_no_identity_columns**: {'forbidden_set': ['condition', 'condition_raw', 'dm_excluded_reason', 'dyad_trial_seq', 'fb_excluded_reason', 'observer_guess', 'outcome', 'pair_id', 'participant_id', 'partner_id', 'points', 'role', 'role_raw', 'round', 'session_id', 'session_order', 'trial'], 'checked_at': 'run() before .values'}

- **9_majority_class_sanity**: {'mean_auroc': 0.5, 'mean_balanced_accuracy': 0.5, 'ok': True}

- **10_reproducibility**: {'max_diff': 0.0, 'per_dyad': {'sub03_sub06': 0.0, 'sub04_sub05': 0.0, 'sub07_sub08': 0.0}}

- **11_grid_n_jobs_invariance**: {'dyad_sampled': 'sub03_sub06', 'n_jobs_1_seconds': 119.61130452156067, 'n_jobs_chosen_seconds': 74.26742935180664, 'n_jobs_chosen': 6, 'metrics_identical': True, 'best_params_n1': {'clf__C': 0.001}, 'best_params_nchosen': {'clf__C': 0.001}}

- **12_metric_crosscheck**: {'dyad': 'sub03_sub06', 'sklearn_auroc': 0.48548934856437176, 'hand_auroc': 0.4854893485643717, 'match_to_1e6': True}

- **13_upstream_files_hashed**: {'hashes': {'models.py': '117ad942f6500edc839774f7f4d07cf4d68a179f18cc25b4de157bb45e43820e', 'gate.json': '900ffddb6b6546e4681549d6d2c5744d596f21bb498691f99d33b0125164dc5d', 'shard_obs_manifest.json': '6ea9d334c19d7108f93c6989ac7699583956921e726b6833c9d038bdb6ff6263', 'frozen_hypotheses.md': 'f312b0fbd33ec494013d7af03c5fba1d3deb753f7182a607c67fb55c445d05fe'}, 'models_py_matches_exp5': True, 'frozen_hypotheses_matches_amended_hash': True}

- **14_data_integrity**: {'row_count_10648_before_nan_drop': True, 'total_frame_hash_verified': 'asserted in load_observer_frame() -- would have raised if mismatched'}

- **15_s16_language_guard**: {'note': 'executed in write_outputs() against the final assembled results dict and markdown; see meta.language_guard_result for the actual scan outcome', 'n_strings_scanned': 639, 'hits': [], 'clean': True, 'evidentiary_conditions_present': True}

- **16_amendment_and_designation_recorded**: {'amendment_3_present': True, 'n_dyads': 10, 'sub19_sub22_absent_from_tested_dyads': True, 'only_hierarchy_labelled_confirmatory': True, 'supported_flag_recomputed_matches': True}

- **17_plausibility**: {'max_auroc': 0.5920155038759689, 'dyads_flagged': [], 'leakage_suspicion': False, 'note': 'Given Fact 5 (prior expectation ~0.53-0.54), anything above 0.75 on observer rows is far more likely a leak than a finding.'}

- **18_convergence**: {'n_convergence_warnings': {'universal': 0}}


## Limitations

- n=10, not the pre-registered n=11 (Amendment 3; sub19_sub22 structurally untestable).
- Analysis 1A (dyad-grain positional split) is confounded with participant identity, in the opposite direction to exp5: the early tercile's observer and the late tercile's observer are different people. This is published as data (per-bin participant ids), not smoothed over.
- Analysis 1B (within-observer grain, unconfounded) is underpowered and exploratory -- reported with CIs, not dropped.
- The S21 null holds the fitted model fixed and permutes only the scored bin's labels; it tests whether the model's ranking of these trials beats chance, not whether the tuning procedure itself would find signal in noise. This is the same class of null exp1 used.
- The observer_guess behavioural-corroboration link is deliberately not built here -- see interpretation.observer_guess_link_out_of_scope. Absent it, exp6 alone cannot license any claim stronger than statistical decodability.
- All conditions fit in one Python-3.8/sklearn-1.3.2 remote environment; not directly comparable point-to-point to laptop-fit experiments.
