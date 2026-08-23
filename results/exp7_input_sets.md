# Experiment 7 -- One Brain vs Two Brains (S17)

**Research question:** S17: does adding the observer's brain, or inter-brain features, improve prediction over the deceiver's brain alone?

**Design:** leave_one_dyad_out, grain=trial, n_dyads=11, n_trials=10158

**Canonical index sha256:** `4006a919f2bb9d12dcb1e40fbbeb7f95ce5f10333ec8c3882ec1d4db85be10bd`

**Delta convention:** AUROC(named_first) - AUROC(named_second) in every test id


## Gate re-check (Amendment 4)

| analysis | test-fold grain | smallest fold | smallest minority | Clause A | Clause B | designation |
|---|---|---|---|---|---|---|
| exp7 LODO | one dyad's trials | 484 (sub19_sub22) | 236 (sub19_sub22) | PASS | 11/11 PASS | CONFIRMATORY |

## Claim hierarchy

- **PRIMARY (confirmatory, uncorrected):** `both_brains_vs_deceiver_eeg`
- **SECONDARY (confirmatory, uncorrected):** `observer_eeg_vs_deceiver_eeg`, `interbrain_vs_deceiver_eeg`, `eeg_plus_behavioral_vs_deceiver_eeg`
- **EXPLORATORY (Benjamini-Hochberg alpha=0.05):** `observer_eeg_vs_both_brains`, `observer_eeg_vs_interbrain`, `observer_eeg_vs_eeg_plus_behavioral`, `both_brains_vs_interbrain`, `both_brains_vs_eeg_plus_behavioral`, `interbrain_vs_eeg_plus_behavioral`
- Per-input-set absolute AUROCs are DESCRIPTIVE only, never a test.

## Behavioral exclusions

- `pinfo_bart_score`: per-participant constant; identity proxy (S19)
- `trials_so_far`: r=1.000 with dyad_trial_seq; positional, not behavioural
- `round`: same reason as trials_so_far
- `outcome`: post-hoc w.r.t. label (M.LEAKAGE_COLUMNS)

## Input sets (descriptive)

| input set | width | median AUROC | CI95 | dyads above chance |
|---|---|---|---|---|
| Deceiver's EEG only (`deceiver_eeg`) | 1770 | 0.5145 | [0.5023, 0.5320] | 9/11 |
| Observer's EEG only (`observer_eeg`) | 1770 | 0.5166 | [0.5023, 0.5257] | 9/11 |
| Both participants' EEG (`both_brains`) | 3540 | 0.5274 | [0.5126, 0.5345] | 11/11 |
| Inter-brain / dyadic features (`interbrain`) | 1560 | 0.4880 | [0.4842, 0.5057] | 4/11 |
| Deceiver's EEG + behavioral history (`eeg_plus_behavioral`) | 1776 | 0.5142 | [0.5017, 0.5319] | 8/11 |

## Tests

| test | designation | median delta | n+/n-  | sign p | permutation p | supported |
|---|---|---|---|---|---|---|
| `both_brains_vs_deceiver_eeg` | confirmatory | +0.0073 | 7/4 | 0.5488 | 0.2887 | False |
| `observer_eeg_vs_deceiver_eeg` | confirmatory | +0.0008 | 6/5 | 1 | 1 | N/A |
| `interbrain_vs_deceiver_eeg` | confirmatory | -0.0213 | 2/9 | 0.06543 | 0.0309 | N/A |
| `eeg_plus_behavioral_vs_deceiver_eeg` | confirmatory | -0.0005 | 5/6 | 1 | 0.7551 | N/A |
| `observer_eeg_vs_both_brains` | exploratory | -0.0097 | 3/8 | 0.2266 | 0.1324 | N/A |
| `observer_eeg_vs_interbrain` | exploratory | +0.0271 | 9/2 | 0.06543 | 0.0293 | N/A |
| `observer_eeg_vs_eeg_plus_behavioral` | exploratory | -0.0004 | 5/6 | 1 | 1 | N/A |
| `both_brains_vs_interbrain` | exploratory | +0.0325 | 11/0 | 0.0009766 | 0.0294 | N/A |
| `both_brains_vs_eeg_plus_behavioral` | exploratory | +0.0083 | 8/3 | 0.2266 | 0.2854 | N/A |
| `interbrain_vs_eeg_plus_behavioral` | exploratory | -0.0210 | 2/9 | 0.06543 | 0.0337 | N/A |

### Exploratory family BH-adjusted p-values

| test | sign p (BH) | permutation p (BH) |
|---|---|---|
| `observer_eeg_vs_both_brains` | 0.2719 | 0.1986 |
| `observer_eeg_vs_interbrain` | 0.1309 | 0.06739 |
| `observer_eeg_vs_eeg_plus_behavioral` | 1 | 1 |
| `both_brains_vs_interbrain` | 0.005859 | 0.06739 |
| `both_brains_vs_eeg_plus_behavioral` | 0.2719 | 0.3424 |
| `interbrain_vs_eeg_plus_behavioral` | 0.1309 | 0.06739 |

## Per-dyad results

| pair_id | n_test | n_train | deceiver_eeg | observer_eeg | both_brains | interbrain | eeg_plus_behavioral | majority |
|---|---|---|---|---|---|---|---|---|
| sub03_sub06 | 968 | 9190 | 0.5161 | 0.5128 | 0.5225 | 0.4810 | 0.5142 | 0.5000 |
| sub04_sub05 | 968 | 9190 | 0.5090 | 0.5233 | 0.5274 | 0.4875 | 0.5063 | 0.5000 |
| sub07_sub08 | 968 | 9190 | 0.5498 | 0.5166 | 0.5463 | 0.5138 | 0.5427 | 0.5000 |
| sub09_sub10 | 968 | 9190 | 0.5002 | 0.5409 | 0.5346 | 0.5068 | 0.4966 | 0.5000 |
| sub11_sub12 | 968 | 9190 | 0.5500 | 0.5042 | 0.5291 | 0.5291 | 0.5540 | 0.5000 |
| sub13_sub14 | 968 | 9190 | 0.4945 | 0.4953 | 0.5018 | 0.4862 | 0.4936 | 0.5000 |
| sub15_sub16 | 968 | 9190 | 0.5187 | 0.5013 | 0.5115 | 0.4743 | 0.5188 | 0.5000 |
| sub17_sub18 | 968 | 9190 | 0.4839 | 0.5186 | 0.5032 | 0.4880 | 0.4844 | 0.5000 |
| sub19_sub22 | 484 | 9674 | 0.5145 | 0.5210 | 0.5429 | 0.5004 | 0.5215 | 0.5000 |
| sub20_sub21 | 966 | 9192 | 0.5087 | 0.5375 | 0.5363 | 0.4875 | 0.5082 | 0.5000 |
| sub23_sub24 | 964 | 9194 | 0.5433 | 0.4827 | 0.5033 | 0.4899 | 0.5446 | 0.5000 |

## Validations

- **V1_row_count_and_grain**: passed=True
- **V2_target_consistency**: passed=True
- **V3_matrix_alignment**: passed=True
- **V4_identical_folds**: passed=True
- **V5_lodo_leakage**: passed=True
- **V6_inner_cv_grouping**: passed=True
- **V7_gate_recheck**: passed=True
- **V8_feature_set_integrity**: passed=True
- **V9_grid_n_jobs_invariance**: passed=True
- **V10_reproducibility**: passed=True
- **V11_metric_cross_check**: passed=True
- **V12_majority_class_sanity**: passed=True
- **V13_plausibility**: passed=True
- **V14_upstream_file_integrity**: passed=True
- **V15_cross_experiment_consistency**: passed=True
- **V16_interpretation_string_scan**: passed=True

## Interpretation guard (inter-brain / interbrain input set)

Any statement about the `interbrain` input set's inter-brain synchrony features describes a statistical relationship between two simultaneously recorded EEG signals, NOT evidence of communication between brains. Small-sample PLV/coherency estimates carry a known upward bias (see data/processed/feature_engineering_notes.md); the interbrain input set's absolute AUROC should be read with that caveat in mind, not as unbiased decodability.

## Limitations

- Every exp7 claim is a comparison between input sets on the same folds; S21's label-shuffling permutation null (absolute decodability) is not run here (it names exp1/exp2/exp6/exp8, not exp7). For an above-chance reference point, exp2's already-computed LODO null (mean 0.5004, sd 0.0077) applies to the deceiver_eeg-equivalent participant-grain analysis, not exp7's trial-grain numbers directly.
- `eeg_plus_behavioral` is anchored on `deceiver_eeg`, not `both_brains`; its delta isolates the behavioural contribution against the same base as the other three anchored tests, not against the widest EEG set.
