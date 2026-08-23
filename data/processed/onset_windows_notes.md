# Section 18 sliding pre-decision window feature extraction -- evidence file

This file plays the same role for §18 that `eeg_preprocessing_notes.md` plays
for §9: it records what was measured, what could not be built as planned, and
what was verified, so a reader does not have to re-run `scripts/build_onset_windows.py`
to trust its output.

## 1. What could not be followed as planned

§18's illustrative ladder is six 250 ms windows spanning -1500 to 0 ms.
Re-measured directly from `data/raw/Preprocessed/DecisionMaking/*.mat`
(23 session files, one archive extent across all of them):

```
fs=100  t=[-500, 2990] ms  n_samples=350
pre-decision samples (t<=0): 51, from -500 to 0 ms
-1500 ms present in t axis: False
```

Four of §18's six windows (-1500..-1250, -1250..-1000, -1000..-750,
-750..-500) lie entirely outside this extent; a fifth (-500..-250) and the
sixth (-250..0) are the only two that exist at all. This is the identical
finding the §9 preprocessing pass already reported in this same file
("Windows chosen", above) and in `src/preprocessing.py`'s module docstring:
the archive's pre-onset extent for DecisionMaking is -500 ms, not -1500 ms
or -2000 ms, and building the longer window would require re-deriving
epochs from `Raw.zip`, which was not downloaded then and is not downloaded
now -- it was outside the approved download scope.

## 2. The substituted scheme

**Primary ladder -- five 100 ms bins tiling the entire available pre-decision
extent, no gaps, no overlaps.**

Scaling rationale: the available extent is 3x shorter than §18 assumed
(500 ms vs 1500 ms), so the bin width shrinks 2.5x (250 -> 100 ms) and the
bin count goes 6 -> 5. 100 ms is the finest bin that is a whole number of
samples at 100 Hz and still leaves more than a handful of samples per
window. `pp._slice_window` slices inclusive of both endpoints, so each
window's `end_ms` is pulled back one sample (10 ms) to tile without a
shared boundary sample.

| id | nominal bin | `start_ms` (slice) | `end_ms` (slice) | samples | t values covered |
|---|---|---|---|---|---|
| `w1` | -500 to -400 ms | -500 | -410 | 10 | -500 ... -410 |
| `w2` | -400 to -300 ms | -400 | -310 | 10 | -400 ... -310 |
| `w3` | -300 to -200 ms | -300 | -210 | 10 | -300 ... -210 |
| `w4` | -200 to -100 ms | -200 | -110 | 10 | -200 ... -110 |
| `w5` | -100 to 0 ms | -100 | 0 | 11 | -100 ... 0 |

10 + 10 + 10 + 10 + 11 = 51 = the full pre-decision extent. Confirmed by
the tiling self-check (Section 5 below).

**Sensitivity ladder -- two 250 ms bins**, so a null on the primary ladder
cannot be waved away as a pure short-window artifact. Pre-declared
EXPLORATORY, corrected within its own family, never used to set an onset.

| id | nominal bin | `start_ms` | `end_ms` | samples |
|---|---|---|---|---|
| `s1` | -500 to -250 ms | -500 | -260 | 25 |
| `s2` | -250 to 0 ms | -250 | 0 | 26 |

**Full-extent anchor -- zero cost.** `single_brain.parquet`'s existing
`dm_pre` columns already cover exactly (-500, 0), the union of the primary
ladder. Reporting the LODO AUROC on that column set is descriptive only,
outside every correction family.

## 3. Band reliability

Applying `src/features.py`'s own `reliability_tier(window_duration_s,
band_low_hz)` rule (n_cycles = duration x band_low; >=3 reliable, >=1
marginal, <1 unreliable) unchanged, to the new windows:

```
Band reliability by window (rule: n_cycles = duration_s * band_low_hz; >=3 reliable, >=1 marginal, <1 unreliable):
  w1 (100 ms): delta=unr(0.05) theta=unr(0.40) alpha=unr(0.80) beta=mar(1.30) gamma=rel(3.00)
  w2 (100 ms): delta=unr(0.05) theta=unr(0.40) alpha=unr(0.80) beta=mar(1.30) gamma=rel(3.00)
  w3 (100 ms): delta=unr(0.05) theta=unr(0.40) alpha=unr(0.80) beta=mar(1.30) gamma=rel(3.00)
  w4 (100 ms): delta=unr(0.05) theta=unr(0.40) alpha=unr(0.80) beta=mar(1.30) gamma=rel(3.00)
  w5 (100 ms): delta=unr(0.05) theta=unr(0.40) alpha=unr(0.80) beta=mar(1.30) gamma=rel(3.00)
  s1 (250 ms): delta=unr(0.12) theta=mar(1.00) alpha=mar(2.00) beta=rel(3.25) gamma=rel(7.50)
  s2 (250 ms): delta=unr(0.12) theta=mar(1.00) alpha=mar(2.00) beta=rel(3.25) gamma=rel(7.50)
```

Alpha is unavailable (unreliable, 0.8 cycles) at 100 ms and available
(marginal, 2.0 cycles) at 250 ms. This is the project's own `reliability_tier`
rule applied unchanged, not a new standard invented for exp8 -- the same rule
excludes delta everywhere in `feature_engineering_notes.md`. At 100 ms the
headline feature set is beta + gamma power + all six `td_*` stats: 30 ch x 2
bands = 60 power columns plus 30 ch x 6 stats = 180 time-domain columns = 240
columns per window per role. Unreliable columns (delta, theta, alpha at
100 ms) are computed and stored, never dropped silently -- the reliability
tier travels with each column in `onset_feature_dictionary.csv` and the
modeling module selects on it.

## 4. Extraction and exclusion counts

Measured by running `scripts/build_onset_windows.py` end to end over all 23
session files, both roles, all 7 windows (5 primary + 2 sensitivity):

```
extracted participant-trials: 21290
excluded participant-trials: 974
excluded_reason breakdown: alignment_not_recoverable=968, rejected_in_archive=6
```

These match `eeg_preprocessing_notes.md`'s "Extraction and exclusion counts"
table exactly, because this script reuses the identical alignment code
(`pp.compute_alignment`) against the identical archive files -- it is a
different window list over the same trial/epoch correspondence, not a
different alignment. `sub19_sub22`'s dyad spans two sessions
(`Player_sub19_Observer_sub22`, `Player_sub22_Observer_sub19`); the second
of those (`Player_sub22_Observer_sub19`, 968 participant-trials, matching
`sub22_sub19`'s own row count) comes out `alignment_not_recoverable` for
100% of its rows, as expected -- its class-label sequence does not
correspond to the trial table at all (a pre-existing archive defect
documented in `eeg_preprocessing_notes.md`, not introduced by this script).
The `sub19_sub22` dyad's *other* session (`Player_sub19_Observer_sub22`)
extracts cleanly, which is why the dyad as a whole is not excluded from
exp8 (n stays at 11 -- exp8 uses no chronological block, so the reason
Amendments 1-3 excluded this dyad from exp3-exp6/exp7 does not apply here;
see Amendment 5).

Output table: `data/processed/features/onset_windows.parquet`, 22,264 rows
(one row per participant-trial per role -- matches `trial_table.csv`'s long
form exactly), 2,320 columns (2,310 feature columns: 7 windows x 30
channels x (5 bands + 6 td stats); plus 10 key/housekeeping columns:
`pair_id, session_id, participant_id, round, trial, dyad_trial_seq, role,
condition, extracted, excluded_reason`). Excluded rows carry NaN feature
values, never imputed.

## 5. Verification performed

- **Tiling assert.** `check_window_tiling()` confirmed the five primary
  windows cover exactly the 51 pre-decision samples with no gaps and no
  overlaps:
  ```
  w1: bin [-500, -400) -> slice [-500, -410] ms, 10 samples
  w2: bin [-400, -300) -> slice [-400, -310] ms, 10 samples
  w3: bin [-300, -200) -> slice [-300, -210] ms, 10 samples
  w4: bin [-200, -100) -> slice [-200, -110] ms, 10 samples
  w5: bin [-100, 0) -> slice [-100, 0] ms, 11 samples
  tiling OK: 51 samples, -500..0 ms, no gaps, no overlaps
  ```

- **`dm_pre` reconstruction check.** Built a temporary (-500, 0) window on
  session `Player_sub01_Observer_sub02` and compared its `pow_gamma_*`
  values against `single_brain.parquet`'s existing `pow_gamma_<ch>_dm_pre`
  column for the same trials (the s2 window, -250..0 ms, is deliberately
  *not* the same window as `dm_pre`, -500..0 ms, so only this temporary
  full-extent window is a genuine identity check). First attempt failed
  (`max_abs_rel_diff = 8.758e-01`) because `pp._slice_window` casts its
  output to `float32`, which silently discards the imaginary part of a
  complex (Hilbert analytic) signal -- confirmed by inspection of
  `preprocessing.py:392`. Fixed by adding a local
  `_slice_analytic_window` helper in `build_onset_windows.py` that performs
  the identical mask-and-gather logic without the lossy cast (this mirrors
  `src/features.py`'s own `process_session`, which never routes the
  analytic signal through `_slice_window` either, for the same reason).
  After the fix: `max_abs_rel_diff = 0.000e+00`, `MATCH` -- exact
  reconstruction, not merely within tolerance.

- **Determinism check.** Re-ran `process_session_onset` twice on
  `Player_sub01_Observer_sub02` and compared every `(role, window)` cell's
  `pow` and `td` arrays with `np.array_equal(..., equal_nan=True)`:
  `determinism check: PASS` for every cell.

- **Column count check.** `2310` feature columns in the written parquet,
  matching the design arithmetic (7 windows x 30 channels x (5 bands + 6 td
  stats) = 2310) exactly.
