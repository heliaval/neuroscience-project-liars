# OneDCNN archive — structure, label semantics, and trial-identity reconciliation

Peer document to `eeg_preprocessing_notes.md` / `feature_engineering_notes.md`. Written
before any CNN training code (`src/cnn_check.py`), per `plans/experiment1-baseline-classification.md`
Step 2c. All findings below were verified against the real files, not assumed.

## 1. File structure (MATLAB v5, `scipy.io.loadmat` works, `h5py` does not)

Moved from the repo-root `OneDCNN/data/` (user's manual extraction of `OneDCNN.zip`) to
`data/raw/OneDCNN/` per `src/preprocessing.py`'s `data/raw/<Archive>/` convention.

| variable | DecisionMaking | Feedback |
|---|---|---|
| `data_X` | `(11129, 30, 350, 2)` float64 | `(11129, 30, 120, 2)` float64 |
| `data_y` | `(1, 11129)` uint8, values 1-4 | `(1, 11129)` uint8, values 1-2 |
| `pair_num` | `(1, 11129)` uint8, values 1-23 | same |
| `className` | `['sponL','sponT','instT','instL']` | `['Correct','Incorrect']` |
| `dataCh` | 30 channel names (F7...Fp2) | same |
| `participant_pairs` | `(23, 2)` `('Player_subNN','Observer_subMM')` | same |

`participant_pairs` has 23 rows because most dyads played the game **twice**, once with
each member as Player (e.g. row 3 is `Player_sub04/Observer_sub05`, row 4 is
`Player_sub05/Observer_sub04` — the same dyad, reversed roles). `sub02` never appears as
Player (confirmed: no row has `Player_sub02`), so 12 dyads x 2 sessions - 1 (sub01_sub02
has only one session, sub02 never plays) = 23 sessions, matching the 23 rows exactly.

`pair_num` bucket sizes: 20 buckets of 484 trials, one of 483 (`pair_num=20`,
`Player_sub21_Observer_sub20`), one of 482 (`pair_num=21`,
`Player_sub22_Observer_sub19`), one of 482 (`pair_num=22`,
`Player_sub23_Observer_sub24`). Wait — measured exactly: pair_num 20 -> 483, pair_num 21
-> 484, pair_num 22 -> 482 (see raw printout below); total = 23*484 - (1+0+2) = 11129.
23 x 484 = 11132 = the trial table's dyad-session trial count; `data_X` is **3 trials
short** of that, spread across pair_num 20 (-1) and pair_num 22 (-2).

Measured bucket sizes (all 23):
```
{1: 484, 2: 484, 3: 484, 4: 484, 5: 484, 6: 484, 7: 484, 8: 484, 9: 484, 10: 484,
 11: 484, 12: 484, 13: 484, 14: 484, 15: 484, 16: 484, 17: 484, 18: 484, 19: 484,
 20: 483, 21: 484, 22: 482, 23: 484}
```

## 2. The two label sets are different constructs (confirmed)

- `DecisionMaking.data_y`: 4-class, `{sponL, sponT, instT, instL}`, counts
  2517 / 3047 / 2782 / 2783. This is the file mapped to Experiment 1's target.
- `Feedback.data_y`: observer-correctness (`{Correct, Incorrect}`, 5546/5583) — the
  *outcome*, not the deception condition. **Not used as a target anywhere in this
  experiment.** `Feedback.data_X` is not used at all (out of scope per the plan; not
  trivial enough to justify the added complexity for a one-line sanity check).

## 3. Player/observer axis of `data_X`'s trailing dimension — verified by correlation

`participant_pairs` rows are `(Player_subNN, Observer_subMM)`. The trailing axis of
`data_X` (size 2) was verified, not assumed, by correlating `data_X[idx, :, :, 0]` and
`data_X[idx, :, :, 1]` against the corresponding subject's full epoch loaded directly
from `data/raw/Preprocessed/DecisionMaking/Player_sub01_Observer_sub02.mat` (the
`player` and `observer` structs `src/preprocessing.py` already documents), for all 484
trials of `pair_num == 1`.

```
slice 0 vs player.x    mean corr = 0.9359
slice 0 vs observer.x  mean corr = 0.0111
slice 1 vs player.x    mean corr = 0.0111
slice 1 vs observer.x  mean corr = 0.9411
```

**Conclusion: axis 0 = Player = deceiver, axis 1 = Observer = observer.** This matches
Step 3c's primary role choice (deceiver rows), so the CNN uses `data_X[..., 0]`.

## 4. THE central finding: the archive's own `className` labels use a swapped L/T
   convention for *instructed* trials, already known to and corrected by `src/preprocessing.py`

`src/preprocessing.py`'s `DM_CLASS_MAP` (already in the codebase, verified there against
the unambiguous spontaneous trials) maps:

```
instL -> ("T", "truth")   # counterintuitive: "instL" is NOT lie in the table's convention
instT -> ("L", "lie")     # counterintuitive: "instT" is NOT truth
sponL -> ("S", "lie")
sponT -> ("S", "truth")
```

Applying the **naive** literal reading of the archive's own class names (`{sponL, instL}
-> lie`, `{sponT, instT} -> truth` — what a reader unfamiliar with this project's own
prior finding would do, and what the plan's Step 2b table described as the default
interpretation) gives Deception=5,300 / Truth=5,829 for all 23 sessions. This is what
the plan flagged as needing reconciliation against the trial table's 5,338 / 5,794 (a
~38-trial gap in each direction).

Applying the **corrected** `DM_CLASS_MAP` swap instead barely changes the aggregate
count (5,299 / 5,830 for all 23 sessions — the swap is nearly symmetric in aggregate,
`instL` count 2,783 approx `instT` count 2,782), so the aggregate imbalance alone could
not have revealed the bug. The swap only shows up per-trial.

**Direct verification:** for `pair_num == 1` (session `Player_sub01_Observer_sub02`,
484 trials), the trial table's `condition` column (sorted by `round, trial` — the same
chronological order `src/timeline.py` and `src/preprocessing.py`'s alignment both use)
was compared position-by-position against the archive's own class-label sequence:

- Using the naive mapping: 242/484 positions agree (50.0% — chance level for a
  ~55/45 split; NOT a real alignment, this is what first raised the flag).
- Using the corrected `DM_CLASS_MAP` swap: **0/484 positions disagree (100.0% exact
  match)**.

This confirms two things simultaneously: (a) the archive's own class labels for
instructed trials must be read through the same `DM_CLASS_MAP` swap `src/preprocessing.py`
already established, not read literally; and (b) once that correction is applied,
`data_X`'s epoch order within a `pair_num` bucket is **exactly** the trial table's
chronological `(round, trial)` order — a direct positional join, no reordering needed,
for sessions where the archive's own trial count also matches the table's.

## 5. Reconciling the ~38-trial discrepancy and the sub19/sub22 asymmetry

`Player_sub22_Observer_sub19` (`pair_num == 21`) is the one session
`src/preprocessing.py`'s own alignment step (`compute_alignment`, `difflib.SequenceMatcher`
ratio against the trial table) found **unrecoverable** (ratio 0.353, vs >=0.999 for every
other session) and excluded entirely from `single_brain.parquet`/`dyadic.parquet`. The
*reverse*-role session, `Player_sub19_Observer_sub22` (`pair_num == 18`), aligned fine
(ratio 1.0) and **is** present in the feature tables (484 deceiver rows under
`pair_id == "sub19_sub22"`) — this is the asymmetry the plan flagged as needing a check:
the feature tables exclude one session of the `sub19_sub22` dyad, not the whole dyad, and
the `.mat` files include both of the dyad's sessions since the OneDCNN archive's own
epoch-vs-label pairing has nothing to do with this project's trial-table alignment step.

Recomputing the label balance with **both** corrections applied — the `DM_CLASS_MAP`
swap, and excluding only `pair_num == 21` (not `pair_num == 18`) — against
`single_brain.parquet`'s deceiver-row totals:

```
data_X (pair_num != 21, corrected labels): 10,645 trials — truth 5,586 / lie 5,059
single_brain.parquet deceiver rows:        10,648 trials — truth 5,588 / lie 5,060
difference:                                    -3         —      -2   /    -1
```

The remaining -3 is exactly the 3 globally-missing `data_X` rows identified in section 1
(pair_num 20 short by 1, pair_num 22 short by 2) — trials the OneDCNN archive's own
(unknown, undocumented) artifact-rejection pipeline dropped, independent of this
project's alignment step. **The ~38-trial discrepancy the plan flagged is fully
explained** by excluding `pair_num == 21` (the whole unrecoverable session, 484 trials,
truth-heavy under either mapping) from the all-23-sessions total. The label-mapping
choice (naive vs. corrected `DM_CLASS_MAP`) changes which *specific* trials are lie vs.
truth but barely moves the aggregate count (section 4 above), so it is not itself a
source of the aggregate 38-trial gap — it matters for per-trial correctness (the CNN's
labels must use the corrected mapping to be right trial-by-trial), while the aggregate
count gap is explained almost entirely by the `pair_num == 21` exclusion, with the
final 3-trial residual explained by section 1's archive-side dropped trials.

The specific 3 trials themselves (which `round`/`trial` in buckets 20 and 22) were not
individually hunted down — the archive carries no rejection reason or index field (see
`src/preprocessing.py`'s FILE FORMAT section, same limitation), and the count is small
enough (3 of 10,648, 0.03%) that per-trial identification would not change any decision.
They are dropped from the CNN's training set with a printed count, not reassigned or
imputed, matching Step 2c's directive.

## 5b. A second, independent labeling bug: `DecisionMaking.mat`'s own `className`
    array is internally inconsistent with the per-session archives it was built from

Discovered while implementing `src/cnn_check.py`'s trial-identity join (Step 6c),
after section 4/5 above were already written and section 4's check had used the raw
per-session archive's `className` directly (bypassing this bug by accident).

`OneDCNN/DecisionMaking.mat`'s own `className` cell array is
`['sponL','sponT','instT','instL']`. Every one of the 22 recoverable per-session raw
archive files under `data/raw/Preprocessed/DecisionMaking/*.mat` — checked for all
22, not just one — consistently uses `['sponL','sponT','instL','instT']` (positions 3
and 4 swapped relative to the OneDCNN file) for the identical `data_y` integer codes.
`data_y`'s numeric index agrees with the raw archive's own per-epoch
`argmax(y)` index at 100% (verified for `pair_num` 1 and 5), so the same integer
means the same physical trial in both files — only the **name attached to that
integer** differs between OneDCNN's own metadata and the per-session archives.

Practically: reading `DecisionMaking.mat`'s `data_y` through its own `className`
array assigns `instT`/`instL` to the wrong trials for every instructed trial (spurious
trials, not spontaneous ones, are unaffected since `sponL`/`sponT` are in the same
positions in both orderings). This is **independent of, and additional to**, the
`DM_CLASS_MAP` L/T swap in section 4/5 above — two separate corrections are both
required: (1) reinterpret `data_y`'s index using `['sponL','sponT','instL','instT']`
(the raw archive's own consistent ordering, not `DecisionMaking.mat`'s own metadata),
then (2) apply `DM_CLASS_MAP`'s further `instL`->truth / `instT`->lie correction.
Verified on `pair_num` 1 and 5: both corrections together give 0/484 exact
mismatches against `trial_table.csv`'s chronological condition sequence.

`src/cnn_check.py` hardcodes the raw archive's own class-name ordering
(`_RAW_ARCHIVE_CLASS_NAMES`) rather than trusting `DecisionMaking.mat`'s
`className` variable, with this finding documented at the point of use.

## 6. Exact trial-identity join is recoverable (not falling back to dyad-level)

Because (a) `data_X`'s per-session epoch order matches the trial table's chronological
`(round, trial)` order exactly (verified in section 4, 0 mismatches over 484 trials) and
(b) `src/preprocessing.py`'s `compute_alignment` already computes and stores a row-level
`table-row -> epoch-index` map for the two sessions with a non-1.0 alignment ratio
(`Player_sub21_Observer_sub20` and `Player_sub23_Observer_sub24` — note these are
*exactly* `pair_num` 20 and 22, the two buckets found short in section 1: the archive's
own missing trials are the same trials `compute_alignment` already flagged as
unrecoverable-per-trial against the table), the **exact-trial fold-sharing plan** in
Step 6c is used, not the dyad-level fallback:

- `pair_num == 21` (`Player_sub22_Observer_sub19`): excluded entirely — matches
  `single_brain.parquet`'s own exclusion of this session.
- All other 22 `pair_num` buckets: joined to `trial_table.csv` deceiver rows
  (`pair_id`, `session_id` derived from `participant_pairs`, sorted by `(round, trial)`)
  by direct position, EXCEPT buckets 20 and 22 which drop the 1 and 2 archive-side-only
  trials respectively (identified by re-running `src/preprocessing.py`'s own
  `compute_alignment` machinery against the corrected-label epoch-key sequence, imported
  directly rather than reimplemented, per this project's own convention of reusing
  established alignment code).
- Each joined trial inherits the fold its `single_brain.parquet` deceiver row received
  in Step 4's `StratifiedKFold` split. Trials with no match (the 3 archive-side-only
  trials, plus `pair_num == 21`) are dropped from the CNN's set with a printed count.

This gives **exact trial-level pairing**, not the dyad-level fallback — `src/cnn_check.py`
asserts the overlap fraction (Step 9 validation 8) and would fall back only if that
assertion failed at runtime, which section 4/5's checks above give no reason to expect.
