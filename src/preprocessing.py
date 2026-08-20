"""
src/preprocessing.py -- Section 9 EEG preprocessing / windowing.

Slices named sub-windows out of the already-epoched, artifact-free EEG data
shipped in the figshare "Preprocessed.zip" archive, aligns each epoch to its
row in `data/processed/trial_table.csv`, and writes:

  - data/processed/epochs/<stimulus>/<session_id>.npz   (one per session per
    stimulus; float32 arrays, one per named window x role)
  - data/processed/epoch_index.csv                       (join table back to
    trial_table.csv; includes rows for excluded trials with a reason)
  - data/processed/eeg_preprocessing_notes.md             (provenance doc)

--------------------------------------------------------------------------
WHY THIS MODULE DOES NOT COMPUTE A WINDOW FROM A CONTINUOUS RECORDING
--------------------------------------------------------------------------
`data/raw/readme.txt` describes `Preprocessed/` as already containing
"preprocessed, artifact-free EEG epochs ... aligned with the decision-making
or feedback onset." Opening one file (step 3 of plans/eeg-preprocessing.md)
confirmed this: each `.mat` holds a `player` struct and an `observer` struct,
each with fields `fs, x, clab, y, className, t, filename`. `x` has shape
(n_samples, n_channels, n_epochs) -- already epoched, not continuous. So
"windowing" in this module means *selecting a sub-window inside an existing
epoch*, never cutting from a continuous trace. No filtering, re-referencing,
baseline correction, or artifact rejection happens here -- the archive
already did that, undocumented as to its exact parameters (see the
provenance doc for exactly what is and is not documented).

--------------------------------------------------------------------------
FILE FORMAT (found by inspection, not assumed)
--------------------------------------------------------------------------
`scipy.io.loadmat` reads every file without raising `NotImplementedError`
(these are MATLAB v7 files, not v7.3/HDF5). Two top-level (1,1) object
structs per file: `player` (the deceiver -- confirmed by
`player.filename == '[...]_player.set'`) and `observer` (confirmed by
`observer.filename == '[...]_observer.set'`) -- role is identified by name,
never by array position, per the plan's requirement.

  fs         : (1,1) uint8, sampling rate in Hz. 100 Hz in every file
               checked (all 23 DecisionMaking + all 23 Feedback sessions).
  x          : (n_samples, n_channels, n_epochs) float64. Axis assignment
               justified by the numbers, not convention: n_channels == 30
               (matches the workflow spec's 30-electrode montage) in every
               file; n_epochs is close to the session's trial count (up to
               artifact rejection, see ALIGNMENT below); n_samples is
               sampling-rate x epoch-duration and matches the `t` field.
  clab       : (1, n_channels) cell array of channel-name strings.
               **Channel order is NOT identical across all 23 sessions** --
               two distinct orderings were found (they differ only in
               whether PO4 precedes or follows O1). This module therefore
               carries each session's own channel-name list through to the
               output rather than assuming one global order.
  y          : (n_classes, n_epochs) uint8, one-hot class label per epoch.
  className  : (1, n_classes) cell array of class-name strings.
               DecisionMaking: ['sponL','sponT','instL','instT'].
               Feedback: ['Correct','Incorrect'].
  t          : (1, n_samples) int16, the epoch's own time axis in ms,
               t=0 is the onset the epoch is aligned to (decision-making
               onset or feedback onset, per the readme).
  filename   : the archive's own EEGLAB `.set` filename, used only to
               confirm player/observer identity (see above).

No trial-index, event-latency, or rejection-mask field exists in the
struct -- the field list above is exhaustive. This is why alignment (below)
has to be recovered from the class-label sequence rather than read off a
provided index.

--------------------------------------------------------------------------
SAMPLING RATE, EPOCH EXTENT, UNITS
--------------------------------------------------------------------------
DecisionMaking : fs=100 Hz, t = -500 .. +2990 ms (350 samples). t=0 is
                 decision-making onset.
Feedback       : fs=100 Hz, t = -200 .. +990 ms (120 samples). t=0 is
                 feedback onset.
Units          : not stated in any field. Observed value range on a sample
                 file was roughly -106 to +85 (mean ~0), which is consistent
                 with microvolts (typical scalp EEG is tens of uV) and
                 inconsistent with volts (~1e-5 scale) or raw ADC counts
                 (typically thousands). Recorded as "likely microvolts,
                 not stated in the archive" -- not rescaled here.

--------------------------------------------------------------------------
PRIOR PREPROCESSING -- WHAT THE ARCHIVE DOCUMENTS AND WHAT IT DOES NOT
--------------------------------------------------------------------------
Documented (readme.txt): the epochs are "artifact-free" and "aligned with
the decision-making or feedback onset." That is the entire documentation.
Not documented anywhere in the archive (no history/comments/etc field in
the struct, no extra readme or text file inside Preprocessed.zip itself):
filter band (high-pass/low-pass/notch), re-referencing scheme,
downsampling method, ICA or other artifact-correction procedure, the exact
artifact-rejection criterion, baseline-correction window. These are
recorded as "not documented in the archive" in
`data/processed/eeg_preprocessing_notes.md` -- not inferred from the signal
and presented as fact, per the plan's explicit instruction.

--------------------------------------------------------------------------
ALIGNMENT -- HOW EPOCHS WERE MAPPED TO TRIAL_TABLE.CSV ROWS
--------------------------------------------------------------------------
No index field exists (see FILE FORMAT above), so alignment is recovered
from an independent quantity: the epoch's own class label (`y`/`className`)
compared against the trial table's own recorded condition for that trial.

DecisionMaking key per trial: (card_type_raw, condition) -- a 4-valued key
(('L','lie'), ('T','truth'), ('S','lie'), ('S','truth')). The archive's
`className` codes were found to use 'L'/'T' with the *opposite* polarity
from `card_type_raw` (archive 'instL' co-occurs, at every matching
position, with the table's ('T','truth'); this is a pure labeling
convention difference, not a misalignment -- confirmed because the 'S'
(spontaneous) trials, which are not subject to any label-swap ambiguity,
matched the table 1:1 at every position in every session that has equal
epoch and trial counts). `_DM_KEY_MAP` below encodes this correction.

Feedback key per trial: (outcome,) -- a 2-valued key ('correct'/'incorrect'
from the table's `outcome` column vs the archive's 'Correct'/'Incorrect'
className).

For each session and stimulus:
  1. Build the table's per-trial key sequence, sorted by (round, trial) --
     this is the same chronological order `src/timeline.py` sorts on.
  2. Build the epoch's per-epoch key sequence from `y`/`className`, in
     on-disk epoch order (dimension 3 of `x`).
  3. Run `difflib.SequenceMatcher` between the two key sequences and take
     its `ratio()` as a global fit score.
  4. If `ratio() >= ALIGNMENT_RATIO_THRESHOLD` (0.98): accept. Walk the
     matcher's opcodes -- 'equal' blocks give a direct table-row ->
     epoch-index mapping; 'delete' blocks (a table trial with no matching
     epoch) are recorded as excluded with reason "rejected_in_archive"
     (the archive's own artifact rejection dropped that trial); any
     'insert' or 'replace' block (an epoch with no confident table match)
     is recorded as excluded with reason "ambiguous_local_alignment" and
     is not silently mapped.
  5. If `ratio() < ALIGNMENT_RATIO_THRESHOLD`: the whole session is
     unrecoverable for that stimulus. Every trial in that
     (session, stimulus) is excluded with reason "alignment_not_recoverable"
     -- no positional guess is made. This fired for exactly one file:
     `Player_sub22_Observer_sub19` (ratio 0.353 on DecisionMaking, 146
     opcodes -- compare to every other session's ratio >= 0.999). A shift
     scan (constant offsets -10..+10) topped out at 268/484 (55%),
     confirming this is not a simple off-by-N alignment problem; no
     latency or reaction-time field exists in the struct to attempt the
     RT-correlation fallback the plan describes, so this session is
     excluded rather than guessed.

Verified empirically, independently for each (session, stimulus) pair (the
code below does not share exclusions across stimuli -- each is its own
`compute_alignment` call): 20 of 23 sessions have equal epoch and trial
counts with ratio 1.0 (0 exclusions), in both DecisionMaking and Feedback.
Two sessions (`Player_sub21_Observer_sub20`, `Player_sub23_Observer_sub24`)
have 1 and 2 fewer epochs than trials respectively, resolved to the same
specific (round, trial) pairs independently in both stimuli at ratio
>= 0.998 -- i.e. Feedback's weaker 2-valued key (`outcome` only) still
converged on the identical excluded trials as DecisionMaking's stronger
4-valued key, which cross-validates both. One session
(`Player_sub22_Observer_sub19`) is wholly excluded in both stimuli per the
ratio-threshold rule above (ratio 0.353 on DecisionMaking, 0.300 on
Feedback -- both far below every other session's >=0.998).

--------------------------------------------------------------------------
WINDOWS CHOSEN
--------------------------------------------------------------------------
Trial spacing (from trial_table.csv alone, `_trial_spacing_report`):
median inter-trial interval 9.37 s, 5th percentile 8.57 s, minimum 7.77 s
(within-session; between-round gaps up to 274 s are excluded from this
figure as expected block breaks). Every window below is far shorter than
the minimum 7.77 s gap, so no window can overlap the *previous* trial --
trial spacing does not constrain window choice here; epoch extent does.

**Section 9's illustrative -2000 ms window and Section 18's six 250 ms
sliding windows from -1500 to 0 ms are NOT constructible from
`Preprocessed.zip`.** DecisionMaking epochs only extend to -500 ms
pre-onset (Feedback only to -200 ms) -- nowhere near -1500 or -2000 ms.
Building either would require re-deriving epochs from `Raw.zip`, which is
explicitly out of this task's download scope. This is reported as a
finding for the user's decision, not solved here.

DecisionMaking windows (t=0 = decision-making onset; extent -500..+2990ms;
sample counts below are inclusive of both endpoints, at 10ms/sample):
  pre    -500 .. 0 ms     (51 samples) -- the entire available pre-decision
                                            anticipation period; this is the
                                            closest constructible analog to
                                            S9's pre-decision window, capped
                                            by the archive at -500ms instead
                                            of -2000ms.
  onset  -200 .. +300 ms  (51 samples) -- spans the decision onset itself,
                                            for ERP-style features.
  post   +300 .. +1300 ms (101 samples)-- post-decision processing,
                                            comfortably inside the +2990ms
                                            ceiling with room to spare (not
                                            used, to keep the window set to
                                            three, not padding it further
                                            for no stated purpose).

Feedback windows (t=0 = feedback onset; extent -200..+990ms):
  pre    -200 .. 0 ms     (21 samples) -- entire available pre-feedback
                                            baseline.
  onset  0 .. +300 ms     (31 samples) -- early feedback-locked response
                                            (e.g. FRN/P300 region).
  post   +300 .. +990 ms  (70 samples) -- late feedback processing.

Three windows were kept for both stimuli (not fewer) because the three
sub-windows are not near-identical relative to the extent (DecisionMaking:
51/51/101 of 350 samples; Feedback: 21/31/70 of 120 samples) -- collapsing
would lose distinguishable signal, per the plan's "use fewer only if
near-identical" instruction.

--------------------------------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------------------------------
Volume estimate (printed again at run time from the real numbers):
11,132 trials x 2 roles x 30 channels x (200 DM-samples + 119 FB-samples)
x 4 bytes (float32) ~= 850 MB. Flat CSV is not viable at this volume (see
`_print_volume_estimate`). One compressed `.npz` per session per stimulus
under `data/processed/epochs/<stimulus>/<session_id>.npz` was chosen over a
single `.h5`: epoch counts are close to uniform across sessions (482-484,
one session dropped entirely) so there is no non-uniformity that would
favour `.h5`'s random access, per-session npz needs no extra dependency,
and it lets Section 10 stream one session at a time rather than loading the
whole corpus. Each npz holds, per window name: a `<window>_deceiver` and
`<window>_observer` float32 array of shape
(n_extracted_trials, n_channels, n_samples), plus `channel_names`,
`sampling_rate_hz`, `window_start_ms`/`window_end_ms` per window, and
`trial_round`/`trial_num` arrays giving the (round, trial) each row
corresponds to.

--------------------------------------------------------------------------
WHY trial_table.csv IS NOT MODIFIED
--------------------------------------------------------------------------
`eeg_window_start` / `eeg_window_end` are left null, and `trial_table.csv`
is not touched. Three reasons (all apply):
  1. `trial_table.csv` is a deterministic output of `src/timeline.py`. An
     in-place edit here would be silently destroyed the next time that
     script runs.
  2. There is more than one window per trial (three, per stimulus) --
     two scalar columns cannot express a multi-window scheme without being
     arbitrary about which window "counts".
  3. The join is clean anyway: `epoch_index.csv` shares trial_table's
     grain keys (pair_id, session_id, round, trial, participant_id), so a
     one-line merge recovers everything Section 10 needs. The index is the
     more expressive artifact.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = REPO_ROOT / "data" / "raw" / "Preprocessed"
TRIAL_TABLE_CSV = REPO_ROOT / "data" / "processed" / "trial_table.csv"
EPOCHS_DIR = REPO_ROOT / "data" / "processed" / "epochs"
EPOCH_INDEX_CSV = REPO_ROOT / "data" / "processed" / "epoch_index.csv"
NOTES_MD = REPO_ROOT / "data" / "processed" / "eeg_preprocessing_notes.md"

STIMULI = ["DecisionMaking", "Feedback"]

# Archive className -> (card_type_raw-compatible code, condition). The 'L'/'T'
# swap relative to card_type_raw is a labeling-convention difference,
# verified against the 'S' (spontaneous) trials which have no swap ambiguity
# -- see module docstring, ALIGNMENT section.
DM_CLASS_MAP = {
    "instL": ("T", "truth"),
    "instT": ("L", "lie"),
    "sponL": ("S", "lie"),
    "sponT": ("S", "truth"),
}
FB_CLASS_MAP = {
    "Correct": ("correct",),
    "Incorrect": ("incorrect",),
}

ALIGNMENT_RATIO_THRESHOLD = 0.98

# Windows, in ms relative to each stimulus's own t=0. See module docstring
# "WINDOWS CHOSEN" for the justification grounded in the measured epoch
# extents (DecisionMaking -500..+2990ms, Feedback -200..+990ms).
WINDOWS = {
    "DecisionMaking": [
        ("pre", -500, 0),
        ("onset", -200, 300),
        ("post", 300, 1300),
    ],
    "Feedback": [
        ("pre", -200, 0),
        ("onset", 0, 300),
        ("post", 300, 990),
    ],
}


class SessionMat:
    """Thin wrapper around one loaded .mat file's player/observer structs."""

    def __init__(self, path: Path):
        m = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
        self.player = m["player"][0, 0]
        self.observer = m["observer"][0, 0]
        # Role identified by the archive's own filename field, not by
        # dict/array position (plan requirement, 3c/5b).
        assert "_player.set" in str(self.player.filename[0]), (
            f"player.filename does not contain '_player.set': {self.player.filename}"
        )
        assert "_observer.set" in str(self.observer.filename[0]), (
            f"observer.filename does not contain '_observer.set': {self.observer.filename}"
        )

    def fs(self) -> int:
        fs_p = int(self.player.fs[0, 0])
        fs_o = int(self.observer.fs[0, 0])
        assert fs_p == fs_o, f"player/observer fs mismatch: {fs_p} vs {fs_o}"
        return fs_p

    def t_axis(self) -> np.ndarray:
        return self.player.t[0].astype(int)

    def channel_names(self, role: str) -> list[str]:
        struct = self.player if role == "deceiver" else self.observer
        return [c[0] for c in struct.clab[0]]

    def n_epochs(self) -> int:
        n_p = self.player.x.shape[2]
        n_o = self.observer.x.shape[2]
        assert n_p == n_o, f"player/observer epoch count mismatch: {n_p} vs {n_o}"
        return n_p

    def epoch_keys(self, stimulus: str) -> list[tuple]:
        cn = [c[0] for c in self.player.className[0]]
        y = self.player.y
        cls_map = DM_CLASS_MAP if stimulus == "DecisionMaking" else FB_CLASS_MAP
        labels = [cn[int(np.argmax(y[:, i]))] for i in range(y.shape[1])]
        return [cls_map[label] for label in labels]

    def x(self, role: str) -> np.ndarray:
        struct = self.player if role == "deceiver" else self.observer
        return struct.x  # (n_samples, n_channels, n_epochs), float64


def _table_key_row(row: pd.Series, stimulus: str) -> tuple:
    if stimulus == "DecisionMaking":
        return (row["card_type_raw"], row["condition"])
    return (row["outcome"],)


def compute_alignment(
    session_trials: pd.DataFrame, epoch_keys: list[tuple], stimulus: str
) -> tuple[Optional[dict], float, list[dict]]:
    """Aligns table rows (sorted by round, trial) to epoch indices.

    Returns (row_to_epoch or None, ratio, exclusion_records). `row_to_epoch`
    maps a 0-based position in `session_trials` (already sorted by
    (round, trial)) to a 0-based epoch index in the archive's x array; it is
    None if the whole session is unrecoverable for this stimulus.
    `exclusion_records` lists {row_position, reason} for any table row that
    did not get an epoch, even when the session as a whole is accepted.
    """
    table_keys = [_table_key_row(r, stimulus) for _, r in session_trials.iterrows()]
    sm = difflib.SequenceMatcher(None, table_keys, epoch_keys, autojunk=False)
    ratio = sm.ratio()
    if ratio < ALIGNMENT_RATIO_THRESHOLD:
        return None, ratio, [
            {"row_position": i, "reason": "alignment_not_recoverable"}
            for i in range(len(table_keys))
        ]

    row_to_epoch: dict[int, int] = {}
    exclusions: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                row_to_epoch[i1 + k] = j1 + k
        elif tag == "delete":
            for i in range(i1, i2):
                exclusions.append({"row_position": i, "reason": "rejected_in_archive"})
        else:  # 'insert' or 'replace': no confident table<->epoch match
            for i in range(i1, i2):
                exclusions.append({"row_position": i, "reason": "ambiguous_local_alignment"})
    return row_to_epoch, ratio, exclusions


def _slice_window(
    x: np.ndarray, t_axis: np.ndarray, epoch_indices: list[int], start_ms: int, end_ms: int
) -> np.ndarray:
    """x: (n_samples, n_channels, n_epochs) -> (n_epochs_selected, n_channels, n_window_samples)."""
    mask = (t_axis >= start_ms) & (t_axis <= end_ms)
    sub = x[mask, :, :]  # (n_window_samples, n_channels, n_epochs)
    sub = sub[:, :, epoch_indices]  # select the wanted epochs
    return np.transpose(sub, (2, 1, 0)).astype(np.float32)  # (n_epochs, n_channels, n_samples)


def _trial_spacing_report(df: pd.DataFrame) -> tuple[float, float, float]:
    diffs = []
    for _, grp in df.groupby(["pair_id", "session_id", "role"]):
        grp = grp.sort_values(["round", "trial"])
        diffs.append(grp["trigger_ts_sec"].diff().dropna())
    all_diffs = pd.concat(diffs)
    return float(all_diffs.median()), float(all_diffs.quantile(0.05)), float(all_diffs.min())


def _print_volume_estimate(n_trials: int) -> None:
    n_channels = 30
    n_samples_dm = sum(
        int(round((end - start) / 10)) + 1 for _, start, end in WINDOWS["DecisionMaking"]
    )
    n_samples_fb = sum(
        int(round((end - start) / 10)) + 1 for _, start, end in WINDOWS["Feedback"]
    )
    total_bytes = n_trials * 2 * n_channels * (n_samples_dm + n_samples_fb) * 4
    print(
        f"Volume estimate: {n_trials} trials x 2 roles x {n_channels} channels x "
        f"({n_samples_dm} DM-samples + {n_samples_fb} FB-samples) x 4 bytes "
        f"= {total_bytes / 1e6:.1f} MB"
    )


def process_session(
    session_id: str, stimulus: str, trial_table: pd.DataFrame
) -> tuple[Optional[dict], list[dict]]:
    """Returns (npz_payload_dict or None, epoch_index_rows)."""
    mat_path = PREPROCESSED_DIR / stimulus / f"{session_id}.mat"
    if not mat_path.exists():
        # No archive file for this session at all.
        sess = trial_table[trial_table["session_id"] == session_id]
        rows = []
        for _, r in sess.iterrows():
            rows.append(
                _index_row(r, stimulus, None, None, None, None, "no_archive_file")
            )
        return None, rows

    smat = SessionMat(mat_path)
    fs = smat.fs()
    t_axis = smat.t_axis()

    deceiver_rows = (
        trial_table[
            (trial_table["session_id"] == session_id) & (trial_table["role"] == "deceiver")
        ]
        .sort_values(["round", "trial"])
        .reset_index(drop=True)
    )
    epoch_keys = smat.epoch_keys(stimulus)
    row_to_epoch, ratio, exclusions = compute_alignment(deceiver_rows, epoch_keys, stimulus)

    print(
        f"  [{stimulus}] {session_id}: n_trials={len(deceiver_rows)} "
        f"n_epochs={len(epoch_keys)} alignment_ratio={ratio:.4f} "
        f"exclusions={len(exclusions)}"
    )

    index_rows: list[dict] = []
    if row_to_epoch is None:
        for pos in range(len(deceiver_rows)):
            r = deceiver_rows.iloc[pos]
            partner_r = trial_table[
                (trial_table["session_id"] == session_id)
                & (trial_table["role"] == "observer")
                & (trial_table["round"] == r["round"])
                & (trial_table["trial"] == r["trial"])
            ]
            for window_name, start_ms, end_ms in WINDOWS[stimulus]:
                index_rows.append(
                    _index_row(
                        r, stimulus, window_name, start_ms, end_ms, None,
                        "alignment_not_recoverable",
                    )
                )
                for _, pr in partner_r.iterrows():
                    index_rows.append(
                        _index_row(
                            pr, stimulus, window_name, start_ms, end_ms, None,
                            "alignment_not_recoverable",
                        )
                    )
        return None, index_rows

    excluded_positions = {e["row_position"]: e["reason"] for e in exclusions}
    extracted_positions = [
        pos for pos in range(len(deceiver_rows)) if pos not in excluded_positions
    ]
    epoch_indices = [row_to_epoch[pos] for pos in extracted_positions]

    channel_names_deceiver = smat.channel_names("deceiver")
    channel_names_observer = smat.channel_names("observer")

    payload: dict = {
        "sampling_rate_hz": np.array([fs]),
        "channel_names_deceiver": np.array(channel_names_deceiver),
        "channel_names_observer": np.array(channel_names_observer),
        "trial_round": deceiver_rows.iloc[extracted_positions]["round"].to_numpy(),
        "trial_num": deceiver_rows.iloc[extracted_positions]["trial"].to_numpy(),
    }

    x_deceiver = smat.x("deceiver")
    x_observer = smat.x("observer")

    for window_name, start_ms, end_ms in WINDOWS[stimulus]:
        arr_deceiver = _slice_window(x_deceiver, t_axis, epoch_indices, start_ms, end_ms)
        arr_observer = _slice_window(x_observer, t_axis, epoch_indices, start_ms, end_ms)
        payload[f"{window_name}_deceiver"] = arr_deceiver
        payload[f"{window_name}_observer"] = arr_observer
        payload[f"{window_name}_start_ms"] = np.array([start_ms])
        payload[f"{window_name}_end_ms"] = np.array([end_ms])
        payload[f"{window_name}_n_samples"] = np.array([arr_deceiver.shape[2]])

    rel_path = f"{stimulus}/{session_id}.npz"
    array_row_for_pos = {pos: i for i, pos in enumerate(extracted_positions)}

    for pos in range(len(deceiver_rows)):
        r = deceiver_rows.iloc[pos]
        reason = excluded_positions.get(pos)
        arr_idx = array_row_for_pos.get(pos)
        for window_name, start_ms, end_ms in WINDOWS[stimulus]:
            n_samples = payload[f"{window_name}_n_samples"][0] if arr_idx is not None else None
            index_rows.append(
                _index_row(
                    r,
                    stimulus,
                    window_name,
                    start_ms,
                    end_ms,
                    n_samples,
                    reason,
                    epoch_file=rel_path if reason is None else None,
                    epoch_array_key=f"{window_name}_deceiver" if reason is None else None,
                    epoch_index_in_array=arr_idx,
                    fs=fs,
                    source_epoch_id=epoch_indices[arr_idx] if arr_idx is not None else None,
                )
            )
        partner_r = trial_table[
            (trial_table["session_id"] == session_id)
            & (trial_table["role"] == "observer")
            & (trial_table["round"] == r["round"])
            & (trial_table["trial"] == r["trial"])
        ]
        for _, pr in partner_r.iterrows():
            for window_name, start_ms, end_ms in WINDOWS[stimulus]:
                n_samples = payload[f"{window_name}_n_samples"][0] if arr_idx is not None else None
                index_rows.append(
                    _index_row(
                        pr,
                        stimulus,
                        window_name,
                        start_ms,
                        end_ms,
                        n_samples,
                        reason,
                        epoch_file=rel_path if reason is None else None,
                        epoch_array_key=f"{window_name}_observer" if reason is None else None,
                        epoch_index_in_array=arr_idx,
                        fs=fs,
                        source_epoch_id=epoch_indices[arr_idx] if arr_idx is not None else None,
                    )
                )

    return payload, index_rows


def _index_row(
    r: pd.Series,
    stimulus: str,
    window_name,
    start_ms,
    end_ms,
    n_samples,
    excluded_reason,
    epoch_file=None,
    epoch_array_key=None,
    epoch_index_in_array=None,
    fs=None,
    source_epoch_id=None,
) -> dict:
    return {
        "pair_id": r["pair_id"],
        "session_id": r["session_id"],
        "dyad_trial_seq": r["dyad_trial_seq"],
        "round": r["round"],
        "trial": r["trial"],
        "participant_id": r["participant_id"],
        "role": r["role"],
        "stimulus": stimulus,
        "window_name": window_name,
        "window_start_ms": start_ms,
        "window_end_ms": end_ms,
        "n_samples": n_samples,
        "sampling_rate_hz": fs,
        "epoch_file": epoch_file,
        "epoch_array_key": epoch_array_key,
        "epoch_index_in_array": epoch_index_in_array,
        "source_epoch_id": source_epoch_id,
        "excluded_reason": excluded_reason,
    }


def run() -> dict:
    trial_table = pd.read_csv(TRIAL_TABLE_CSV)
    median_gap, p5_gap, min_gap = _trial_spacing_report(trial_table)
    print(
        f"Trial spacing (within-session, trigger_ts_sec diffs): "
        f"median={median_gap:.3f}s p5={p5_gap:.3f}s min={min_gap:.3f}s"
    )

    n_trials = trial_table.drop_duplicates(
        subset=["pair_id", "session_id", "round", "trial"]
    ).shape[0]
    _print_volume_estimate(n_trials)

    session_ids = sorted(trial_table["session_id"].unique())
    all_index_rows: list[dict] = []

    for stimulus in STIMULI:
        EPOCHS_DIR.joinpath(stimulus).mkdir(parents=True, exist_ok=True)
        print(f"=== {stimulus} ===")
        for session_id in session_ids:
            payload, index_rows = process_session(session_id, stimulus, trial_table)
            all_index_rows.extend(index_rows)
            if payload is not None:
                out_path = EPOCHS_DIR / stimulus / f"{session_id}.npz"
                np.savez_compressed(out_path, **payload)

    index_df = pd.DataFrame(all_index_rows)
    EPOCH_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(EPOCH_INDEX_CSV, index=False)
    print(f"Wrote {len(index_df)} rows to {EPOCH_INDEX_CSV}")

    return {
        "n_trials": n_trials,
        "index_rows": len(index_df),
        "median_gap": median_gap,
        "p5_gap": p5_gap,
        "min_gap": min_gap,
    }


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
