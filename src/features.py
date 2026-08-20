"""
src/features.py -- Section 10 feature engineering.

Builds the named, interpretable feature tables that Experiments 1-8
(sections 11-18), the interpretability work (section 25), and the
inter-brain network analysis (section 26) consume. No model is fit here,
no fold is defined, no metric is computed, and nothing under `results/`
is read or written.

--------------------------------------------------------------------------
GRAIN AND SCHEMA
--------------------------------------------------------------------------
Single-brain table: one row per (session_id, dyad_trial_seq, participant_id)
-- one row per participant per trial, matching trial_table.csv's long form.
Experiment 1 classifies participant-trials; Experiment 6 (observer-only)
selects rows by role == 'observer'; Experiment 3 splits chronologically
within participant -- all of these are row filters on this grain.

Both stimuli (DecisionMaking, Feedback) become distinct columns
(`..._dm_...` / `..._fb_...`) on the SAME participant-trial row, not a
doubled row count with a stimulus key column: the experiments never treat
a trial's two stimulus epochs as separate observations. If one stimulus's
epoch could not be extracted for a trial (see EXCLUSION SET DIVERGENCE
below) that stimulus's columns are null on that row, with the reason
recorded in `dm_excluded_reason` / `fb_excluded_reason` -- the row itself
is not dropped unless BOTH stimuli are unavailable.

Partner features are not materialized as columns. Experiment 7's "both
participants' EEG" condition is served by a self-join on
(session_id, dyad_trial_seq) swapping participant_id. Doubling ~2,400
columns to avoid a one-line join is not worth it.

Dyadic table: one row per (session_id, dyad_trial_seq) -- a second table,
not extra columns hung off one arbitrary participant's row. A dyadic
feature belongs to the trial, not to either participant.

Direction convention, stated once and applied everywhere: DECEIVER -> OBSERVER.
Signal A is always the deceiver, signal B always the observer. Positive lag
(where used) means the observer's signal lags the deceiver's. `deceiver_id`
and `observer_id` are carried as explicit columns.

--------------------------------------------------------------------------
EXCLUSION SET DIVERGENCE BETWEEN STIMULI (verified, not assumed)
--------------------------------------------------------------------------
Reading epoch_index.csv directly shows the two stimuli's exclusion sets
are NOT identical. `alignment_not_recoverable` (the whole sub22_sub19
session, 968 participant-trials) hits both stimuli identically -- verified
by set comparison, zero symmetric difference on that subset. But
`rejected_in_archive` differs: DecisionMaking excludes trial=2 of round 8
in sub21_sub20 while Feedback excludes trial=1 of round 8 in the same
session (and similarly for two trials in sub23_sub24) -- 6 participant-
trials excluded in DM only, 6 different ones excluded in FB only, 12 total,
zero overlap. This is a real per-archive divergence (each stimulus's
artifact rejection was evidently run independently), not a bug in this
module or in preprocessing.py. Consequence: per-stimulus nullity (the
`dm_excluded_reason` / `fb_excluded_reason` design above) is required, not
optional -- a single joint-grain assumption would have been wrong.

--------------------------------------------------------------------------
FREQUENCY-DOMAIN METHOD: BANDPASS ON THE FULL EPOCH, THEN WINDOW -- NOT
NAIVE WELCH/FFT BAND BINNING
--------------------------------------------------------------------------
At fs=100 Hz the windows are 210-1010 ms wide. A periodogram computed
directly on a short window has a Rayleigh resolution of 1/T: ~4.8 Hz for
the 210 ms Feedback-pre window, ~1.0 Hz for the 1010 ms window. Binning
that into delta (0.5-4 Hz) would put delta below the first non-DC bin and
return a number shaped like a feature that is not one.

Instead: for each session, stimulus, and role, take the FULL epoch as
stored in `data/raw/Preprocessed/<stimulus>/<session>.mat` (350 samples
for DecisionMaking, 120 for Feedback -- these are the same archive files
`src/preprocessing.py` already reads, reused via direct import of
`SessionMat`, `compute_alignment`, `WINDOWS`, and `STIMULI` rather than
reimplementing alignment). Bandpass the full epoch with a zero-phase IIR
(`scipy.signal.butter(4, [lo, hi], btype='band', output='sos')` +
`scipy.signal.sosfiltfilt`, axis=samples) -- SOS form because a 0.5 Hz
corner at fs=100 Hz is numerically fragile in transfer-function (b, a)
form. Filtering the full ~3.5s/1.2s epoch and THEN slicing the window
avoids the edge-transient contamination that filtering a 21-sample window
directly would produce. Take the Hilbert analytic signal
(`scipy.signal.hilbert`, axis=samples) of the filtered full epoch, then
slice the window; band power = mean(|analytic|^2) over the window's
samples, per trial x channel. The same analytic signals are reused for
the dyadic phase measures (PLV, coherency) -- computed once per session,
used twice. Filter stability was verified empirically before this module
was written: `sosfreqz` on all 5 bands at order 4 showed finite impulse
responses and a mean passband gain of 0.96-0.96 for every band; order 4
was kept (no need to lower it).

`scipy.signal.welch` is used exactly once in this module, as a diagnostic
for the gamma decision below -- never as a feature.

--------------------------------------------------------------------------
RELIABILITY TIERING
--------------------------------------------------------------------------
For each (band, window) combination: n_cycles = window_duration_seconds x
band_low_edge_Hz. n_cycles >= 3 -> reliable; 1 <= n_cycles < 3 -> marginal;
n_cycles < 1 -> unreliable. Computed and published for every frequency and
dyadic feature in `feature_dictionary.csv` -- never dropped silently, never
computed silently. Measured window durations here: DecisionMaking pre/onset
= 0.50 s, post = 1.00 s; Feedback pre = 0.20 s, onset = 0.30 s,
post = 0.69 s.

Result (see `feature_engineering_notes.md` for the full matrix): **delta
(0.5 Hz low edge) is unreliable in every single window in this dataset** --
its best case is DecisionMaking-post at n_cycles=0.5, still under 1. Theta
is marginal almost everywhere, reliable only in DecisionMaking-post
(n_cycles=4.0). Alpha and beta are reliable in most windows, marginal only
in the two shortest Feedback windows. Gamma is reliable everywhere (its
high low-edge means even the 200 ms Feedback-pre window clears 3 cycles).
Unreliable-tier features are computed and tagged, never dropped -- any
downstream use must carry the tier label.

--------------------------------------------------------------------------
GAMMA DECISION
--------------------------------------------------------------------------
Before emitting gamma features, mean Welch PSD (1-50 Hz, full 350-sample
DecisionMaking epoch, 3 sessions x 30 channels x 484 trials, nperseg=256)
was inspected. Power in 30-45 Hz (mean 0.256) is the same order of
magnitude as 13-30 Hz beta (mean 0.344) and far above the values at
48-50 Hz (0.002-0.0002), which is where the antialiasing roll-off toward
Nyquist (fs/2 = 50 Hz) actually shows up. There is no sudden drop at
30 Hz consistent with a low-pass filter placed below gamma -- the decline
from beta into gamma is the ordinary 1/f-like EEG spectral shape, not a
noise floor. Decision: GAMMA IS KEPT, banded 30-45 Hz (Nyquist is 50 Hz,
and the top few Hz below Nyquist show real roll-off, so the upper edge is
set at 45 Hz rather than 50 to stay clear of that roll-off).

--------------------------------------------------------------------------
BASELINE CORRECTION
--------------------------------------------------------------------------
Whether the archive itself was baseline-corrected is unknown (documented
as unknown in `eeg_preprocessing_notes.md`). No NEW baseline correction is
applied here -- that would compound an unknown prior state with an
assumption. The `pre` window's own `td_mean_*` feature gives a model
explicit access to baseline-level amplitude instead.

--------------------------------------------------------------------------
TIME-DOMAIN FEATURES
--------------------------------------------------------------------------
Computed on the UNFILTERED windowed signal, per channel x stimulus x
window: mean, std, ptp (max-min), peakamp (signed amplitude at the sample
of max |amplitude|), peaklat (that sample's latency in ms, relative to the
stimulus onset t=0), slope (least-squares line fit, microvolt/s). All six
are kept -- none judged redundant even at 21 samples (the shortest window).

--------------------------------------------------------------------------
DYADIC / INTER-BRAIN FEATURES
--------------------------------------------------------------------------
Computed on the 30 HOMOLOGOUS channel pairs only (same electrode name in
both brains), using the analytic signals from the frequency-domain step:
  corr : Pearson correlation of A and B's RAW (unfiltered) window samples,
         per channel. Broadband (band='broad'). Range [-1, 1].
  plv  : abs(mean(exp(i*(phase_A - phase_B)))) over window samples, per
         band. Range [0, 1].
  coh  : amplitude-weighted coherency magnitude,
         abs(sum(A * conj(B))) / sqrt(sum(|A|^2) * sum(|B|^2)), over window
         samples, per band. Range [0, 1].

`scipy.signal.coherence` (Welch coherence) is deliberately NOT used --
magnitude-squared coherence on a single un-averaged segment is identically
1.0 by construction; Welch coherence needs multiple segments/tapers to
average over, and a 21-101 sample window cannot supply them. The
analytic-signal coherency magnitude above is the correct single-trial
substitute and is named `coh` here to mean exactly that -- not classical
Welch coherence.

MANDATORY CAVEAT (also present in `feature_engineering_notes.md`):
**Inter-brain synchrony here is a statistical relationship between two
simultaneously recorded signals; it is NOT evidence of communication
between brains.**

Small-sample bias: single-trial PLV and coherency are upward-biased at
small sample counts -- a 21-sample window will show high `plv`/`coh` values
for genuinely unrelated signals. The bias is roughly constant given equal
window length, so it does not by itself manufacture a truth-vs-deception
difference, but absolute `plv`/`coh` values must never be read as
"synchrony strength."

Skipped, with reason: cross-correlation-lag (`xcorrmax`/`xcorrlag`) and
mutual information were marked optional/lower-priority in the plan and are
skipped here in favor of shipping the required set (corr/plv/coh) fully
validated -- both would need meaningful extra validation work (a 21-sample
window makes a lag estimate close to meaningless, and MI needs a binning
scheme that itself needs tuning) that this task's time budget does not
extend to. Wavelet coherence is skipped outright: it needs several cycles
per scale within the window and these windows do not supply them for any
band where it would add anything over the Hilbert measures above.

For section 26's network analysis, a region-level all-pairs coupling
tensor is additionally written to `interbrain_networks.npz`. 30 electrodes
are grouped into 5 named regions (assignment below, by 10-20 topographic
proximity, not a data-driven clustering) and each region's complex
analytic signal is the mean of its member channels' analytic signals.
Region x region (deceiver-region x observer-region) PLV is computed per
band per window: 5 x 5 x 5 x 6 = 750 values per trial, stored as an array
(one 4D block per trial) since section 26 consumes a matrix, not a named
feature vector.

--------------------------------------------------------------------------
MISSING DATA
--------------------------------------------------------------------------
Never imputed. sub22_sub19 (whole session, both stimuli,
alignment_not_recoverable) produces NO feature rows at all -- its 968
participant-trials are absent from the single-brain table and its 484
trials absent from the dyadic table. The 12 participant-trials affected by
the DM/FB exclusion-set divergence (see above) keep their row but get null
`..._dm_...` or `..._fb_...` columns with a reason in
`dm_excluded_reason`/`fb_excluded_reason`. `prior_outcome`/`prior_condition`
stay null on each dyad's first trial (joined from trial_table.csv, a real
absence, not recomputed).

--------------------------------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------------------------------
Parquet (via pyarrow, added to requirements.txt and installed in this
task) -- chosen over `.npz` because it preserves named columns and dtypes
(required for interpretability, section 25/34) and compresses well; a flat
CSV at ~21k rows x ~2,400 float32 columns would be several hundred MB of
text and slow to reload, the same reasoning `src/preprocessing.py` used to
choose `.npz` over CSV for the epoch arrays.

--------------------------------------------------------------------------
CONVENTIONS MATCHED FROM src/preprocessing.py
--------------------------------------------------------------------------
Module docstring justifying every decision, plain functions (no classes
except the thin numeric SessionMat already defined there and reused here),
`if __name__ == "__main__"` entry point, fully deterministic, no network
access, session-by-session processing to bound memory (a session's full
analytic-signal set across 5 bands x 2 roles is roughly ~400 MB and is
released at the end of each session's iteration).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.signal as sig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocessing as pp  # SessionMat, compute_alignment, WINDOWS, STIMULI, PREPROCESSED_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
TRIAL_TABLE_CSV = REPO_ROOT / "data" / "processed" / "trial_table.csv"
EPOCH_INDEX_CSV = REPO_ROOT / "data" / "processed" / "epoch_index.csv"
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
NOTES_MD = REPO_ROOT / "data" / "processed" / "feature_engineering_notes.md"

STIM_CODE = {"DecisionMaking": "dm", "Feedback": "fb"}

# Band edges (Hz). Gamma kept per the PSD diagnostic -- see module docstring.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
FILTER_ORDER = 4

TD_STATS = ["mean", "std", "ptp", "peakamp", "peaklat", "slope"]

# Region assignment for the 30 known channels -- by 10-20 topographic
# proximity, not data-driven. Every one of the 30 channels appears exactly
# once; verified in build_channel_order().
REGIONS = {
    "frontal": ["Fp2", "F9", "F10", "F7", "F3", "Fz", "F4", "F8"],
    "central": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "temporal": ["T7", "T8", "CP5", "CP6"],
    "parietal": ["CP1", "CP2", "P7", "P3", "Pz", "P4", "P8"],
    "occipital": ["PO3", "PO4", "O1", "O2"],
}
REGION_NAMES = list(REGIONS.keys())


def reliability_tier(window_duration_s: float, band_low_hz: float) -> tuple[str, float]:
    n_cycles = window_duration_s * band_low_hz
    if n_cycles >= 3:
        tier = "reliable"
    elif n_cycles >= 1:
        tier = "marginal"
    else:
        tier = "unreliable"
    return tier, n_cycles


def build_sos_filters() -> dict:
    """Builds and stability-checks one SOS bandpass filter per band. Prints
    the check (finite impulse response + passband gain) so the run's stdout
    is evidence, not a claim."""
    fs = 100
    sos_filters = {}
    print("Filter stability check (order=%d, fs=%d):" % (FILTER_ORDER, fs))
    for name, (lo, hi) in BANDS.items():
        sos = sig.butter(FILTER_ORDER, [lo, hi], btype="band", fs=fs, output="sos")
        imp = np.zeros(2000)
        imp[0] = 1.0
        out = sig.sosfiltfilt(sos, imp)
        finite = bool(np.all(np.isfinite(out)))
        w, h = sig.sosfreqz(sos, worN=2048, fs=fs)
        passband_mask = (w >= lo) & (w <= hi)
        passband_gain = float(np.abs(h[passband_mask]).mean())
        print(f"  {name}: finite={finite} passband_mean_gain={passband_gain:.3f}")
        if not finite:
            raise RuntimeError(f"Unstable filter for band {name} at order {FILTER_ORDER}")
        sos_filters[name] = sos
    return sos_filters


def gamma_psd_diagnostic() -> None:
    """Welch PSD diagnostic (1-50 Hz) used ONLY to decide whether gamma is
    representable in this archive -- never used as a feature. Printed for
    the record; the decision itself is documented in the module docstring."""
    fs = 100
    sessions = [
        "Player_sub01_Observer_sub02",
        "Player_sub09_Observer_sub10",
        "Player_sub24_Observer_sub23",
    ]
    all_psd = []
    freqs = None
    for s in sessions:
        smat = pp.SessionMat(pp.PREPROCESSED_DIR / "DecisionMaking" / f"{s}.mat")
        x = smat.x("deceiver")  # (n_samples, n_channels, n_epochs)
        xr = x.transpose(2, 1, 0).reshape(-1, x.shape[0])
        f, pxx = sig.welch(xr, fs=fs, nperseg=min(256, xr.shape[1]), axis=-1)
        all_psd.append(pxx.mean(axis=0))
        freqs = f
    mean_psd = np.mean(all_psd, axis=0)
    print("Gamma PSD diagnostic (Welch, 1-50 Hz, 3-session sample):")
    for lo, hi in [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 50)]:
        mask = (freqs >= lo) & (freqs < hi)
        print(f"  {lo}-{hi} Hz: mean power {mean_psd[mask].mean():.4f}")


def build_channel_order() -> list[str]:
    """Canonical, alphabetically sorted channel order used for every column
    name and every array's channel axis in this module's output, regardless
    of which of the archive's two orderings a given session used. Verified
    against a session of each ordering and against the full REGIONS mapping."""
    smat_a = pp.SessionMat(pp.PREPROCESSED_DIR / "DecisionMaking" / "Player_sub01_Observer_sub02.mat")
    smat_b = pp.SessionMat(pp.PREPROCESSED_DIR / "DecisionMaking" / "Player_sub21_Observer_sub20.mat")
    names_a = set(smat_a.channel_names("deceiver"))
    names_b = set(smat_b.channel_names("deceiver"))
    assert names_a == names_b, "Channel name sets differ across sessions"
    region_channels = set()
    for chans in REGIONS.values():
        region_channels.update(chans)
    assert region_channels == names_a, (
        f"REGIONS does not cover exactly the archive's 30 channels: "
        f"missing={names_a - region_channels} extra={region_channels - names_a}"
    )
    return sorted(names_a)


def channel_region_map(channel_order: list[str]) -> dict:
    ch_to_region = {}
    for region, chans in REGIONS.items():
        for c in chans:
            ch_to_region[c] = region
    return {i: ch_to_region[c] for i, c in enumerate(channel_order)}


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_table = pd.read_csv(TRIAL_TABLE_CSV)
    epoch_index = pd.read_csv(EPOCH_INDEX_CSV, low_memory=False)
    return trial_table, epoch_index


def _corr_broadband(raw_a: np.ndarray, raw_b: np.ndarray) -> np.ndarray:
    """raw_a, raw_b: (n_win_samples, n_channels, n_epochs) unfiltered. Returns
    (n_channels, n_epochs) Pearson correlation, NaN where either side has
    zero variance in the window."""
    a = raw_a - raw_a.mean(axis=0, keepdims=True)
    b = raw_b - raw_b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def _plv(analytic_a: np.ndarray, analytic_b: np.ndarray) -> np.ndarray:
    """analytic_a/b: (n_win_samples, n_channels, n_epochs) complex. Returns
    (n_channels, n_epochs)."""
    phase_diff = np.angle(analytic_a) - np.angle(analytic_b)
    return np.abs(np.mean(np.exp(1j * phase_diff), axis=0))


def _coh(analytic_a: np.ndarray, analytic_b: np.ndarray) -> np.ndarray:
    num = np.abs(np.sum(analytic_a * np.conj(analytic_b), axis=0))
    den = np.sqrt(
        np.sum(np.abs(analytic_a) ** 2, axis=0) * np.sum(np.abs(analytic_b) ** 2, axis=0)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def _time_domain_stats(raw_win: np.ndarray, t_win_ms: np.ndarray) -> dict:
    """raw_win: (n_win_samples, n_channels, n_epochs) unfiltered.
    Returns dict of stat -> (n_channels, n_epochs)."""
    n_win = raw_win.shape[0]
    mean_ = raw_win.mean(axis=0)
    std_ = raw_win.std(axis=0)
    ptp_ = raw_win.max(axis=0) - raw_win.min(axis=0)
    abs_idx = np.argmax(np.abs(raw_win), axis=0)  # (n_channels, n_epochs)
    ch_idx, ep_idx = np.indices(abs_idx.shape)
    peakamp = raw_win[abs_idx, ch_idx, ep_idx]
    peaklat = t_win_ms[abs_idx].astype(np.float64)
    t_sec = t_win_ms.astype(np.float64) / 1000.0
    t_centered = t_sec - t_sec.mean()
    denom = float((t_centered ** 2).sum())
    if denom == 0 or n_win < 2:
        slope = np.full(mean_.shape, np.nan)
    else:
        centered = raw_win - mean_[None, :, :]
        slope = (t_centered[:, None, None] * centered).sum(axis=0) / denom
    return {
        "mean": mean_,
        "std": std_,
        "ptp": ptp_,
        "peakamp": peakamp,
        "peaklat": peaklat,
        "slope": slope,
    }


def process_session(
    session_id: str,
    trial_table: pd.DataFrame,
    channel_order: list[str],
    sos_filters: dict,
    ch_region: dict,
) -> dict:
    """Processes one session across both stimuli. Returns a dict with:
      - 'trials': DataFrame, one row per (round, trial), the session's
        official trial list.
      - per-stimulus feature blocks for deceiver-role and observer-role
        single-brain features, dyadic features, and the region network
        tensor -- all NaN-filled for excluded trials or an unavailable
        stimulus.
    """
    deceiver_rows = (
        trial_table[
            (trial_table["session_id"] == session_id) & (trial_table["role"] == "deceiver")
        ]
        .sort_values(["round", "trial"])
        .reset_index(drop=True)
    )
    n_trials = len(deceiver_rows)
    trials_df = deceiver_rows[["round", "trial"]].copy()

    result = {"trials": trials_df, "n_trials": n_trials}

    for stimulus in pp.STIMULI:
        code = STIM_CODE[stimulus]
        mat_path = pp.PREPROCESSED_DIR / stimulus / f"{session_id}.mat"
        block = {
            "extracted_mask": np.zeros(n_trials, dtype=bool),
            "excluded_reason": np.full(n_trials, None, dtype=object),
        }
        if not mat_path.exists():
            block["excluded_reason"][:] = "no_archive_file"
            result[code] = block
            continue

        smat = pp.SessionMat(mat_path)
        fs = smat.fs()
        t_axis = smat.t_axis()
        epoch_keys = smat.epoch_keys(stimulus)
        row_to_epoch, ratio, exclusions = pp.compute_alignment(
            deceiver_rows, epoch_keys, stimulus
        )
        excl_by_pos = {e["row_position"]: e["reason"] for e in exclusions}

        if row_to_epoch is None:
            block["excluded_reason"][:] = "alignment_not_recoverable"
            result[code] = block
            continue

        valid_positions = [p for p in range(n_trials) if p not in excl_by_pos]
        e_idx = np.array([row_to_epoch[p] for p in valid_positions], dtype=int)
        block["extracted_mask"][valid_positions] = True
        for p, reason in excl_by_pos.items():
            block["excluded_reason"][p] = reason

        # Reindex channel axis to the canonical order for BOTH roles.
        names = smat.channel_names("deceiver")
        assert smat.channel_names("observer") == names, (
            f"{session_id}/{stimulus}: deceiver/observer clab differ"
        )
        perm = [names.index(c) for c in channel_order]

        x_dec = smat.x("deceiver")[:, perm, :]  # (n_samples, 30, n_epochs_archive)
        x_obs = smat.x("observer")[:, perm, :]

        n_epochs_archive = x_dec.shape[2]

        # Bandpass + Hilbert on the FULL epoch, once per band, reused across
        # all windows (see module docstring).
        analytic_dec = {}
        analytic_obs = {}
        for band, sos in sos_filters.items():
            filt_dec = sig.sosfiltfilt(sos, x_dec, axis=0)
            filt_obs = sig.sosfiltfilt(sos, x_obs, axis=0)
            analytic_dec[band] = sig.hilbert(filt_dec, axis=0).astype(np.complex64)
            analytic_obs[band] = sig.hilbert(filt_obs, axis=0).astype(np.complex64)
        del filt_dec, filt_obs

        n_ch = len(channel_order)
        n_reg = len(REGION_NAMES)

        pow_dec = {}   # (window, band) -> (n_ch, n_trials) NaN-filled
        pow_obs = {}
        td_dec = {}    # (window, stat) -> (n_ch, n_trials)
        td_obs = {}
        dy_corr = {}   # window -> (n_ch, n_trials)
        dy_plv = {}    # (window, band) -> (n_ch, n_trials)
        dy_coh = {}    # (window, band) -> (n_ch, n_trials)
        region_tensor = np.full((n_reg, n_reg, len(BANDS), 3, n_trials), np.nan, dtype=np.float32)

        for win_idx, (window_name, start_ms, end_ms) in enumerate(pp.WINDOWS[stimulus]):
            mask = (t_axis >= start_ms) & (t_axis <= end_ms)
            t_win = t_axis[mask]
            raw_dec_win_full = x_dec[mask, :, :]  # (n_win, 30, n_epochs_archive)
            raw_obs_win_full = x_obs[mask, :, :]

            def _gather(full_arr):
                out = np.full((full_arr.shape[0], full_arr.shape[1], n_trials), np.nan, dtype=full_arr.dtype)
                out[:, :, valid_positions] = full_arr[:, :, e_idx]
                return out

            raw_dec_win = _gather(raw_dec_win_full)
            raw_obs_win = _gather(raw_obs_win_full)

            td_stats_dec = _time_domain_stats(raw_dec_win, t_win)
            td_stats_obs = _time_domain_stats(raw_obs_win, t_win)
            for stat in TD_STATS:
                td_dec[(window_name, stat)] = td_stats_dec[stat]
                td_obs[(window_name, stat)] = td_stats_obs[stat]

            dy_corr[window_name] = _corr_broadband(raw_dec_win, raw_obs_win)

            for band_idx, band in enumerate(BANDS):
                a_dec_full = analytic_dec[band][mask, :, :]
                a_obs_full = analytic_obs[band][mask, :, :]
                a_dec = np.full((a_dec_full.shape[0], n_ch, n_trials), np.nan + 0j, dtype=np.complex64)
                a_obs = np.full((a_obs_full.shape[0], n_ch, n_trials), np.nan + 0j, dtype=np.complex64)
                a_dec[:, :, valid_positions] = a_dec_full[:, :, e_idx]
                a_obs[:, :, valid_positions] = a_obs_full[:, :, e_idx]

                pow_dec[(window_name, band)] = np.nanmean(np.abs(a_dec) ** 2, axis=0)
                pow_obs[(window_name, band)] = np.nanmean(np.abs(a_obs) ** 2, axis=0)

                # PLV/coh only need valid trials -- compute on the gathered
                # (already-NaN-safe-shaped) arrays but only over valid
                # positions to avoid NaN propagation through angle()/abs().
                plv_out = np.full((n_ch, n_trials), np.nan)
                coh_out = np.full((n_ch, n_trials), np.nan)
                if len(valid_positions) > 0:
                    a_dec_v = a_dec_full[:, :, e_idx]
                    a_obs_v = a_obs_full[:, :, e_idx]
                    plv_out[:, valid_positions] = _plv(a_dec_v, a_obs_v)
                    coh_out[:, valid_positions] = _coh(a_dec_v, a_obs_v)
                dy_plv[(window_name, band)] = plv_out
                dy_coh[(window_name, band)] = coh_out

                # Region-level network tensor (deceiver-region x observer-region PLV).
                if len(valid_positions) > 0:
                    reg_a = np.zeros((a_dec_v.shape[0], n_reg, a_dec_v.shape[2]), dtype=np.complex64)
                    reg_b = np.zeros((a_obs_v.shape[0], n_reg, a_obs_v.shape[2]), dtype=np.complex64)
                    for ri, rname in enumerate(REGION_NAMES):
                        cols = [i for i in range(n_ch) if ch_region[i] == rname]
                        reg_a[:, ri, :] = a_dec_v[:, cols, :].mean(axis=1)
                        reg_b[:, ri, :] = a_obs_v[:, cols, :].mean(axis=1)
                    for ri in range(n_reg):
                        for rj in range(n_reg):
                            phase_diff = np.angle(reg_a[:, ri, :]) - np.angle(reg_b[:, rj, :])
                            plv_reg = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))
                            region_tensor[ri, rj, band_idx, win_idx, valid_positions] = plv_reg.astype(np.float32)

        block.update(
            pow_dec=pow_dec, pow_obs=pow_obs, td_dec=td_dec, td_obs=td_obs,
            dy_corr=dy_corr, dy_plv=dy_plv, dy_coh=dy_coh,
            region_tensor=region_tensor, valid_positions=valid_positions,
        )
        result[code] = block
        del x_dec, x_obs, analytic_dec, analytic_obs

    return result


def build_feature_dictionary(channel_order: list[str]) -> pd.DataFrame:
    rows = []
    windur_s = {
        ("DecisionMaking", "pre"): 0.50, ("DecisionMaking", "onset"): 0.50, ("DecisionMaking", "post"): 1.00,
        ("Feedback", "pre"): 0.20, ("Feedback", "onset"): 0.30, ("Feedback", "post"): 0.69,
    }
    windows = {"DecisionMaking": ["pre", "onset", "post"], "Feedback": ["pre", "onset", "post"]}

    for stimulus in pp.STIMULI:
        code = STIM_CODE[stimulus]
        for window in windows[stimulus]:
            dur = windur_s[(stimulus, window)]
            n_samp_val = next(
                int(round((e - s) / 10)) + 1
                for name, s, e in pp.WINDOWS[stimulus] if name == window
            )
            for ch in channel_order:
                for band, (lo, hi) in BANDS.items():
                    tier, n_cycles = reliability_tier(dur, lo)
                    rows.append(dict(
                        feature_name=f"pow_{band}_{ch}_{code}_{window}", table="single_brain",
                        feature_group="freq", channel=ch, band=band, stimulus=stimulus, window=window,
                        statistic="mean_hilbert_power", metric=None, reliability=tier,
                        reliability_reason=f"n_cycles={n_cycles:.2f} at band low edge {lo}Hz over {dur*1000:.0f}ms window",
                        n_samples_in_window=n_samp_val, window_duration_ms=dur * 1000, n_cycles=round(n_cycles, 3),
                    ))
                for stat in TD_STATS:
                    rows.append(dict(
                        feature_name=f"td_{stat}_{ch}_{code}_{window}", table="single_brain",
                        feature_group="time", channel=ch, band=None, stimulus=stimulus, window=window,
                        statistic=stat, metric=None, reliability=None, reliability_reason=None,
                        n_samples_in_window=n_samp_val, window_duration_ms=dur * 1000, n_cycles=None,
                    ))
            # dyadic
            for ch in channel_order:
                rows.append(dict(
                    feature_name=f"dy_corr_broad_{ch}_{code}_{window}", table="dyadic",
                    feature_group="dyadic", channel=ch, band="broad", stimulus=stimulus, window=window,
                    statistic=None, metric="corr", reliability="reliable",
                    reliability_reason="pearson correlation, no cycle-count constraint",
                    n_samples_in_window=n_samp_val, window_duration_ms=dur * 1000, n_cycles=None,
                ))
                for metric in ["plv", "coh"]:
                    for band, (lo, hi) in BANDS.items():
                        tier, n_cycles = reliability_tier(dur, lo)
                        rows.append(dict(
                            feature_name=f"dy_{metric}_{band}_{ch}_{code}_{window}", table="dyadic",
                            feature_group="dyadic", channel=ch, band=band, stimulus=stimulus, window=window,
                            statistic=None, metric=metric, reliability=tier,
                            reliability_reason=(
                                f"n_cycles={n_cycles:.2f} at band low edge {lo}Hz over {dur*1000:.0f}ms window; "
                                "single-trial PLV/coherency are additionally upward-biased at small sample counts"
                            ),
                            n_samples_in_window=n_samp_val, window_duration_ms=dur * 1000, n_cycles=round(n_cycles, 3),
                        ))
    return pd.DataFrame(rows)


def assemble_single_brain(
    session_results: dict, trial_table: pd.DataFrame, channel_order: list[str]
) -> pd.DataFrame:
    key_cols = [
        "pair_id", "session_id", "session_order", "round", "trial", "dyad_trial_seq",
        "participant_id", "partner_id", "role", "condition", "outcome",
    ]
    behavioral_cols = [
        "reaction_time_sec", "trials_so_far", "prior_deception_count", "prior_deception_rate",
        "prior_outcome", "prior_condition", "pinfo_bart_score",
    ]
    all_rows = []
    for session_id, res in session_results.items():
        trials_df = res["trials"]
        n_trials = res["n_trials"]
        pos_by_rt = {(r, t): p for p, (r, t) in enumerate(zip(trials_df["round"], trials_df["trial"]))}

        sess_trial_table = trial_table[trial_table["session_id"] == session_id]

        for _, participant_row in sess_trial_table.iterrows():
            pos = pos_by_rt.get((participant_row["round"], participant_row["trial"]))
            if pos is None:
                continue
            role = participant_row["role"]
            row = {c: participant_row[c] for c in key_cols}
            for c in behavioral_cols:
                row[c] = participant_row[c]

            any_stim_available = False
            for stimulus in pp.STIMULI:
                code = STIM_CODE[stimulus]
                block = res[code]
                extracted = block["extracted_mask"][pos]
                reason_col = f"{code}_excluded_reason"
                if not extracted:
                    # Leave pow_*/td_* columns for this stimulus unset --
                    # pd.DataFrame(all_rows) fills any key absent from a
                    # given row's dict with NaN once columns are unioned
                    # across all rows, which is exactly the "explicitly
                    # null, never imputed" behavior required here.
                    row[reason_col] = block["excluded_reason"][pos]
                    continue
                row[reason_col] = None
                any_stim_available = True
                pow_side = block["pow_dec"] if role == "deceiver" else block["pow_obs"]
                td_side = block["td_dec"] if role == "deceiver" else block["td_obs"]
                for (window_name, band), arr in pow_side.items():
                    for ci, ch in enumerate(channel_order):
                        row[f"pow_{band}_{ch}_{code}_{window_name}"] = arr[ci, pos]
                for (window_name, stat), arr in td_side.items():
                    for ci, ch in enumerate(channel_order):
                        row[f"td_{stat}_{ch}_{code}_{window_name}"] = arr[ci, pos]
            # Step 7: if NEITHER stimulus was extracted for this
            # participant-trial (the whole-session alignment_not_recoverable
            # case, sub22_sub19), the row is absent entirely -- not present
            # with all-null EEG columns. A prior version of this function
            # appended the row unconditionally, which put sub22_sub19's 968
            # participant-trials into single_brain.parquet with 22,264 total
            # rows instead of the expected ~21,296; caught in Step 9
            # validation and fixed here before any downstream artifact used
            # the file.
            if any_stim_available:
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df


def assemble_dyadic(
    session_results: dict, trial_table: pd.DataFrame, channel_order: list[str]
) -> pd.DataFrame:
    all_rows = []
    for session_id, res in session_results.items():
        trials_df = res["trials"]
        sess_deceiver = trial_table[
            (trial_table["session_id"] == session_id) & (trial_table["role"] == "deceiver")
        ].sort_values(["round", "trial"]).reset_index(drop=True)
        sess_observer = trial_table[
            (trial_table["session_id"] == session_id) & (trial_table["role"] == "observer")
        ].sort_values(["round", "trial"]).reset_index(drop=True)

        for pos in range(res["n_trials"]):
            dec_r = sess_deceiver.iloc[pos]
            obs_r = sess_observer.iloc[pos]
            any_stim_available = False
            row = {
                "pair_id": dec_r["pair_id"], "session_id": session_id,
                "round": dec_r["round"], "trial": dec_r["trial"],
                "dyad_trial_seq": dec_r["dyad_trial_seq"],
                "deceiver_id": dec_r["participant_id"], "observer_id": obs_r["participant_id"],
                "condition": dec_r["condition"], "outcome": dec_r["outcome"],
            }
            for stimulus in pp.STIMULI:
                code = STIM_CODE[stimulus]
                block = res[code]
                extracted = block["extracted_mask"][pos]
                reason_col = f"{code}_excluded_reason"
                if not extracted:
                    row[reason_col] = block["excluded_reason"][pos]
                    continue
                row[reason_col] = None
                any_stim_available = True
                for ci, ch in enumerate(channel_order):
                    for window_name, arr in block["dy_corr"].items():
                        row[f"dy_corr_broad_{ch}_{code}_{window_name}"] = arr[ci, pos]
                    for (window_name, band), arr in block["dy_plv"].items():
                        row[f"dy_plv_{band}_{ch}_{code}_{window_name}"] = arr[ci, pos]
                    for (window_name, band), arr in block["dy_coh"].items():
                        row[f"dy_coh_{band}_{ch}_{code}_{window_name}"] = arr[ci, pos]
            if any_stim_available:
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df


def build_interbrain_networks(session_results: dict, trial_table: pd.DataFrame) -> dict:
    """Region x region PLV tensor per trial, per stimulus. Returns a dict
    ready for np.savez_compressed: one array per stimulus of shape
    (n_trials_with_data, n_regions, n_regions, n_bands, n_windows), plus a
    parallel key table (session_id, round, trial) and the region names."""
    payload = {"region_names": np.array(REGION_NAMES), "band_names": np.array(list(BANDS.keys()))}
    for stimulus in pp.STIMULI:
        code = STIM_CODE[stimulus]
        tensors = []
        keys = []
        for session_id, res in session_results.items():
            block = res[code]
            if "region_tensor" not in block:
                continue
            trials_df = res["trials"]
            for pos in range(res["n_trials"]):
                if not block["extracted_mask"][pos]:
                    continue
                tensors.append(block["region_tensor"][:, :, :, :, pos])
                keys.append((session_id, int(trials_df.iloc[pos]["round"]), int(trials_df.iloc[pos]["trial"])))
        if tensors:
            payload[f"{code}_tensor"] = np.stack(tensors, axis=0)
            payload[f"{code}_session_id"] = np.array([k[0] for k in keys])
            payload[f"{code}_round"] = np.array([k[1] for k in keys])
            payload[f"{code}_trial"] = np.array([k[2] for k in keys])
    return payload


def run() -> dict:
    t0 = time.time()
    trial_table, epoch_index = load_tables()
    channel_order = build_channel_order()
    print(f"Canonical channel order ({len(channel_order)}): {channel_order}")
    sos_filters = build_sos_filters()
    gamma_psd_diagnostic()
    ch_region = channel_region_map(channel_order)

    session_ids = sorted(trial_table["session_id"].unique())
    session_results = {}
    for session_id in session_ids:
        print(f"Processing {session_id} ...")
        res = process_session(session_id, trial_table, channel_order, sos_filters, ch_region)
        session_results[session_id] = res

    print("Assembling single-brain table ...")
    single_brain = assemble_single_brain(session_results, trial_table, channel_order)
    print("Assembling dyadic table ...")
    dyadic = assemble_dyadic(session_results, trial_table, channel_order)
    print("Building feature dictionary ...")
    feature_dict = build_feature_dictionary(channel_order)
    print("Building inter-brain network tensor ...")
    networks = build_interbrain_networks(session_results, trial_table)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    single_brain_path = FEATURES_DIR / "single_brain.parquet"
    dyadic_path = FEATURES_DIR / "dyadic.parquet"
    dict_path = FEATURES_DIR / "feature_dictionary.csv"
    networks_path = FEATURES_DIR / "interbrain_networks.npz"

    print(f"single_brain shape: {single_brain.shape}")
    print(f"dyadic shape: {dyadic.shape}")
    print(f"feature_dictionary rows: {len(feature_dict)}")

    single_brain.to_parquet(single_brain_path, index=False)
    dyadic.to_parquet(dyadic_path, index=False)
    feature_dict.to_csv(dict_path, index=False)
    np.savez_compressed(networks_path, **networks)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. Wrote:")
    print(f"  {single_brain_path}")
    print(f"  {dyadic_path}")
    print(f"  {dict_path}")
    print(f"  {networks_path}")

    return {
        "single_brain_shape": single_brain.shape,
        "dyadic_shape": dyadic.shape,
        "feature_dict_rows": len(feature_dict),
    }


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
