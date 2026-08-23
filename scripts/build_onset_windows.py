"""Section 18 sliding pre-decision window feature extraction.

WHY THIS SCRIPT EXISTS
Section 18's illustrative ladder -- six 250 ms windows spanning -1500 to
0 ms -- is NOT constructible from `Preprocessed.zip`. The DecisionMaking
epochs run t = -500 to +2990 ms, 350 samples, fs = 100 Hz (re-measured
directly from the archive by this script's own preflight check, not taken
on faith from any planning document). Four of Section 18's six windows lie
entirely outside the recorded data. This was already found and reported by
the Section 9 preprocessing pass (`data/processed/eeg_preprocessing_notes.md`,
"Windows chosen"), which declined to download `Raw.zip` to work around it.
This script does not download `Raw.zip` either. It builds the best honest
ladder the archive actually supports and records the substitution.

WHAT IT DOES NOT DO
It does not re-run preprocessing, does not read or write
`data/processed/epochs/`, and does not modify `src/preprocessing.py` or
`src/features.py`. It re-slices the same `.mat` archive with a finer window
list, reusing those modules' alignment, filtering, and statistics code
verbatim.

FILTER ORDER
Bandpass + Hilbert run once on the FULL 350-sample epoch, then windows are
sliced out of the analytic signal -- copied from `src/features.py`'s
process_session. This is the only valid order here: a 10-sample window is
far too short to filter directly, and doing so would return edge artefacts.

AXIS CONVENTIONS (verified against src/features.py, not assumed)
`pp._slice_window` returns (n_epochs_selected, n_channels, n_window_samples)
-- selected/valid epochs only, not NaN-padded to the full trial count.
`feat._time_domain_stats` expects (n_window_samples, n_channels, n_epochs)
and returns per-stat arrays shaped (n_channels, n_epochs). These two shapes
are NOT directly compatible: `_slice_window`'s output must be transposed
(n_epochs, n_channels, n_samples) -> (n_samples, n_channels, n_epochs)
before being passed to `_time_domain_stats`, and each returned stat
transposed back (n_channels, n_epochs) -> (n_epochs, n_channels) before
being scattered into the NaN-filled full-trial-count array. This was
confirmed by reading `src/features.py:436-598` (`process_session`) end to
end, which does the equivalent gather manually rather than via
`_slice_window`; the reconstruction check in `--self-check-session` is the
empirical proof this script's axis handling is right.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import scipy.signal as sig

import preprocessing as pp
import features as feat

SEED = 0
STIMULUS = "DecisionMaking"          # exp8 is pre-decision only; Feedback unused
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "processed" / "features"
NOTES = REPO_ROOT / "data" / "processed" / "onset_windows_notes.md"
TRIAL_TABLE_CSV = REPO_ROOT / "data" / "processed" / "trial_table.csv"

# (id, start_ms, end_ms, bin_start_ms, bin_end_ms)
# start_ms/end_ms are INCLUSIVE slice bounds -- pp._slice_window takes both
# endpoints, so each end is pulled back one sample (10 ms) to tile without
# sharing a boundary sample. bin_* is the nominal 100 ms bin, for labels only.
PRIMARY_WINDOWS = [
    ("w1", -500, -410, -500, -400),
    ("w2", -400, -310, -400, -300),
    ("w3", -300, -210, -300, -200),
    ("w4", -200, -110, -200, -100),
    ("w5", -100,    0, -100,    0),
]

SENSITIVITY_WINDOWS = [
    ("s1", -500, -260, -500, -250),
    ("s2", -250,    0, -250,    0),
]

ALL_WINDOWS = PRIMARY_WINDOWS + SENSITIVITY_WINDOWS


def check_window_tiling(t_axis: np.ndarray) -> None:
    """Asserts the primary ladder tiles the pre-decision extent exactly once.
    Printed rather than assumed, so the run's stdout is the evidence."""
    pre = t_axis[t_axis <= 0]
    covered = []
    for wid, start, end, bs, be in PRIMARY_WINDOWS:
        sel = t_axis[(t_axis >= start) & (t_axis <= end)]
        print("  %s: bin [%d, %d) -> slice [%d, %d] ms, %d samples"
              % (wid, bs, be, start, end, len(sel)))
        covered.extend(sel.tolist())
    covered = np.array(sorted(covered))
    assert len(covered) == len(np.unique(covered)), "primary windows overlap"
    assert covered.tolist() == pre.tolist(), (
        "primary windows do not tile the pre-decision extent exactly: "
        "covered %d of %d samples" % (len(covered), len(pre))
    )
    print("  tiling OK: %d samples, %d..%d ms, no gaps, no overlaps"
          % (len(covered), covered[0], covered[-1]))


def print_reliability_table() -> dict:
    """Applies src/features.py's own reliability_tier rule to the new windows.
    Returns {(window_id, band): (tier, n_cycles)}."""
    tiers = {}
    print("Band reliability by window (rule: n_cycles = duration_s * band_low_hz;"
          " >=3 reliable, >=1 marginal, <1 unreliable):")
    for wid, start, end, bs, be in ALL_WINDOWS:
        dur_s = (be - bs) / 1000.0
        row = []
        for band, (lo, hi) in feat.BANDS.items():
            tier, n_cyc = feat.reliability_tier(dur_s, lo)
            tiers[(wid, band)] = (tier, n_cyc)
            row.append("%s=%s(%.2f)" % (band, tier[:3], n_cyc))
        print("  %s (%d ms): %s" % (wid, be - bs, " ".join(row)))
    return tiers


def _slice_analytic_window(a_full, t_axis, epoch_indices, start_ms, end_ms):
    """Same masking logic as pp._slice_window, but preserves the complex
    dtype. pp._slice_window casts its return value to float32, which
    silently discards the imaginary part when handed an analytic (Hilbert)
    signal -- confirmed by the reconstruction self-check failing at
    max_abs_rel_diff ~= 0.88 until this was split out. features.py's own
    process_session never routes the analytic signal through
    `_slice_window` either -- it masks manually for exactly this reason.
    a_full: (n_samples, n_channels, n_epochs_archive) complex64.
    Returns (n_epochs_selected, n_channels, n_window_samples) complex64."""
    mask = (t_axis >= start_ms) & (t_axis <= end_ms)
    sub = a_full[mask, :, :]
    sub = sub[:, :, epoch_indices]
    return np.transpose(sub, (2, 1, 0))


def process_session_onset(session_id, trial_table, channel_order, sos_filters):
    """One session, DecisionMaking only, both roles, all windows in ALL_WINDOWS.
    Returns (trials_df, {(role, window_id): {'pow': {...}, 'td': {...}}},
    extracted_mask, excluded_reason). NaN-filled for excluded trials."""
    deceiver_rows = (
        trial_table[(trial_table["session_id"] == session_id)
                    & (trial_table["role"] == "deceiver")]
        .sort_values(["round", "trial"]).reset_index(drop=True)
    )
    n_trials = len(deceiver_rows)
    mat_path = pp.PREPROCESSED_DIR / STIMULUS / (session_id + ".mat")
    extracted = np.zeros(n_trials, dtype=bool)
    reason = np.full(n_trials, None, dtype=object)
    if not mat_path.exists():
        reason[:] = "no_archive_file"
        return deceiver_rows[["round", "trial"]], {}, extracted, reason

    smat = pp.SessionMat(mat_path)
    t_axis = smat.t_axis()
    row_to_epoch, ratio, exclusions = pp.compute_alignment(
        deceiver_rows, smat.epoch_keys(STIMULUS), STIMULUS
    )
    excl_by_pos = dict((e["row_position"], e["reason"]) for e in exclusions)
    if row_to_epoch is None:
        reason[:] = "alignment_not_recoverable"
        return deceiver_rows[["round", "trial"]], {}, extracted, reason

    valid = [p for p in range(n_trials) if p not in excl_by_pos]
    e_idx = np.array([row_to_epoch[p] for p in valid], dtype=int)
    extracted[valid] = True
    for p, r in excl_by_pos.items():
        reason[p] = r

    names = smat.channel_names("deceiver")
    assert smat.channel_names("observer") == names, (
        session_id + ": deceiver/observer clab differ"
    )
    perm = [names.index(c) for c in channel_order]
    x = {"deceiver": smat.x("deceiver")[:, perm, :],
         "observer": smat.x("observer")[:, perm, :]}

    # Bandpass + Hilbert on the FULL epoch, once per band -- see module docstring.
    analytic = {}
    for role in ("deceiver", "observer"):
        analytic[role] = {}
        for band, sos in sos_filters.items():
            filt = sig.sosfiltfilt(sos, x[role], axis=0)
            analytic[role][band] = sig.hilbert(filt, axis=0).astype(np.complex64)

    n_ch = len(channel_order)
    out = {}
    for role in ("deceiver", "observer"):
        for wid, start, end, bs, be in ALL_WINDOWS:
            cell = {"pow": {}, "td": {}}
            for band in feat.BANDS:
                # (n_valid, n_ch, n_win_samples), complex64 preserved
                a_win = _slice_analytic_window(
                    analytic[role][band], t_axis, e_idx, start, end
                )
                # mean squared envelope over the sample axis (last) -> (n_valid, n_ch)
                p = (np.abs(a_win) ** 2).mean(axis=-1)
                full = np.full((n_trials, n_ch), np.nan, dtype=np.float32)
                full[valid] = p.astype(np.float32)
                cell["pow"][band] = full
            # (n_valid, n_ch, n_win_samples) -- raw, unfiltered
            raw_win = pp._slice_window(x[role], t_axis, e_idx, start, end)
            t_win = t_axis[(t_axis >= start) & (t_axis <= end)]
            # _time_domain_stats expects (n_win_samples, n_channels, n_epochs);
            # _slice_window gives (n_epochs, n_channels, n_win_samples) -- transpose.
            raw_win_t = np.transpose(raw_win, (2, 1, 0))
            stats = feat._time_domain_stats(raw_win_t, t_win)
            for stat_name, arr in stats.items():
                # arr: (n_channels, n_valid) -> transpose to (n_valid, n_channels)
                arr_t = np.asarray(arr, dtype=np.float32).T
                full = np.full((n_trials, n_ch), np.nan, dtype=np.float32)
                full[valid] = arr_t
                cell["td"][stat_name] = full
            out[(role, wid)] = cell
    return deceiver_rows[["round", "trial"]], out, extracted, reason


def _feature_name(kind, band_or_stat, ch, wid):
    return "%s_%s_%s_dm_%s" % (kind, band_or_stat, ch, wid)


def _self_check_session(session_id: str) -> None:
    """Builds a temporary (-500, 0) window on one session and confirms its
    pow_gamma_* values match single_brain.parquet's pow_gamma_<ch>_dm_pre
    for the same trials. This is the real check: s2 (-250,0) is NOT the
    same window as dm_pre (-500,0), so only a window that IS reconstructible
    proves the pipeline matches src/features.py's own numbers."""
    trial_table = pd.read_csv(TRIAL_TABLE_CSV)
    channel_order = feat.build_channel_order()
    sos_filters = feat.build_sos_filters()

    deceiver_rows = (
        trial_table[(trial_table["session_id"] == session_id)
                    & (trial_table["role"] == "deceiver")]
        .sort_values(["round", "trial"]).reset_index(drop=True)
    )
    n_trials = len(deceiver_rows)
    mat_path = pp.PREPROCESSED_DIR / STIMULUS / (session_id + ".mat")
    smat = pp.SessionMat(mat_path)
    t_axis = smat.t_axis()
    row_to_epoch, ratio, exclusions = pp.compute_alignment(
        deceiver_rows, smat.epoch_keys(STIMULUS), STIMULUS
    )
    excl_by_pos = dict((e["row_position"], e["reason"]) for e in exclusions)
    valid = [p for p in range(n_trials) if p not in excl_by_pos]
    e_idx = np.array([row_to_epoch[p] for p in valid], dtype=int)

    names = smat.channel_names("deceiver")
    perm = [names.index(c) for c in channel_order]
    x_dec = smat.x("deceiver")[:, perm, :]

    sos = sos_filters["gamma"]
    filt = sig.sosfiltfilt(sos, x_dec, axis=0)
    analytic = sig.hilbert(filt, axis=0).astype(np.complex64)
    a_win = _slice_analytic_window(analytic, t_axis, e_idx, -500, 0)  # (n_valid, n_ch, n_samples)
    p = (np.abs(a_win) ** 2).mean(axis=-1)  # (n_valid, n_ch)

    sb = pd.read_parquet(OUT_DIR / "single_brain.parquet",
                          columns=["session_id", "role", "round", "trial"]
                          + ["pow_gamma_%s_dm_pre" % c for c in channel_order])
    sb = sb[(sb["session_id"] == session_id) & (sb["role"] == "deceiver")]
    sb = sb.merge(deceiver_rows.iloc[valid][["round", "trial"]].assign(_ord=range(len(valid))),
                   on=["round", "trial"], how="inner").sort_values("_ord")
    assert len(sb) == len(valid), "row count mismatch in self-check: %d vs %d" % (len(sb), len(valid))
    ref = sb[["pow_gamma_%s_dm_pre" % c for c in channel_order]].to_numpy(dtype=np.float64)
    got = p.astype(np.float64)
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12)
    max_rel = float(np.nanmax(rel))
    print("dm_pre reconstruction check: max_abs_rel_diff = %.3e" % max_rel)
    if max_rel < 1e-4:
        print("MATCH")
    else:
        print("MISMATCH")
        raise AssertionError("dm_pre reconstruction check failed: max_abs_rel_diff=%.3e" % max_rel)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check-session", default=None)
    args = parser.parse_args()

    if args.self_check_session:
        _self_check_session(args.self_check_session)
        return 0

    print("=== Step 1: measured extent ===")
    paths = sorted((pp.PREPROCESSED_DIR / STIMULUS).glob("*.mat"))
    print("n session files:", len(paths))
    s0 = pp.SessionMat(paths[0])
    t_axis0 = s0.t_axis()
    print("fs=%d  t=[%d, %d] ms  n_samples=%d" % (s0.fs(), t_axis0[0], t_axis0[-1], len(t_axis0)))

    print("=== Step 2: tiling check ===")
    check_window_tiling(t_axis0)

    print("=== Step 3: reliability table ===")
    print_reliability_table()

    print("=== Step 4: filter stability ===")
    sos_filters = feat.build_sos_filters()
    channel_order = feat.build_channel_order()

    print("=== Step 5: extraction, %d sessions ===" % len(paths))
    trial_table = pd.read_csv(TRIAL_TABLE_CSV)
    session_ids = [p.stem for p in paths]

    all_rows = []
    n_extracted_total = 0
    n_excluded_total = 0
    reason_counts = {}
    dict_rows = []
    dict_written = False

    for i, sid in enumerate(session_ids):
        trials_df, out, extracted, reason = process_session_onset(
            sid, trial_table, channel_order, sos_filters
        )
        n_trials = len(trials_df)
        n_ext = int(extracted.sum())
        n_exc = n_trials - n_ext
        n_extracted_total += n_ext
        n_excluded_total += n_exc
        for r in reason:
            if r is not None:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        print("  [%2d/%2d] %s: %d trials, %d extracted, %d excluded"
              % (i + 1, len(session_ids), sid, n_trials, n_ext, n_exc))

        for pos in range(n_trials):
            r = trials_df.iloc[pos]
            for role in ("deceiver", "observer"):
                role_row = trial_table[
                    (trial_table["session_id"] == sid)
                    & (trial_table["role"] == role)
                    & (trial_table["round"] == r["round"])
                    & (trial_table["trial"] == r["trial"])
                ]
                if len(role_row) == 0:
                    continue
                rr = role_row.iloc[0]
                row = {
                    "pair_id": rr["pair_id"], "session_id": sid,
                    "participant_id": rr["participant_id"],
                    "round": int(r["round"]), "trial": int(r["trial"]),
                    "dyad_trial_seq": rr["dyad_trial_seq"], "role": role,
                    "condition": rr["condition"],
                    "extracted": bool(extracted[pos]),
                    "excluded_reason": reason[pos],
                }
                if out:
                    for wid, _, _, _, _ in ALL_WINDOWS:
                        cell = out[(role, wid)]
                        for band in feat.BANDS:
                            vals = cell["pow"][band][pos]  # (n_ch,)
                            for ci, ch in enumerate(channel_order):
                                v = vals[ci]
                                row[_feature_name("pow", band, ch, wid)] = None if np.isnan(v) else float(v)
                        for stat_name in feat.TD_STATS:
                            vals = cell["td"][stat_name][pos]
                            for ci, ch in enumerate(channel_order):
                                v = vals[ci]
                                row[_feature_name("td", stat_name, ch, wid)] = None if np.isnan(v) else float(v)
                all_rows.append(row)

    print("=== Step 6: assemble table ===")
    wide = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = OUT_DIR / "onset_windows.parquet"
    wide.to_parquet(out_parquet, index=False)
    print("Wrote %s: %d rows x %d cols" % (out_parquet, wide.shape[0], wide.shape[1]))

    print("=== Step 7: feature dictionary ===")
    dict_rows = []
    tiers = print_reliability_table()
    for wid, start, end, bs, be in ALL_WINDOWS:
        n_samples = (end - start) // 10 + 1
        dur_ms = be - bs
        for band in feat.BANDS:
            tier, n_cyc = tiers[(wid, band)]
            for ch in channel_order:
                dict_rows.append({
                    "table": "onset_windows",
                    "feature_name": _feature_name("pow", band, ch, wid),
                    "window": wid, "band": band, "stat": None, "channel": ch,
                    "reliability": tier, "n_cycles": n_cyc,
                    "n_samples_in_window": n_samples, "window_duration_ms": dur_ms,
                })
        for stat_name in feat.TD_STATS:
            for ch in channel_order:
                dict_rows.append({
                    "table": "onset_windows",
                    "feature_name": _feature_name("td", stat_name, ch, wid),
                    "window": wid, "band": None, "stat": stat_name, "channel": ch,
                    "reliability": None, "n_cycles": None,
                    "n_samples_in_window": n_samples, "window_duration_ms": dur_ms,
                })
    fdict = pd.DataFrame(dict_rows)
    out_dict = OUT_DIR / "onset_feature_dictionary.csv"
    fdict.to_csv(out_dict, index=False)
    print("Wrote %s: %d rows" % (out_dict, len(fdict)))

    print("=== Final counts ===")
    print("extracted participant-trials:", n_extracted_total)
    print("excluded participant-trials:", n_excluded_total)
    print("reason breakdown:", reason_counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
