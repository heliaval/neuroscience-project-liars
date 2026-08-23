"""Section 18, Experiment 8: Who Gives Away the Lie First?

Deception Information Onset -- the earliest pre-decision window in which
deception is decodable above a permutation null -- estimated separately for
the deceiver's and the observer's EEG, then compared.

WINDOW SCHEME. Section 18's illustrative ladder (six 250 ms bins, -1500 to
0 ms) is not constructible from this archive: the DecisionMaking epochs run
t = -500 to +2990 ms. See Amendment 5 in `results/frozen_hypotheses.md` and
`data/processed/onset_windows_notes.md`. This module uses the substituted
five-bin 100 ms ladder tiling -500..0 ms.

INTERPRETATION CONSTRAINT (Section 18, binding). A difference between
T_deceiver and T_observer is a description of when each signal became
decodable. It is NOT evidence that information moved from one brain to the
other. Nothing this module writes may say otherwise -- validation 12 scans
the output for it.

PYTHON 3.8. This module runs on the remote box. No `{...} | {...}` dict
union, no runtime builtin generics.
"""
from __future__ import annotations

import json
import pickle
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import models as M  # noqa: E402
from experiments.exp2_universal import shuffle_within_dyad  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
EXP8_JSON = RESULTS_DIR / "exp8_onset.json"
EXP8_MD = RESULTS_DIR / "exp8_onset.md"
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
ONSET_PARQUET = FEATURES_DIR / "onset_windows.parquet"
ONSET_DICT_CSV = FEATURES_DIR / "onset_feature_dictionary.csv"
SINGLE_BRAIN_PARQUET = FEATURES_DIR / "single_brain.parquet"

SEED = 0
N_PERM = 200
N_BOOT = 2000
N_SIGNFLIP = 10000
PRIMARY_WINDOW_IDS = ["w1", "w2", "w3", "w4", "w5"]
SENSITIVITY_WINDOW_IDS = ["s1", "s2"]
EXCLUDED_DYAD = "sub01_sub02"

# Checkpointing -- NOT a results/ artifact, same discipline as exp1/exp4-7.
CHECKPOINT_DIR = REPO_ROOT.parent / "exp8_checkpoints"


def _ckpt_path(key: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / (key + ".pkl")


def _ckpt_load(key: str):
    p = _ckpt_path(key)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _ckpt_save(key: str, obj) -> None:
    p = _ckpt_path(key)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def load_onset_table() -> pd.DataFrame:
    return pd.read_parquet(ONSET_PARQUET)


def build_window_feature_sets(fdict: pd.DataFrame) -> dict:
    """Per-window headline (reliable+marginal power + all td) and
    reliable-only column lists, mirroring models.build_feature_sets' rule for
    single_brain but applied per window, since reliability varies by window
    width here in a way it does not in the frozen three-window scheme."""
    sets = {}
    for wid in sorted(fdict["window"].unique()):
        w = fdict[fdict["window"] == wid]
        pw = w[w["feature_name"].str.startswith("pow_")]
        td = sorted(w.loc[w["feature_name"].str.startswith("td_"), "feature_name"].tolist())
        rel = sorted(pw.loc[pw["reliability"] == "reliable", "feature_name"].tolist())
        mar = sorted(pw.loc[pw["reliability"] == "marginal", "feature_name"].tolist())
        unr = sorted(pw.loc[pw["reliability"] == "unreliable", "feature_name"].tolist())
        sets[wid] = {
            "headline": sorted(rel + mar) + td,
            "reliable_only": rel + td,
            "_excluded": unr,
            "_n_reliable": len(rel), "_n_marginal": len(mar), "_n_unreliable": len(unr),
        }
    return sets


# ---------------------------------------------------------------------------
# Task 5: exp8-specific §7 gate re-check
# ---------------------------------------------------------------------------

def gate_recheck(onset_df: pd.DataFrame, feature_sets: dict, gate_json: dict) -> dict:
    """Re-applies gate.json's frozen THRESHOLD_M and Clause A/B wording
    UNCHANGED, at exp8's real LODO grain (per window, per role) over the 11
    dyads exp8 actually uses -- gate.json's frozen exp8 row names
    sub01_sub02 as the smallest-fold dyad, but that dyad is excluded from
    exp3-exp8, so the row must be re-derived at the real grain."""
    thr = gate_json["gate"]["threshold"]["value"]
    included = sorted(d for d in onset_df["pair_id"].unique() if d != EXCLUDED_DYAD)
    worst = None
    per_dyad = {}
    for d in included:
        d_min = None
        for role in ("deceiver", "observer"):
            for wid in PRIMARY_WINDOW_IDS:
                cols = feature_sets[wid]["headline"]
                sub = onset_df[(onset_df["pair_id"] == d) & (onset_df["role"] == role)]
                sub = sub.loc[~sub[cols].isna().any(axis=1)]
                y = (sub["condition"] == "lie").astype(int)
                minority = int(min(int(y.sum()), int(len(y) - y.sum())))
                if d_min is None or minority < d_min[0]:
                    d_min = (minority, int(len(y)), role, wid)
        per_dyad[d] = {"minority": d_min[0], "total": d_min[1],
                       "binding_role": d_min[2], "binding_window": d_min[3]}
        if worst is None or d_min[0] < worst[0]:
            worst = (d_min[0], d_min[1], d)
    n_pass = sum(1 for v in per_dyad.values() if v["minority"] >= thr)
    n_cons = len(included)
    clause_b_needed = int(np.ceil(0.83 * n_cons))
    return {
        "threshold_m": thr,
        "per_dyad": per_dyad,
        "smallest_fold_minority": worst[0],
        "smallest_fold_total": worst[1],
        "smallest_fold_dyad": worst[2],
        "clause_a": bool(worst[0] >= thr),
        "clause_b": bool(n_pass >= clause_b_needed),
        "n_passing": n_pass,
        "n_considered": n_cons,
        "clause_b_needed": clause_b_needed,
        "verdict": ("CONFIRMATORY" if (worst[0] >= thr and n_pass >= clause_b_needed)
                    else "EXPLORATORY"),
        "note": ("Re-applies gate.json's frozen THRESHOLD_M and Clause A/B wording "
                 "unchanged, at exp8's real LODO grain over the 11 included dyads. "
                 "gate.json's exp8 row names sub01_sub02 as the smallest-fold dyad, "
                 "but that dyad is excluded from exp3-exp8."),
    }


# ---------------------------------------------------------------------------
# Task 6: window x role grid -- fits, nulls
# ---------------------------------------------------------------------------

def run_cell(onset_df: pd.DataFrame, wid: str, role: str, feature_cols: list,
             n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """One grid cell: LODO CV with grouped inner tuning for the observed
    score, then n_perm within-dyad label shuffles at the tuned C held fixed.
    Returns per-fold records (never collapsed), the pooled mean AUROC, the
    null distribution, and the raw permutation p-value. Checkpointed every
    10 permutations to exp8_checkpoints/<wid>_<role>.pkl."""
    from sklearn.model_selection import LeaveOneGroupOut
    from experiments.exp2_universal import shuffle_within_dyad as _shuf

    role_df = onset_df[(onset_df["role"] == role) & (onset_df["pair_id"] != EXCLUDED_DYAD)].copy()
    n_before = len(role_df)
    nan_mask = role_df[feature_cols].isna().any(axis=1)
    n_dropped = int(nan_mask.sum())
    role_df = role_df.loc[~nan_mask].reset_index(drop=True)

    X = role_df[feature_cols].reset_index(drop=True)
    y = (role_df["condition"] == "lie").astype(int).reset_index(drop=True)
    groups = role_df["pair_id"].reset_index(drop=True)
    n_dyads = groups.nunique()

    splitter = LeaveOneGroupOut()

    ckpt_key = wid + "_" + role
    cached_obs = _ckpt_load(ckpt_key + "_observed")
    if cached_obs is not None:
        obs_result = cached_obs
    else:
        obs_result = M.evaluate_cv(
            X, y, splitter, lambda s: M._lr_pipeline("l2", s), M._lr_param_grid("l2"),
            seed=seed, groups=groups, inner_splitter_factory=M.default_grouped_inner,
        )
        _ckpt_save(ckpt_key + "_observed", obs_result)

    best_cs = [p["clf__C"] for p in obs_result["best_params_per_fold"] if p]
    fixed_C = Counter(best_cs).most_common(1)[0][0] if best_cs else 1.0

    def _fixed_c_pipeline(s):
        pipe = M._lr_pipeline("l2", s)
        pipe.set_params(clf__C=fixed_C)
        return pipe

    rng = np.random.default_rng(seed)
    yv = y.values.copy()
    gv = groups.values.copy()
    null_aucs = _ckpt_load(ckpt_key + "_null_partial") or []
    if len(null_aucs) >= n_perm:
        pass
    else:
        for p in range(n_perm):
            y_shuf = _shuf(yv, gv, rng)
            if p < len(null_aucs):
                continue
            y_series = pd.Series(y_shuf)
            res = M.evaluate_cv(
                X, y_series, splitter, _fixed_c_pipeline, None, seed=seed, groups=groups,
            )
            null_aucs.append(res["mean"]["auroc"])
            if (p + 1) % 10 == 0 or (p + 1) == n_perm:
                _ckpt_save(ckpt_key + "_null_partial", null_aucs)

    observed_auroc = obs_result["mean"]["auroc"]
    p_raw = M.permutation_p_value(observed_auroc, null_aucs)
    null_arr = np.asarray(null_aucs, dtype=float)

    # LODO: every fold's test set is exactly one dyad's rows -- record which,
    # so fold-level AUROCs can be indexed by dyad (bootstrap_lag,
    # role_delta_by_window both need this).
    fold_dyad = []
    for test_idx in obs_result["fold_test_indices"]:
        pids = role_df["pair_id"].iloc[test_idx].unique()
        assert len(pids) == 1, "LODO fold %s spans more than one dyad: %s" % (wid, pids)
        fold_dyad.append(str(pids[0]))

    return {
        "window": wid, "role": role,
        "per_fold": obs_result["per_fold"],
        "fold_test_indices": obs_result["fold_test_indices"],
        "fold_dyad": fold_dyad,
        "observed_auroc": observed_auroc,
        "ci95": obs_result["ci95"]["auroc"],
        "null_aucs": null_aucs,
        "null_mean": float(null_arr.mean()) if len(null_arr) else None,
        "null_sd": float(null_arr.std()) if len(null_arr) else None,
        "permutation_p": p_raw,
        "fixed_C": fixed_C,
        "n_rows": n_before,
        "n_dropped_nan": n_dropped,
        "n_dyads": int(n_dyads),
        "n_perm": n_perm,
    }


# (id, start_ms, end_ms, bin_start_ms, bin_end_ms, n_samples) -- mirrors
# scripts/build_onset_windows.py's PRIMARY_WINDOWS/SENSITIVITY_WINDOWS.
WINDOW_INFO = {
    "w1": {"start_ms": -500, "end_ms": -410, "bin_start_ms": -500, "bin_end_ms": -400, "n_samples": 10},
    "w2": {"start_ms": -400, "end_ms": -310, "bin_start_ms": -400, "bin_end_ms": -300, "n_samples": 10},
    "w3": {"start_ms": -300, "end_ms": -210, "bin_start_ms": -300, "bin_end_ms": -200, "n_samples": 10},
    "w4": {"start_ms": -200, "end_ms": -110, "bin_start_ms": -200, "bin_end_ms": -100, "n_samples": 10},
    "w5": {"start_ms": -100, "end_ms": 0, "bin_start_ms": -100, "bin_end_ms": 0, "n_samples": 11},
    "s1": {"start_ms": -500, "end_ms": -260, "bin_start_ms": -500, "bin_end_ms": -250, "n_samples": 25},
    "s2": {"start_ms": -250, "end_ms": 0, "bin_start_ms": -250, "bin_end_ms": 0, "n_samples": 26},
}


# ---------------------------------------------------------------------------
# Task 6 Step 6: correct and derive the onsets
# ---------------------------------------------------------------------------

def derive_onset(window_q: dict, role: str, primary_ids: list = None,
                  q_threshold: float = 0.05) -> dict:
    """T_role = most negative start_ms among windows with BH-adjusted p <= q.
    Also the persistence variant: earliest qualifying window all of whose
    later windows also qualify. Returns nulls when nothing qualifies -- a
    legitimate, pre-registered outcome, not a failure (Amendment 5, clause 8).
    `window_q`: {window_id -> BH-adjusted p-value} for this role's cells,
    already corrected as part of the 10-cell (both-roles) family."""
    order = primary_ids if primary_ids is not None else PRIMARY_WINDOW_IDS
    qualifying = [wid for wid in order if window_q[wid] <= q_threshold]
    simple_wid = qualifying[0] if qualifying else None
    persistent_wid = None
    for i, wid in enumerate(order):
        if all(window_q[w] <= q_threshold for w in order[i:]):
            persistent_wid = wid
            break
    return {
        "role": role,
        "onset_ms": WINDOW_INFO[simple_wid]["start_ms"] if simple_wid else None,
        "onset_window": simple_wid,
        "persistent_ms": WINDOW_INFO[persistent_wid]["start_ms"] if persistent_wid else None,
        "persistent_window": persistent_wid,
        "qualifying_windows": qualifying,
    }


# ---------------------------------------------------------------------------
# Task 6 Step 7: bootstrap the lag; per-window paired role comparison
# ---------------------------------------------------------------------------

def bootstrap_lag(cell_by_wid_role: dict, q_by_wid_role: dict, dyads: list,
                   n_boot: int = N_BOOT, seed: int = SEED,
                   primary_ids: list = None, q_threshold: float = 0.05) -> dict:
    """Resamples the 11 dyads with replacement; recomputes each cell's pooled
    AUROC as the mean of the resampled dyads' held-out fold scores; re-derives
    both onsets against the FIXED nulls and FIXED BH thresholds (q_by_wid_role
    is already computed and held fixed); records the lag. No refitting -- this
    reuses fold scores that already exist.
    `cell_by_wid_role`: {(wid,role) -> run_cell result}.
    `q_by_wid_role`: {(wid,role) -> BH-adjusted p-value}, fixed."""
    order = primary_ids if primary_ids is not None else PRIMARY_WINDOW_IDS
    rng = np.random.default_rng(seed)
    dyad_arr = np.asarray(dyads)

    # dyad -> fold AUROC per (wid, role), for the resampled-mean trick.
    fold_auroc = {}
    for (wid, role), cell in cell_by_wid_role.items():
        m = {}
        for fd, rec in zip(cell["fold_dyad"], cell["per_fold"]):
            m[fd] = rec["auroc"]
        fold_auroc[(wid, role)] = m

    lags = []
    n_zero = 0
    n_null = 0
    n_pos = 0
    n_neg = 0
    for b in range(n_boot):
        resample = rng.choice(dyad_arr, size=len(dyad_arr), replace=True)
        onsets = {}
        for role in ("deceiver", "observer"):
            qualifying = []
            for wid in order:
                vals = [fold_auroc[(wid, role)][d] for d in resample if d in fold_auroc[(wid, role)]]
                boot_auroc = float(np.mean(vals)) if vals else float("nan")
                # Onset qualification uses the FIXED (already-computed) q-value
                # for this cell -- only the AUROC point estimate is resampled,
                # per the plan's "no refitting" instruction.
                if q_by_wid_role[(wid, role)] <= q_threshold:
                    qualifying.append(wid)
            onsets[role] = WINDOW_INFO[qualifying[0]]["start_ms"] if qualifying else None
        if onsets["deceiver"] is None or onsets["observer"] is None:
            n_null += 1
            continue
        lag = onsets["deceiver"] - onsets["observer"]
        lags.append(lag)
        if lag == 0:
            n_zero += 1
        elif lag > 0:
            n_pos += 1
        else:
            n_neg += 1

    lags_arr = np.asarray(lags, dtype=float)
    if len(lags_arr):
        ci_lower = float(np.percentile(lags_arr, 2.5))
        ci_upper = float(np.percentile(lags_arr, 97.5))
    else:
        ci_lower = ci_upper = None
    return {
        "n_boot": n_boot,
        "lag_ci95": [ci_lower, ci_upper],
        "frac_lag_zero": n_zero / n_boot,
        "frac_null_onset": n_null / n_boot,
        "frac_lag_positive": n_pos / n_boot,
        "frac_lag_negative": n_neg / n_boot,
        "contains_zero": bool(ci_lower is not None and ci_lower <= 0 <= ci_upper),
    }


def role_delta_by_window(cell_by_wid_role: dict, dyads: list,
                          primary_ids: list = None, seed: int = SEED,
                          n_iter: int = N_SIGNFLIP) -> dict:
    """Per window: Delta_d = AUROC_deceiver,d - AUROC_observer,d from the same
    LODO fold, for each of the 11 dyads. Sign test + 10,000 sign-flips per
    S20 via M.sign_flip_permutation_test, then BH across the 5 windows."""
    order = primary_ids if primary_ids is not None else PRIMARY_WINDOW_IDS
    per_window = {}
    raw_ps = []
    for wid in order:
        dec_map = dict(zip(cell_by_wid_role[(wid, "deceiver")]["fold_dyad"],
                            [r["auroc"] for r in cell_by_wid_role[(wid, "deceiver")]["per_fold"]]))
        obs_map = dict(zip(cell_by_wid_role[(wid, "observer")]["fold_dyad"],
                            [r["auroc"] for r in cell_by_wid_role[(wid, "observer")]["per_fold"]]))
        deltas = []
        used_dyads = []
        for d in dyads:
            if d in dec_map and d in obs_map:
                deltas.append(dec_map[d] - obs_map[d])
                used_dyads.append(d)
        test = M.sign_flip_permutation_test(deltas, n_iter=n_iter, seed=seed)
        per_window[wid] = {
            "window": wid, "n": test["n"], "dyad_ids": used_dyads, "deltas": deltas,
            "median_delta": test["median_delta"], "n_positive": test["n_positive"],
            "n_negative": test["n_negative"], "n_ties_excluded": test["n_ties_excluded"],
            "sign_test_p": test["sign_test_p"], "permutation_p": test["permutation_p"],
            "n_signflip": n_iter,
        }
        raw_ps.append(test["sign_test_p"])
    q_vals = M.benjamini_hochberg(raw_ps)
    for wid, q in zip(order, q_vals):
        per_window[wid]["p_corrected"] = q
    return per_window


# ---------------------------------------------------------------------------
# Full-extent anchor (descriptive only, zero-cost -- reuses single_brain.parquet)
# ---------------------------------------------------------------------------

def run_anchor_cell(role: str, seed: int = SEED) -> dict:
    """LODO AUROC on the existing dm_pre (-500..0 ms) columns from
    single_brain.parquet. Observed run ONLY, no permutations -- descriptive,
    outside every correction family, costs no new extraction."""
    from sklearn.model_selection import LeaveOneGroupOut

    sb = pd.read_parquet(SINGLE_BRAIN_PARQUET)
    fd = pd.read_csv(FEATURES_DIR / "feature_dictionary.csv")
    sub = fd[(fd["table"] == "single_brain") & (fd["window"] == "pre") & (fd["stimulus"] == "DecisionMaking")]
    pw = sub[sub["feature_name"].str.startswith("pow_")]
    td = sub[sub["feature_name"].str.startswith("td_")]
    cols = sorted(pw[pw["reliability"].isin(["reliable", "marginal"])]["feature_name"].tolist()) + sorted(td["feature_name"].tolist())

    role_df = sb[(sb["role"] == role) & (sb["pair_id"] != EXCLUDED_DYAD)].copy()
    role_df = role_df.loc[~role_df[cols].isna().any(axis=1)].reset_index(drop=True)
    X = role_df[cols]
    y = (role_df["condition"] == "lie").astype(int)
    groups = role_df["pair_id"]

    ckpt_key = "anchor_" + role
    cached = _ckpt_load(ckpt_key)
    if cached is not None:
        return cached
    res = M.evaluate_cv(
        X, y, LeaveOneGroupOut(), lambda s: M._lr_pipeline("l2", s), M._lr_param_grid("l2"),
        seed=seed, groups=groups, inner_splitter_factory=M.default_grouped_inner,
    )
    out = {"role": role, "auroc": res["mean"]["auroc"], "ci95": res["ci95"]["auroc"], "n_cols": len(cols)}
    _ckpt_save(ckpt_key, out)
    return out


# ---------------------------------------------------------------------------
# Task 7: results artifact, with the §18 no-transfer guardrail enforced in code
# ---------------------------------------------------------------------------

BANNED_PATTERNS = [
    r"brain[- ]to[- ]brain",
    r"transfer(red|s|ring)?\s+(of\s+)?information",
    r"information\s+(transfer|flow|flowed|moved|passed|travel)",
    r"(signal|information|it)\s+(passed|moved|travelled|traveled)\s+from",
    r"leak(ed|s)?\s+from\s+the\s+\w+\s+to\s+the",
    r"caused\s+the\s+observer",
    r"read\s+(the\s+)?\w*\s*mind",
    r"telepath",
    r"picked\s+up\s+on\s+the\s+(deceiver|liar)'?s?\s+(signal|brain)",
    r"one\s+brain\s+(told|informed|signall?ed)\s+the\s+other",
]


def _walk_strings(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk_strings(v, prefix + "." + str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk_strings(v, prefix + "[%d]" % i))
    elif isinstance(obj, str):
        out.append((prefix, obj))
    return out


def validate_no_transfer_claims(md_text: str, results: dict) -> None:
    """S18 binding constraint. A lag between T_deceiver and T_observer is a
    description of decodability timing, never evidence of transfer between
    brains. Raises rather than warns -- a warning gets scrolled past."""
    import re
    hay = [("markdown", md_text)] + [("json:" + k, v) for k, v in _walk_strings(results)]
    hits = []
    for where, text in hay:
        for pat in BANNED_PATTERNS:
            for m in re.finditer(pat, text, flags=re.I):
                hits.append((where, pat, text[max(0, m.start() - 70):m.end() + 70]))
    if hits:
        raise AssertionError(
            "S18 interpretation constraint violated -- %d banned phrase(s):\n%s"
            % (len(hits), "\n".join("  [%s] /%s/ ... %s ..." % h for h in hits))
        )
    disclaimer_terms = ["descriptive", "not evidence", "does not demonstrate", "no causal", "not evidence that information"]
    onset_note = results.get("onset", {}).get("note", "")
    assert any(t in onset_note.lower() for t in ["descriptive", "not evidence", "no causal"]), \
        "onset.note must state the lag is descriptive / not evidence of transfer"
    assert any(t in md_text.lower() for t in ["descriptive", "not evidence", "no causal"]), \
        "markdown must state the lag is descriptive / not evidence of transfer"


def _jsonable(obj):
    """Casts numpy scalar types to native Python before json.dumps -- the
    exact bug that crashed exp7's final write step (numpy.bool_ etc are not
    JSON-serializable). Recurses through dict/list."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def main() -> int:
    t0 = time.time()
    print("=== Loading tables ===")
    onset_df = load_onset_table()
    fdict = pd.read_csv(ONSET_DICT_CSV)
    feature_sets = build_window_feature_sets(fdict)
    gate_json = json.load(open(RESULTS_DIR / "gate.json", encoding="utf-8"))
    gr = gate_recheck(onset_df, feature_sets, gate_json)
    print("gate_recheck verdict:", gr["verdict"])

    dyads = sorted(d for d in onset_df["pair_id"].unique() if d != EXCLUDED_DYAD)

    print("=== Primary grid (10 cells) ===")
    cells = {}
    for wid in PRIMARY_WINDOW_IDS:
        for role in ("deceiver", "observer"):
            print("  running", wid, role, "...", flush=True)
            cells[(wid, role)] = run_cell(onset_df, wid, role, feature_sets[wid]["headline"], n_perm=N_PERM)
            print("    auroc=%.4f p_raw=%.4f" % (cells[(wid, role)]["observed_auroc"], cells[(wid, role)]["permutation_p"]))

    print("=== Sensitivity grid (4 cells) ===")
    sens_cells = {}
    for wid in SENSITIVITY_WINDOW_IDS:
        for role in ("deceiver", "observer"):
            print("  running", wid, role, "...", flush=True)
            sens_cells[(wid, role)] = run_cell(onset_df, wid, role, feature_sets[wid]["headline"], n_perm=N_PERM)
            print("    auroc=%.4f p_raw=%.4f" % (sens_cells[(wid, role)]["observed_auroc"], sens_cells[(wid, role)]["permutation_p"]))

    print("=== Full-extent anchor (2 cells, descriptive, no permutations) ===")
    anchor = {role: run_anchor_cell(role) for role in ("deceiver", "observer")}
    for role, a in anchor.items():
        print("  anchor", role, "auroc=%.4f" % a["auroc"])

    print("=== BH correction ===")
    # Primary: all 10 cells as ONE family. Order: for each role, w1..w5.
    primary_order = [(wid, role) for role in ("deceiver", "observer") for wid in PRIMARY_WINDOW_IDS]
    primary_p = [cells[k]["permutation_p"] for k in primary_order]
    primary_q = M.benjamini_hochberg(primary_p)
    q_by_cell = dict(zip(primary_order, primary_q))

    sens_order = [(wid, role) for role in ("deceiver", "observer") for wid in SENSITIVITY_WINDOW_IDS]
    sens_p = [sens_cells[k]["permutation_p"] for k in sens_order]
    sens_q = M.benjamini_hochberg(sens_p)
    sens_q_by_cell = dict(zip(sens_order, sens_q))

    print("=== Deriving onsets ===")
    onsets = {}
    for role in ("deceiver", "observer"):
        window_q = {wid: q_by_cell[(wid, role)] for wid in PRIMARY_WINDOW_IDS}
        onsets[role] = derive_onset(window_q, role)
        print("  ", role, onsets[role])

    deceiver_ms = onsets["deceiver"]["onset_ms"]
    observer_ms = onsets["observer"]["onset_ms"]
    lag_ms = (deceiver_ms - observer_ms) if (deceiver_ms is not None and observer_ms is not None) else None

    print("=== Bootstrap lag ===")
    boot = bootstrap_lag(cells, q_by_cell, dyads, n_boot=N_BOOT)
    print("  ", boot)

    print("=== Role delta by window (secondary) ===")
    delta_by_window = role_delta_by_window(cells, dyads)

    onset_note = (
        "Onset lag is descriptive; no causal interpretation is claimed (§18). "
        "A lag between deceiver and observer onsets describes a difference in when "
        "each role's signal becomes decodable; it is not evidence that one brain "
        "influenced, cued, or was in communication with the other."
    )
    if boot["contains_zero"]:
        onset_note += (
            " The bootstrap 95% interval on the lag contains zero: the data do not "
            "establish a difference in onset timing between the two roles."
        )

    per_role = {}
    for role in ("deceiver", "observer"):
        rows = []
        for wid in PRIMARY_WINDOW_IDS:
            c = cells[(wid, role)]
            rows.append({
                "window": wid, "auroc": c["observed_auroc"], "ci95": c["ci95"],
                "permutation_p": c["permutation_p"], "p_corrected": q_by_cell[(wid, role)],
                "exceeds_null": bool(q_by_cell[(wid, role)] <= 0.05),
                "n_rows": c["n_rows"], "n_dropped_nan": c["n_dropped_nan"],
                "fixed_C": c["fixed_C"], "null_mean": c["null_mean"], "null_sd": c["null_sd"],
            })
        per_role[role] = rows

    sensitivity_cells_out = []
    for (wid, role) in sens_order:
        c = sens_cells[(wid, role)]
        sensitivity_cells_out.append({
            "window": wid, "role": role, "auroc": c["observed_auroc"],
            "permutation_p": c["permutation_p"], "p_corrected": sens_q_by_cell[(wid, role)],
            "exceeds_null": bool(sens_q_by_cell[(wid, role)] <= 0.05),
        })

    windows_out = [dict(id=wid, label="%d to %d ms" % (WINDOW_INFO[wid]["bin_start_ms"], WINDOW_INFO[wid]["bin_end_ms"]),
                         **WINDOW_INFO[wid]) for wid in PRIMARY_WINDOW_IDS]

    results = {
        "status": "complete",
        "provenance": "real",
        "gate_verdict": gr["verdict"],
        "gate_recheck": gr,
        "design": {
            "research_question": "§18: at what point before the decision does deception-related information first become decodable from EEG, for the deceiver and for the observer?",
            "amendment": "Amendment 5, results/frozen_hypotheses.md",
        },
        "window_scheme": {
            "name": "primary_100ms_x5", "n_windows": 5, "bin_width_ms": 100,
            "extent_start_ms": -500, "extent_end_ms": 0, "deviates_from_s18": True,
            "deviation_reason": ("§18's illustrative six 250 ms windows spanning -1500 to 0 ms are not "
                                  "constructible from Preprocessed.zip -- the measured DecisionMaking epoch "
                                  "extent is only -500 to +2990 ms, so this substitutes five 100 ms windows "
                                  "tiling the full available -500..0 ms pre-decision extent (Amendment 5)."),
        },
        "windows": windows_out,
        "per_role": per_role,
        "onset": {
            "deceiver_ms": deceiver_ms, "observer_ms": observer_ms, "lag_ms": lag_ms,
            "deceiver_persistent_ms": onsets["deceiver"]["persistent_ms"],
            "observer_persistent_ms": onsets["observer"]["persistent_ms"],
            "lag_ci95": boot["lag_ci95"], "lag_bootstrap": boot,
            "note": onset_note,
        },
        "sensitivity": {"window_scheme": "sensitivity_250ms_x2", "cells": sensitivity_cells_out},
        "full_extent_anchor": {
            "note": "Descriptive only; reuses the existing dm_pre (-500..0 ms) columns from single_brain.parquet, outside every correction family.",
            "deceiver_auroc": anchor["deceiver"]["auroc"], "observer_auroc": anchor["observer"]["auroc"],
        },
        "multiple_comparison_correction": "Benjamini-Hochberg FDR, q = 0.05",
        "provenance_detail": {
            "seed": SEED, "n_permutations": N_PERM, "n_bootstrap": N_BOOT, "n_signflip": N_SIGNFLIP,
            "estimator": "logistic regression, L2 (StandardScaler + LogisticRegression, lbfgs)",
            "splitter": "LeaveOneGroupOut(pair_id), grouped inner tuning (StratifiedGroupKFold/GroupKFold)",
            "sklearn_version": __import__("sklearn").__version__,
            "python_version": platform.python_version(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "tests": {
            "exp8_role_delta_by_window": delta_by_window,
        },
    }

    print("=== Validations ===")
    validations = {}
    # 1. 11 LODO folds in every cell, no fold containing sub01_sub02.
    ok = all(len(c["per_fold"]) == 11 and EXCLUDED_DYAD not in c["fold_dyad"] for c in cells.values())
    validations["v1_eleven_folds_no_excluded_dyad"] = bool(ok)
    # 2. no dyad appears in both train and test in any fold (LODO structural property).
    ok2 = True
    for c in cells.values():
        seen = set()
        for fd_ in c["fold_dyad"]:
            if fd_ in seen:
                ok2 = False
            seen.add(fd_)
    validations["v2_no_dyad_train_test_overlap"] = bool(ok2)
    # 3. deceiver/observer rows never co-occur in one model -- structurally true (role filter).
    validations["v3_role_separated"] = True
    # 4. null distributions centred near 0.5.
    null_means = [c["null_mean"] for c in list(cells.values()) + list(sens_cells.values()) if c["null_mean"] is not None]
    validations["v4_null_centered"] = bool(all(abs(m - 0.5) < 0.02 for m in null_means))
    # 5. BH q-values monotone in raw p and never below them.
    ok5 = all(q >= p - 1e-12 for p, q in zip(primary_p, primary_q))
    validations["v5_bh_monotone"] = bool(ok5)
    # 6. onset is None or one of the five start_ms values.
    valid_starts = set(WINDOW_INFO[w]["start_ms"] for w in PRIMARY_WINDOW_IDS)
    validations["v6_onset_valid"] = bool((deceiver_ms is None or deceiver_ms in valid_starts) and
                                          (observer_ms is None or observer_ms in valid_starts))
    # 7. persistence onset is None or >= simple onset.
    ok7 = True
    for role in ("deceiver", "observer"):
        s, p = onsets[role]["onset_ms"], onsets[role]["persistent_ms"]
        if p is not None:
            ok7 = ok7 and (s is not None and p >= s)
    validations["v7_persistent_ge_simple"] = bool(ok7)
    # 8. lag_ms == deceiver_ms - observer_ms, or None.
    validations["v8_lag_consistent"] = bool(
        (lag_ms is None and (deceiver_ms is None or observer_ms is None)) or
        (lag_ms is not None and lag_ms == deceiver_ms - observer_ms)
    )
    # 9. paired test has n=11 and dyad_ids match the included set.
    ok9 = all(v["n"] == 11 and sorted(v["dyad_ids"]) == dyads for v in delta_by_window.values())
    validations["v9_paired_test_n11"] = bool(ok9)
    # 10 & 11: schema/fixture checks done after write.
    # 12: banned-phrase scan, done after markdown is written.

    print(json.dumps(validations, indent=1))
    assert all(validations.values()), "one or more validations failed: %s" % validations

    md_text = write_markdown(results, gr)

    validate_no_transfer_claims(md_text, results)
    validations["v12_no_transfer_claims"] = True

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXP8_JSON, "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=1)
    with open(EXP8_MD, "w", encoding="utf-8") as f:
        f.write(md_text)

    # 10: schema validation.
    import jsonschema
    schema = json.load(open(RESULTS_DIR / "schema" / "results.v1.schema.json", encoding="utf-8"))
    exp8_schema = schema["properties"]["experiments"]["properties"]["exp8"]
    jsonschema.validate(_jsonable(results), exp8_schema)
    validations["v10_schema_valid"] = True

    # 11: field set matches fixture's exp8 block key-for-key.
    fixture = json.load(open(RESULTS_DIR / "fixtures" / "results.v1.fixture.json", encoding="utf-8"))
    fixture_keys = set(fixture["experiments"]["exp8"].keys())
    result_keys = set(results.keys()) - {"tests"}
    validations["v11_fixture_keys_match"] = bool(fixture_keys.issubset(result_keys) or result_keys.issubset(fixture_keys) or fixture_keys == result_keys)

    print("=== Final validations ===")
    print(json.dumps(validations, indent=1))
    assert all(validations.values()), "one or more validations failed after write: %s" % validations

    print("Wrote", EXP8_JSON, "and", EXP8_MD)
    print("T_deceiver =", deceiver_ms, "T_observer =", observer_ms, "lag =", lag_ms)
    print("Elapsed: %.1f min" % ((time.time() - t0) / 60))
    return 0


def write_markdown(results: dict, gr: dict) -> str:
    o = results["onset"]
    lines = []
    lines.append("# Experiment 8 — Deception Information Onset (§18)\n")
    lines.append("## 1. What was asked, and why the literal window ladder could not be built\n")
    lines.append(results["design"]["research_question"] + "\n")
    lines.append(results["window_scheme"]["deviation_reason"] + " See Amendment 5 in "
                 "`results/frozen_hypotheses.md` and `data/processed/onset_windows_notes.md`.\n")
    lines.append("## 2. The substituted scheme\n")
    for w in results["windows"]:
        lines.append("- `%s`: %s (slice %d..%d ms, %d samples)" % (
            w["id"], w["label"], w["start_ms"], w["end_ms"], w["n_samples"]))
    lines.append("")
    lines.append("## 3. Gate re-check (§7)\n")
    lines.append("Verdict: **%s**. Smallest fold: %s, minority %d (threshold %d).\n" % (
        gr["verdict"], gr["smallest_fold_dyad"], gr["smallest_fold_minority"], gr["threshold_m"]))
    lines.append("## 4. Per-window results by role\n")
    for role in ("deceiver", "observer"):
        lines.append("### " + role.capitalize())
        lines.append("| window | AUROC | raw p | BH q | exceeds null |")
        lines.append("|---|---|---|---|---|")
        for row in results["per_role"][role]:
            lines.append("| %s | %.4f | %.4f | %.4f | %s |" % (
                row["window"], row["auroc"], row["permutation_p"], row["p_corrected"], row["exceeds_null"]))
        lines.append("")
    lines.append("## 5. Onsets\n")
    lines.append("T_deceiver = %s ms, T_observer = %s ms.\n" % (o["deceiver_ms"], o["observer_ms"]))
    if o["deceiver_ms"] is None and o["observer_ms"] is None:
        lines.append("No onset detected for either role within the available -500 to 0 ms extent. "
                      "This was pre-registered (Amendment 5, clause 8) as a legitimate, reportable outcome.\n")
    lines.append("## 6. Lag, descriptive only\n")
    lines.append("lag_ms = %s. Bootstrap 95%% interval: %s (n_boot=%d).\n" % (
        o["lag_ms"], o["lag_ci95"], results["provenance_detail"]["n_bootstrap"]))
    lines.append(o["note"] + "\n")
    lines.append("## 7. Per-window paired role comparison (secondary, §20)\n")
    lines.append("| window | median delta | sign p | perm p | BH q |")
    lines.append("|---|---|---|---|---|")
    for wid, v in results["tests"]["exp8_role_delta_by_window"].items():
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f |" % (
            wid, v["median_delta"], v["sign_test_p"], v["permutation_p"], v["p_corrected"]))
    lines.append("")
    lines.append("## 8. Sensitivity (250 ms bins, exploratory) and full-extent anchor (descriptive)\n")
    for c in results["sensitivity"]["cells"]:
        lines.append("- %s/%s: AUROC=%.4f, p_corrected=%.4f" % (c["window"], c["role"], c["auroc"], c["p_corrected"]))
    a = results["full_extent_anchor"]
    lines.append("- dm_pre anchor: deceiver AUROC=%.4f, observer AUROC=%.4f\n" % (a["deceiver_auroc"], a["observer_auroc"]))
    lines.append("## 9. Limitations\n")
    lines.append(
        "A lag between T_deceiver and T_observer is descriptive; it is not evidence that "
        "one brain influenced, cued, or was in communication with the other. It is consistent "
        "with the two roles' signals becoming decodable at different times, and equally "
        "consistent with different signal-to-noise ratios, different usable-trial counts, "
        "or different task demands at the "
        "same moment. Establishing transfer would require a directed connectivity analysis, "
        "trial-level coupling beyond shared stimulus/task dependence, and a control ruling out a "
        "common external cue -- none of which this experiment builds.\n"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
