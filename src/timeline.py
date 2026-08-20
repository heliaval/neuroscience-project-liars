"""
src/timeline.py -- Section 6 trial-table reconstruction.

Rebuilds the per-trial interaction timeline that every downstream experiment in
`can_your_brain_learn_a_liar_workflow.md` consumes, from the raw figshare
"behavioral log and trigger timestamp" archive plus the participant-info
spreadsheet. Writes `data/processed/trial_table.csv`.

--------------------------------------------------------------------------
SOURCE DATA (found by inspection in step 2 of plans/reconstruct-trial-table.md)
--------------------------------------------------------------------------

`data/raw/behavioral/behavioral log and trigger timestamp/` holds 23 sessions,
each a (Player_subXX, Observer_subYY) pair, each with two files:

  Player_subXX_Observer_subYY_Behavioral.txt
      2 header lines, then one row per completed trial:
      round trial trial_type card_num card_col player_response player_input
      Obs_input score sub_rea obs_rea
      - round: 1..11 (block index, resets every session)
      - trial: 1..44 (index within round)
      - trial_type: card-color instruction code (L/S/T)
      - card_num / card_col: the displayed card's number and color code
      - player_response: "lie" or "true" -- the ONLY place the deception
        condition is recorded; it describes the player's decision for this
        trial and applies to the whole trial (both participants' rows).
      - player_input: the number the player reported
      - Obs_input: the observer's raw guess, header-documented as
        "Lie[0]/Truth[1]" -- cross-checked against the Timestamp file's
        explicit `obsInp_Lie` / `obsInp_True` trigger labels and against the
        `result_trial_observer_correct/incorrect` trigger for the same
        trial, and found to be INVERTED from its own header: the data
        actually has Obs_input=1 when the observer guessed "Lie" and
        Obs_input=0 when they guessed "True". This is preserved verbatim in
        `observer_guess_raw` and documented, not silently "fixed"; the
        canonical `observer_guess` column is derived from the Timestamp
        file's unambiguous text labels instead.
      - score: "[player_delta, observer_delta]" as a Python-literal string,
        e.g. ['+10', '0'].
      - sub_rea / obs_rea: player's / observer's reaction time in seconds.

  Player_subXX_Observer_subYY_Timestamp.txt
      No header. Each line: "<date> <time> <elapsed_seconds> <event_name>".
      `obsInp_*` events occur exactly once per completed trial in every
      session (verified: 484 of 484, matching 11 rounds x 44 trials, in all
      23 sessions) and are used as the anchor. `playerinput_*` events are
      NOT reliably one-per-trial -- a player can submit an invalid input
      (`error_player_input_wrong` / `error_player_input_timeout`) and retry,
      producing 2+ `playerinput_*` events before the trial resolves; this
      was caught empirically (13 of 23 sessions had more `playerinput_*`
      events than completed trials) and is why each trial's player-side
      trigger is taken as the *last* `playerinput_*` event seen before the
      matching `obsInp_*`, not a naive positional zip -- see
      `_session_trigger_timestamps`.
      `obsInp_*` is used as the observer-side trigger timestamp (decision
      onset); the paired `playerinput_*` as the player-side trigger
      timestamp. The session's `experiment_start` event's wall-clock time is
      used to order a dyad's sessions chronologically (see session_order
      below).

`Participant information and risk taking tendency.xlsx` (single sheet,
24 rows): `Participant No.` ("subNN"), `Gender`, `Age`, `BART Score` (the
risk-taking-tendency measure named in the workflow doc). The id scheme
("subNN") matches the behavioral log's ids exactly -- no remapping needed.

--------------------------------------------------------------------------
DATA-QUALITY NOTE: sessions per dyad
--------------------------------------------------------------------------
23 sessions exist for 12 dyads. Every dyad has a session with each member as
Player *except* the (sub01, sub02) dyad, which has only one session
(sub01 as Player) -- the reverse-role session is simply absent from the
archive. This means sub02 never appears as Player anywhere in this dataset.
Flagged again at runtime by the validators below.

--------------------------------------------------------------------------
GRAIN AND SORT
--------------------------------------------------------------------------
Long form: one row per (trial x participant) -- two rows per completed
trial, distinguished by `role`. Chosen per the plan because downstream
observer-only analyses and per-role trial counts both index on
participant-within-trial; the wide form would force a melt later.

Sort key: (pair_id, session_order, round, trial, role). This deliberately
sorts by *session first*, not by round-then-session as the plan's literal
"(pair_id, round)" shorthand suggested -- round numbers (1..11) reset at
the start of every session, and a dyad can have two sessions (deceiver role
reversed), so sorting by round before session would interleave two
unrelated sessions' round-1 trials instead of preserving true chronology.
`session_order` (1 = earliest, by the session's `experiment_start`
wall-clock time) makes the intended chronological grouping explicit and
verifiable, rather than leaning on trigger-timestamp tiebreaking alone.
Within a completed trial, the two rows (deceiver, observer) are ordered
deceiver-then-observer for stable output.

--------------------------------------------------------------------------
HISTORY FIELDS
--------------------------------------------------------------------------
"Previous interaction history" is a property of the *dyad's* chronological
trial sequence (computed once per trial from rows strictly earlier in that
same dyad's order, spanning both of its sessions if two exist), then copied
onto both of that trial's rows -- not computed separately per participant,
since there is exactly one deceiver per trial and the history describes the
shared interaction, per the plan's step-3 instruction.

--------------------------------------------------------------------------
EEG_WINDOW_REF -- placeholder, per plan step 3
--------------------------------------------------------------------------
No EEG data is downloaded in this task. `eeg_source_file_ref` records the
raw-EEG filename this trial's participant *would* map to, per the README's
documented Raw.zip naming convention (`Player_subXX.eeg` /
`Observer_subYY.eeg`) -- enough to locate the segment later. Window bounds
(`eeg_window_start`, `eeg_window_end`) are left null; populating them is
Section 9 / Phase 2 work, out of scope here.
"""

from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BEHAVIORAL_DIR = (
    REPO_ROOT / "data" / "raw" / "behavioral" / "behavioral log and trigger timestamp"
)
PARTICIPANT_XLSX = REPO_ROOT / "data" / "raw" / "Participant information and risk taking tendency.xlsx"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "trial_table.csv"

SESSION_NAME_RE = re.compile(r"^Player_(sub\d+)_Observer_(sub\d+)$")
TS_LINE_RE = re.compile(r"^(\S+ \S+)\s+([\d.]+)\s+(\S+)\s*$")

# Explicit dtypes for reading the CSV back (CSV does not preserve dtypes).
TRIAL_TABLE_DTYPES = {
    "pair_id": "string",
    "session_id": "string",
    "session_order": "Int64",
    "round": "Int64",
    "trial": "Int64",
    "dyad_trial_seq": "Int64",
    "participant_id": "string",
    "partner_id": "string",
    "role": "string",
    "role_raw": "string",
    "condition": "string",
    "condition_raw": "string",
    "outcome": "string",
    "observer_guess": "string",
    "observer_guess_raw": "Int64",
    "points": "Int64",
    "reaction_time_sec": "float64",
    "card_type_raw": "string",
    "card_number": "Int64",
    "card_color_code": "Int64",
    "trigger_ts_sec": "float64",
    "trigger_ts_wall": "string",
    "eeg_source_file_ref": "string",
    "eeg_window_start": "float64",
    "eeg_window_end": "float64",
    "trials_so_far": "Int64",
    "prior_deception_count": "Int64",
    "prior_deception_rate": "float64",
    "prior_outcome": "string",
    "prior_condition": "string",
    "pinfo_gender": "string",
    "pinfo_age": "Int64",
    "pinfo_bart_score": "Int64",
}


def read_participant_info(path: Path = PARTICIPANT_XLSX) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")
    df = df.rename(
        columns={
            "Participant No.": "participant_id",
            "Gender": "pinfo_gender",
            "Age": "pinfo_age",
            "BART Score": "pinfo_bart_score",
        }
    )
    return df[["participant_id", "pinfo_gender", "pinfo_age", "pinfo_bart_score"]]


def _parse_timestamp_events(path: Path) -> list[tuple[str, float, str]]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = TS_LINE_RE.match(line.rstrip("\n"))
            if not m:
                continue
            wall, elapsed, name = m.group(1), float(m.group(2)), m.group(3)
            events.append((wall, elapsed, name))
    return events


def _session_trigger_timestamps(events: list[tuple[str, float, str]]) -> dict:
    """Returns dict with ordered player/observer trigger events and session_start.

    `playerinput_*` events are NOT reliably one-per-trial: a player can submit
    an invalid input (`error_player_input_wrong` / `error_player_input_timeout`)
    and retry, producing 2+ `playerinput_*` events before the trial actually
    completes -- verified empirically (13 of 23 sessions have more
    `playerinput_*` events than completed trials). `obsInp_*` events, by
    contrast, occur exactly once per completed trial in every session (the
    observer only ever responds once, since the trial is already resolved by
    then). So each completed trial is anchored on its `obsInp_*` event, and
    the paired player-side trigger is the *last* `playerinput_*` event seen
    before that `obsInp_*` -- i.e. the input that actually stuck, discarding
    any earlier failed attempts for the same trial.
    """
    player_events = []
    observer_events = []
    observer_guess_labels = []
    last_playerinput = None
    for (w, e, name) in events:
        if name.startswith("playerinput_"):
            last_playerinput = (w, e)
        elif name.startswith("obsInp_"):
            player_events.append(last_playerinput)
            observer_events.append((w, e))
            guess = name.split("_", 1)[1]  # "Lie" or "True"
            observer_guess_labels.append(guess.lower().replace("true", "truth"))
    session_start = None
    for (w, e, name) in events:
        if name == "experiment_start":
            session_start = w
            break
    return {
        "player_events": player_events,
        "observer_events": observer_events,
        "observer_guess_labels": observer_guess_labels,
        "session_start_wall": session_start,
    }


def load_session(behavioral_path: Path, timestamp_path: Path) -> Optional[pd.DataFrame]:
    """Load one session's Behavioral + Timestamp files into a wide per-trial frame."""
    m = SESSION_NAME_RE.match(behavioral_path.stem.replace("_Behavioral", ""))
    if not m:
        print(f"WARNING: unexpected session filename, skipping: {behavioral_path.name}")
        return None
    player_id, observer_id = m.group(1), m.group(2)
    session_id = f"Player_{player_id}_Observer_{observer_id}"

    rows = []
    with open(behavioral_path, encoding="utf-8") as fh:
        lines = fh.readlines()
    data_lines = lines[2:]  # skip the 2 header lines
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 8)
        # round trial trial_type card_num card_col player_response player_input
        # Obs_input score sub_rea obs_rea  -- score is a bracketed list, may
        # contain spaces, so we take the remainder and re-split carefully.
        round_ = int(parts[0])
        trial = int(parts[1])
        trial_type = parts[2]
        card_num = int(parts[3])
        card_col = int(parts[4])
        player_response = parts[5]
        player_input = parts[6]
        obs_input_raw = int(parts[7])
        rest = parts[8]
        # rest = "['0', '+10'] 2.06080431718 0.899339127383"
        score_end = rest.index("]") + 1
        score_str = rest[:score_end]
        remainder = rest[score_end:].split()
        sub_rea, obs_rea = float(remainder[0]), float(remainder[1])
        score = ast.literal_eval(score_str)
        player_points = int(score[0])
        observer_points = int(score[1])
        rows.append(
            {
                "round": round_,
                "trial": trial,
                "trial_type": trial_type,
                "card_num": card_num,
                "card_col": card_col,
                "player_response": player_response,
                "player_input": player_input,
                "obs_input_raw": obs_input_raw,
                "player_points": player_points,
                "observer_points": observer_points,
                "sub_rea": sub_rea,
                "obs_rea": obs_rea,
            }
        )
    beh = pd.DataFrame(rows)

    events = _parse_timestamp_events(timestamp_path)
    ts = _session_trigger_timestamps(events)

    n_trials = len(beh)
    trigger_ok = len(ts["player_events"]) == n_trials and len(ts["observer_events"]) == n_trials
    if not trigger_ok:
        print(
            f"WARNING: {session_id}: trigger-event count mismatch "
            f"(trials={n_trials}, playerinput={len(ts['player_events'])}, "
            f"obsInp={len(ts['observer_events'])}). Trigger timestamps for this "
            f"session will be left null."
        )

    if trigger_ok:
        beh["player_trigger_wall"] = [w for (w, e) in ts["player_events"]]
        beh["player_trigger_sec"] = [e for (w, e) in ts["player_events"]]
        beh["observer_trigger_wall"] = [w for (w, e) in ts["observer_events"]]
        beh["observer_trigger_sec"] = [e for (w, e) in ts["observer_events"]]
        beh["observer_guess"] = ts["observer_guess_labels"]
    else:
        beh["player_trigger_wall"] = None
        beh["player_trigger_sec"] = float("nan")
        beh["observer_trigger_wall"] = None
        beh["observer_trigger_sec"] = float("nan")
        beh["observer_guess"] = None

    beh["session_id"] = session_id
    beh["player_id"] = player_id
    beh["observer_id"] = observer_id
    beh["session_start_wall"] = ts["session_start_wall"]
    return beh


def _pair_id(a: str, b: str) -> str:
    return "_".join(sorted([a, b]))


def _to_long_form(session_frames: list[pd.DataFrame]) -> pd.DataFrame:
    long_rows = []
    for beh in session_frames:
        player_id = beh["player_id"].iloc[0]
        observer_id = beh["observer_id"].iloc[0]
        session_id = beh["session_id"].iloc[0]
        pair_id = _pair_id(player_id, observer_id)
        for _, r in beh.iterrows():
            condition = "truth" if r["player_response"] == "true" else "lie"
            outcome = "correct" if r["observer_guess"] == condition else (
                "incorrect" if r["observer_guess"] is not None else None
            )
            base = dict(
                pair_id=pair_id,
                session_id=session_id,
                round=r["round"],
                trial=r["trial"],
                condition=condition,
                condition_raw=r["player_response"],
                outcome=outcome,
                observer_guess=r["observer_guess"],
                observer_guess_raw=r["obs_input_raw"],
                card_type_raw=r["trial_type"],
                card_number=r["card_num"],
                card_color_code=r["card_col"],
                session_start_wall=r["session_start_wall"],
            )
            long_rows.append(
                {
                    **base,
                    "participant_id": player_id,
                    "partner_id": observer_id,
                    "role": "deceiver",
                    "role_raw": "player",
                    "points": r["player_points"],
                    "reaction_time_sec": r["sub_rea"],
                    "trigger_ts_sec": r["player_trigger_sec"],
                    "trigger_ts_wall": r["player_trigger_wall"],
                    "eeg_source_file_ref": f"Player_{player_id}.eeg",
                }
            )
            long_rows.append(
                {
                    **base,
                    "participant_id": observer_id,
                    "partner_id": player_id,
                    "role": "observer",
                    "role_raw": "observer",
                    "points": r["observer_points"],
                    "reaction_time_sec": r["obs_rea"],
                    "trigger_ts_sec": r["observer_trigger_sec"],
                    "trigger_ts_wall": r["observer_trigger_wall"],
                    "eeg_source_file_ref": f"Observer_{observer_id}.eeg",
                }
            )
    return pd.DataFrame(long_rows)


def _assign_session_order(df: pd.DataFrame) -> pd.DataFrame:
    """Rank each dyad's sessions chronologically by experiment_start wall time."""
    session_starts = (
        df[["pair_id", "session_id", "session_start_wall"]]
        .drop_duplicates()
        .sort_values(["pair_id", "session_start_wall"])
    )
    session_starts["session_order"] = session_starts.groupby("pair_id").cumcount() + 1
    df = df.merge(
        session_starts[["session_id", "session_order"]], on="session_id", how="left"
    )
    return df


def _add_history_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dyad-level, strictly-backward-looking history fields.

    Computed once per (pair_id, trial) at trial grain -- not per role-row --
    then broadcast to both rows of that trial, since the history describes
    the shared dyad interaction and there is exactly one condition/outcome
    per trial.
    """
    trial_grain = (
        df[
            [
                "pair_id",
                "session_id",
                "session_order",
                "round",
                "trial",
                "condition",
                "outcome",
            ]
        ]
        .drop_duplicates(subset=["pair_id", "session_id", "round", "trial"])
        .sort_values(["pair_id", "session_order", "round", "trial"])
        .reset_index(drop=True)
    )

    trials_so_far = []
    prior_deception_count = []
    prior_deception_rate = []
    prior_outcome = []
    prior_condition = []

    for pair_id, grp in trial_grain.groupby("pair_id", sort=False):
        grp = grp.sort_values(["session_order", "round", "trial"])
        n_seen = 0
        n_lie = 0
        prev_outcome = None
        prev_condition = None
        for _, row in grp.iterrows():
            trials_so_far.append(n_seen)
            prior_deception_count.append(n_lie)
            prior_deception_rate.append(n_lie / n_seen if n_seen > 0 else float("nan"))
            prior_outcome.append(prev_outcome)
            prior_condition.append(prev_condition)
            # update using *current* row for the *next* iteration
            n_seen += 1
            if row["condition"] == "lie":
                n_lie += 1
            prev_outcome = row["outcome"]
            prev_condition = row["condition"]

    trial_grain = trial_grain.sort_values(["pair_id", "session_order", "round", "trial"]).reset_index(drop=True)
    trial_grain["trials_so_far"] = trials_so_far
    trial_grain["prior_deception_count"] = prior_deception_count
    trial_grain["prior_deception_rate"] = prior_deception_rate
    trial_grain["prior_outcome"] = prior_outcome
    trial_grain["prior_condition"] = prior_condition

    hist_cols = [
        "pair_id",
        "session_id",
        "round",
        "trial",
        "trials_so_far",
        "prior_deception_count",
        "prior_deception_rate",
        "prior_outcome",
        "prior_condition",
    ]
    df = df.merge(
        trial_grain[hist_cols],
        on=["pair_id", "session_id", "round", "trial"],
        how="left",
    )
    return df


def build_trial_table() -> pd.DataFrame:
    behavioral_files = sorted(BEHAVIORAL_DIR.glob("*_Behavioral.txt"))
    session_frames = []
    for beh_path in behavioral_files:
        ts_path = beh_path.with_name(beh_path.name.replace("_Behavioral.txt", "_Timestamp.txt"))
        if not ts_path.exists():
            print(f"WARNING: missing timestamp file for {beh_path.name}, skipping session")
            continue
        frame = load_session(beh_path, ts_path)
        if frame is not None:
            session_frames.append(frame)

    df = _to_long_form(session_frames)
    df = _assign_session_order(df)
    df = _add_history_fields(df)

    pinfo = read_participant_info()
    df = df.merge(pinfo, on="participant_id", how="left")

    # dyad_trial_seq: 1-based sequential index within the dyad's full
    # chronological trial order (spans sessions). Same for both rows of a
    # trial: trials_so_far + 1.
    df["dyad_trial_seq"] = df["trials_so_far"] + 1

    # eeg placeholders -- not populated in this task (see module docstring)
    df["eeg_window_start"] = pd.NA
    df["eeg_window_end"] = pd.NA

    df = df.sort_values(
        ["pair_id", "session_order", "round", "trial", "role"],
        key=lambda col: col.map({"deceiver": 0, "observer": 1}) if col.name == "role" else col,
    ).reset_index(drop=True)

    column_order = [
        "pair_id",
        "session_id",
        "session_order",
        "round",
        "trial",
        "dyad_trial_seq",
        "participant_id",
        "partner_id",
        "role",
        "role_raw",
        "condition",
        "condition_raw",
        "outcome",
        "observer_guess",
        "observer_guess_raw",
        "points",
        "reaction_time_sec",
        "card_type_raw",
        "card_number",
        "card_color_code",
        "trigger_ts_sec",
        "trigger_ts_wall",
        "eeg_source_file_ref",
        "eeg_window_start",
        "eeg_window_end",
        "trials_so_far",
        "prior_deception_count",
        "prior_deception_rate",
        "prior_outcome",
        "prior_condition",
        "pinfo_gender",
        "pinfo_age",
        "pinfo_bart_score",
    ]
    return df[column_order]


def write_trial_table(df: pd.DataFrame, path: Path = OUTPUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def read_trial_table(path: Path = OUTPUT_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype=TRIAL_TABLE_DTYPES)


# --------------------------------------------------------------------------
# Validators (step 5 of plans/reconstruct-trial-table.md)
# --------------------------------------------------------------------------


def validate_trial_table(df: pd.DataFrame) -> dict:
    """Runs the nine validations. Prints results and returns a summary dict.

    Raises AssertionError for checks that the plan says must stop the run
    (duplicate keys). Other checks report and continue.
    """
    report: dict = {}

    # 1. Total trial count / total row count
    n_rows = len(df)
    n_trials = df.drop_duplicates(subset=["pair_id", "session_id", "round", "trial"]).shape[0]
    print(f"[1] total rows = {n_rows}, total trials = {n_trials} (expect rows == 2 * trials)")
    assert n_rows == 2 * n_trials, "row count is not exactly 2x trial count for long form"
    report["n_rows"] = n_rows
    report["n_trials"] = n_trials

    # 2. Trials per dyad
    per_dyad_trials = (
        df.drop_duplicates(subset=["pair_id", "session_id", "round", "trial"])
        .groupby("pair_id")
        .size()
        .sort_index()
    )
    print("[2] trials per dyad:")
    print(per_dyad_trials.to_string())
    median = per_dyad_trials.median()
    flagged = per_dyad_trials[
        (per_dyad_trials < median * 0.75) | (per_dyad_trials > median * 1.25)
    ]
    if len(flagged):
        print(f"    FLAGGED dyads outside +/-25% of median ({median}):")
        print(flagged.to_string())
    else:
        print(f"    none flagged (median = {median})")
    report["per_dyad_trials"] = per_dyad_trials.to_dict()

    # 3. Exactly 12 dyads, 24 participants, each in one dyad, partner_id symmetric
    n_dyads = df["pair_id"].nunique()
    n_participants = df["participant_id"].nunique()
    print(f"[3] n_dyads = {n_dyads} (expect 12), n_participants = {n_participants} (expect 24)")
    part_dyads = df.groupby("participant_id")["pair_id"].nunique()
    bad_part = part_dyads[part_dyads != 1]
    if len(bad_part):
        print(f"    VIOLATION: participants in != 1 dyad:\n{bad_part.to_string()}")
    else:
        print("    each participant appears in exactly one dyad: OK")
    # partner symmetry
    pp = df[["participant_id", "partner_id"]].drop_duplicates()
    partner_map = dict(zip(pp["participant_id"], pp["partner_id"]))
    asym = [
        (p, q)
        for p, q in partner_map.items()
        if partner_map.get(q) != p
    ]
    if asym:
        print(f"    VIOLATION: asymmetric partner pairs: {asym}")
    else:
        print("    partner_id symmetric for all participants: OK")
    report["n_dyads"] = n_dyads
    report["n_participants"] = n_participants

    # 4. No duplicate (pair_id, round, role) -- NOTE: extended with session_id
    # since a dyad can have 2 sessions and round numbers repeat across them;
    # (pair_id, round, role) alone is expected to collide for 2-session
    # dyads by design, so we check the true grain key
    # (pair_id, session_id, round, trial, role) here and report both.
    dup_key_plan = df.duplicated(subset=["pair_id", "round", "role"]).sum()
    dup_key_true = df.duplicated(subset=["pair_id", "session_id", "round", "trial", "role"]).sum()
    print(
        f"[4] duplicate (pair_id, round, role) rows = {dup_key_plan} "
        f"(non-zero is EXPECTED for the 11 dyads with two sessions, since "
        f"round numbers restart per session -- see module docstring); "
        f"duplicate (pair_id, session_id, round, trial, role) rows = {dup_key_true} (must be 0)"
    )
    assert dup_key_true == 0, "duplicate true-grain keys found -- grain assumption is wrong"
    report["dup_key_true_grain"] = int(dup_key_true)
    report["dup_key_plan_literal"] = int(dup_key_plan)

    # 5. Rounds strictly monotonically increasing within each dyad after sort
    print("[5] round monotonicity within each dyad's session-ordered sequence:")
    mono_fail = []
    for pair_id, grp in df.sort_values(["pair_id", "session_order", "round", "trial"]).groupby(
        "pair_id"
    ):
        # dyad_trial_seq must be strictly increasing by 1 across the two
        # rows of each trial repeated, so check on trial grain instead
        tg = grp.drop_duplicates(subset=["session_id", "round", "trial"])
        seq = tg["dyad_trial_seq"].tolist()
        if seq != sorted(seq) or seq != list(range(1, len(seq) + 1)):
            mono_fail.append(pair_id)
    if mono_fail:
        print(f"    FAIL for dyads: {mono_fail}")
    else:
        print("    all dyads OK (dyad_trial_seq is 1..N with no gaps, in sorted order)")
    report["round_monotonicity_failures"] = mono_fail

    # 6. No missing pair_id/participant_id/partner_id/role/condition; null
    # counts for every column
    print("[6] null counts per column:")
    nulls = df.isnull().sum()
    print(nulls.to_string())
    required = ["pair_id", "participant_id", "partner_id", "role", "condition"]
    bad_required = {c: int(nulls[c]) for c in required if nulls[c] > 0}
    if bad_required:
        print(f"    VIOLATION: required columns have nulls: {bad_required}")
    else:
        print("    all required columns fully populated: OK")
    report["null_counts"] = nulls.to_dict()

    # 7. Condition class balance per dyad (truth vs deception)
    print("[7] condition class balance per dyad (trial grain, deceiver row only):")
    deceiver_rows = df[df["role"] == "deceiver"]
    balance = deceiver_rows.groupby(["pair_id", "condition"]).size().unstack(fill_value=0)
    print(balance.to_string())
    report["condition_balance_per_dyad"] = balance.to_dict()

    # 8. History-field leakage check (recompute on one dyad)
    sample_pair = df["pair_id"].iloc[0]
    sample = (
        df[(df["pair_id"] == sample_pair) & (df["role"] == "deceiver")]
        .sort_values(["session_order", "round", "trial"])
        .reset_index(drop=True)
    )
    ok = True
    running_lie = 0
    for i, row in sample.iterrows():
        if row["trials_so_far"] != i:
            ok = False
            print(f"    MISMATCH at row {i}: trials_so_far={row['trials_so_far']} expected {i}")
        if row["prior_deception_count"] != running_lie:
            ok = False
            print(
                f"    MISMATCH at row {i}: prior_deception_count={row['prior_deception_count']} "
                f"expected {running_lie}"
            )
        if row["condition"] == "lie":
            running_lie += 1
    print(f"[8] history leakage check on dyad {sample_pair}: {'OK' if ok else 'FAILED'}")
    assert ok, f"history leakage check failed for dyad {sample_pair}"
    report["history_leakage_check_pair"] = sample_pair
    report["history_leakage_check_ok"] = ok

    # 9. Round-trip check
    write_trial_table(df)
    df2 = read_trial_table()
    same_len = len(df2) == len(df)
    same_order = list(df2["dyad_trial_seq"]) == list(df["dyad_trial_seq"]) and list(
        df2["pair_id"]
    ) == list(df["pair_id"])
    print(f"[9] round-trip check: same row count = {same_len}, same sort order = {same_order}")
    assert same_len and same_order, "round-trip check failed"
    report["round_trip_ok"] = same_len and same_order

    return report


def main() -> int:
    df = build_trial_table()
    write_trial_table(df)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")
    validate_trial_table(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
