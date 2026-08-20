"""
src/gate.py — Phase 1b/1c: trial-count gate (§7) and hypothesis freeze (§8) for
"Can Your Brain Learn a Liar?"

Reads data/processed/trial_table.csv, computes the six §7 count families, applies
a pre-registered go/no-go threshold to Experiments 2-8, resolves the sub01_sub02
data-availability shortfall, and writes:

    results/trial_count_gate.md
    results/gate.json
    results/frozen_hypotheses.md

Deterministic, no network, no modeling, no EEG.

--------------------------------------------------------------------------------
THRESHOLD (fixed here, in the module docstring and the THRESHOLD constant below,
BEFORE this module ever reads the trial table). This is the anti-backfill
mechanism required by the plan: the number below was derived from a standard
statistical formula and a stated assumption about plausible effect size, not from
looking at this dataset's fold sizes.

Formula: Hanley-McNeil (1982) standard error of an AUROC estimate,

    SE(AUC) = sqrt[ AUC(1-AUC) + (n1-1)(Q1-AUC^2) + (n2-1)(Q2-AUC^2) ] / sqrt(n1*n2)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC^2 / (1 + AUC)

applied at n1 = n2 = m (balanced classes — the realistic case here; see §2d, pooled
minority fraction is ~0.48), for an assumed plausible effect size AUC = 0.65 (the
conservative end of the 0.65-0.70 range the literature on interpersonal/interbrain
deception EEG typically reports; 0.65 is used rather than 0.70 because the
conservative anchor produces the more defensible (larger, harder-to-clear)
threshold, consistent with "do not reach for a threshold that lets everything
pass").

Solving for the minority-class count m at which the resulting 95% CI half-width
(1.96 * SE) falls at or below ~0.10 AUROC:

    m = 50   half-width = 0.1074   (fails; still above 0.10)
    m = 55   half-width = 0.1024   (fails)
    m = 60   half-width = 0.0980   (clears 0.10)

m = 55 and 60 bracket the crossing point; 60 is chosen as the threshold because it
is the first round number below which the interval degrades past the 0.10 target,
giving a small safety margin (0.098 vs the 0.10 target) without inflating the bar
past what the target actually demands.

THIS IS NOT THE PLAN'S "EXPECTED LANDING ZONE" (~25-30). That figure assumed a
smaller m would suffice; the Hanley-McNeil arithmetic at AUC=0.65-0.70 and n1=n2=m
does not support it — at m=25-30 the half-width is ~0.14-0.15, more than 40% wider
than the ±0.10 target and not tight enough to distinguish AUC=0.65 from chance with
any confidence. Per the plan's own instruction ("if your arithmetic lands
elsewhere, use your arithmetic and show it — do not round to a familiar number
because it looks tidy"), the arithmetic result (m=60) is used instead of the
expected figure. See results/trial_count_gate.md for the full table this was
solved from.

Why not a lower threshold: a lower m produces a CI wide enough (>0.10 half-width)
that a demoted, "exploratory" experiment and a genuine null result would be
visually and numerically indistinguishable from an experiment that actually could
support a claim — defeating the purpose of the gate.

CLAUSES (both required for CONFIRMATORY):
  Clause A (fold size): the experiment's smallest test fold has >= THRESHOLD_M
    trials in its minority class.
  Clause B (paired-test coverage): at least 10 of the dyads included in that
    experiment individually satisfy Clause A. (§20's paired test needs 12, or
    fewer after the §7-Step-5 sub01_sub02 exclusion, usable per-dyad differences;
    an experiment where Clause A holds only on average but fails for several
    dyads cannot produce enough usable paired differences, and the sign test's
    minimum attainable p-value degrades sharply as n drops.) This is a judgment
    call, not a derived number — 10/12 (or 9/11) was chosen as "at most 2 dyads
    may be dropped" rather than a stricter or looser fraction, because losing a
    sixth of the paired sample already meaningfully weakens the sign test, and
    the threshold should not be read as more precise than that.
--------------------------------------------------------------------------------
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TRIAL_TABLE_PATH = REPO_ROOT / "data" / "processed" / "trial_table.csv"
RESULTS_DIR = REPO_ROOT / "results"
GATE_MD_PATH = RESULTS_DIR / "trial_count_gate.md"
GATE_JSON_PATH = RESULTS_DIR / "gate.json"
FROZEN_MD_PATH = RESULTS_DIR / "frozen_hypotheses.md"

# ------------------------------------------------------------------------------
# THRESHOLD — module-level constant, defined above and independent of any
# function that reads the trial table. See module docstring for full derivation.
# ------------------------------------------------------------------------------
ASSUMED_AUC = 0.65
TARGET_HALF_WIDTH = 0.10
THRESHOLD_M = 60  # minority-class trials required in the smallest test fold
CLAUSE_B_MIN_DYADS_OF_12 = 10
CLAUSE_B_MIN_FRACTION = CLAUSE_B_MIN_DYADS_OF_12 / 12  # 0.8333...


def hanley_mcneil_se(auc: float, n1: int, n2: int) -> float:
    """Hanley-McNeil (1982) standard error of an AUROC estimate."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)) / (
        n1 * n2
    )
    return math.sqrt(var)


def threshold_solve_table():
    """Recompute the table the THRESHOLD_M constant was solved from, for the
    output files. Does not touch the trial table — pure arithmetic."""
    rows = []
    for auc in (0.65, 0.70):
        for m in (15, 20, 22, 24, 25, 26, 28, 30, 35, 40, 50, 55, 58, 60, 65, 70, 80, 90, 100):
            se = hanley_mcneil_se(auc, m, m)
            hw = 1.96 * se
            rows.append({"assumed_auc": auc, "m": m, "se": round(se, 4), "half_width": round(hw, 4)})
    return rows


def get_threshold_info():
    se_at_threshold = hanley_mcneil_se(ASSUMED_AUC, THRESHOLD_M, THRESHOLD_M)
    hw_at_threshold = 1.96 * se_at_threshold
    return {
        "threshold_m": THRESHOLD_M,
        "formula": "Hanley-McNeil (1982) SE(AUC) = sqrt[AUC(1-AUC) + (n1-1)(Q1-AUC^2) + (n2-1)(Q2-AUC^2)] / sqrt(n1*n2); Q1=AUC/(2-AUC), Q2=2*AUC^2/(1+AUC)",
        "assumed_auc": ASSUMED_AUC,
        "assumed_class_balance": "balanced (n1=n2=m), consistent with the pooled minority fraction ~0.48 observed in this dataset (2d)",
        "target_ci_half_width": TARGET_HALF_WIDTH,
        "solved_se": round(se_at_threshold, 4),
        "solved_half_width": round(hw_at_threshold, 4),
        "clause_a": f"smallest test fold has >= {THRESHOLD_M} trials in its minority class",
        "clause_b": f"at least {CLAUSE_B_MIN_DYADS_OF_12} of 12 dyads (or the equivalent ~{CLAUSE_B_MIN_FRACTION:.0%} fraction at reduced n) individually satisfy Clause A",
        "rejected_lower_threshold_reason": (
            "The plan's a-priori expected landing zone (m~25-30) yields a half-width of "
            "~0.14-0.15 at AUC=0.65-0.70 -- more than 40% wider than the +/-0.10 target, "
            "wide enough that a demoted 'exploratory' experiment's CI would look no worse "
            "than a genuinely underpowered one. m=60 was used instead because it is what "
            "the Hanley-McNeil arithmetic actually solves to at the stated assumptions; "
            "rounding down to the expected figure would have fitted the threshold to a "
            "prior guess rather than to the target CI precision."
        ),
    }


# ------------------------------------------------------------------------------
# Step 1 — load and validate
# ------------------------------------------------------------------------------
def load_and_validate() -> pd.DataFrame:
    df = pd.read_csv(TRIAL_TABLE_PATH)

    errors = []
    if len(df) != 22264:
        errors.append(f"expected 22264 rows, got {len(df)}")

    trial_grain = df.drop_duplicates(["pair_id", "dyad_trial_seq"])
    if len(trial_grain) != 11132:
        errors.append(f"expected 11132 unique trials on (pair_id, dyad_trial_seq), got {len(trial_grain)}")

    if df["pair_id"].nunique() != 12:
        errors.append(f"expected 12 unique pair_id, got {df['pair_id'].nunique()}")
    if df["participant_id"].nunique() != 24:
        errors.append(f"expected 24 unique participant_id, got {df['participant_id'].nunique()}")

    role_counts_per_trial = df.groupby(["pair_id", "dyad_trial_seq"])["role"].apply(
        lambda s: sorted(s.tolist())
    )
    bad_trials = role_counts_per_trial[role_counts_per_trial.apply(lambda r: r != ["deceiver", "observer"])]
    if len(bad_trials) != 0:
        errors.append(f"{len(bad_trials)} trials do not have exactly one deceiver and one observer row")

    cond_values = set(df["condition"].unique())
    if cond_values != {"lie", "truth"}:
        errors.append(f"expected condition values {{'lie','truth'}}, got {cond_values}")

    if errors:
        msg = "Trial table validation FAILED (table may have been regenerated or this plan is stale):\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        print(msg, file=sys.stderr)
        raise SystemExit(1)

    print(f"Validation OK: {len(df)} rows, {len(trial_grain)} unique trials, "
          f"{df['pair_id'].nunique()} dyads, {df['participant_id'].nunique()} participants.")
    return df


# ------------------------------------------------------------------------------
# Step 2 — count families
# ------------------------------------------------------------------------------
def tercile_bin(seq: int, n: int) -> str:
    """Tercile of a dyad's own dyad_trial_seq range. Remainder from a
    non-divisible count is assigned to the earliest bins so the late bin is
    never inflated."""
    base = n // 3
    rem = n % 3
    b1 = base + (1 if rem > 0 else 0)
    b2 = base + (1 if rem > 1 else 0)
    if seq <= b1:
        return "early"
    elif seq <= b1 + b2:
        return "middle"
    else:
        return "late"


def compute_counts(df: pd.DataFrame) -> dict:
    trial_grain = df.drop_duplicates(["pair_id", "dyad_trial_seq"]).copy()

    # 2a. Trials per participant (degenerate with 2b: every participant takes
    # part in every trial of their own dyad, in one role or the other).
    trials_per_participant = (
        df.groupby("participant_id")["pair_id"]
        .apply(lambda s: trial_grain[trial_grain.pair_id == s.iloc[0]].shape[0])
        .to_dict()
    )

    # 2b. Trials per dyad, sessions per dyad, seq range.
    trials_per_dyad = trial_grain.groupby("pair_id").size().to_dict()
    sessions_per_dyad = df.groupby("pair_id")["session_id"].nunique().to_dict()
    seq_range_per_dyad = (
        trial_grain.groupby("pair_id")["dyad_trial_seq"].agg(["min", "max"]).to_dict("index")
    )

    # 2c. Trials per role per participant.
    role_counts = df.groupby(["participant_id", "role"]).size().unstack(fill_value=0)
    trials_per_role_per_participant = role_counts.to_dict("index")

    # 2d. Trials per condition per dyad, class balance.
    cond = trial_grain.groupby(["pair_id", "condition"]).size().unstack(fill_value=0)
    for c in ("lie", "truth"):
        if c not in cond.columns:
            cond[c] = 0
    cond["n_total"] = cond["lie"] + cond["truth"]
    cond["minority_class_fraction"] = cond[["lie", "truth"]].min(axis=1) / cond["n_total"]
    condition_balance_per_dyad = cond.to_dict("index")

    pooled_lie = int((trial_grain.condition == "lie").sum())
    pooled_truth = int((trial_grain.condition == "truth").sum())
    pooled_balance = {
        "n_lie": pooled_lie,
        "n_truth": pooled_truth,
        "n_total": pooled_lie + pooled_truth,
        "minority_class_fraction": min(pooled_lie, pooled_truth) / (pooled_lie + pooled_truth),
    }

    # 2e. Early/middle/late chronological (per-dyad tercile) splits.
    tercile_rows = []
    for pid, g in trial_grain.groupby("pair_id"):
        n = int(g["dyad_trial_seq"].max())
        g = g.copy()
        g["bin"] = g["dyad_trial_seq"].apply(lambda s: tercile_bin(int(s), n))
        for b in ("early", "middle", "late"):
            sub = g[g["bin"] == b]
            n_lie = int((sub.condition == "lie").sum())
            n_truth = int((sub.condition == "truth").sum())
            tercile_rows.append(
                {
                    "pair_id": pid,
                    "n_dyad": n,
                    "bin": b,
                    "n_trials": int(len(sub)),
                    "n_lie": n_lie,
                    "n_truth": n_truth,
                    "minority": min(n_lie, n_truth),
                }
            )

    return {
        "trials_per_participant": trials_per_participant,
        "trials_per_dyad": trials_per_dyad,
        "sessions_per_dyad": sessions_per_dyad,
        "seq_range_per_dyad": seq_range_per_dyad,
        "trials_per_role_per_participant": trials_per_role_per_participant,
        "condition_balance_per_dyad": condition_balance_per_dyad,
        "pooled_condition_balance": pooled_balance,
        "tercile_rows": tercile_rows,
        "pooled_trial_count": int(len(trial_grain)),
    }


def fold_2f(counts: dict) -> dict:
    """Smallest test fold implied by each experiment's design (2f)."""
    tercile_rows = counts["tercile_rows"]
    late_rows = [r for r in tercile_rows if r["bin"] == "late"]
    late_by_dyad = {r["pair_id"]: r for r in late_rows}

    smallest_dyad = min(counts["trials_per_dyad"], key=lambda k: counts["trials_per_dyad"][k])
    smallest_dyad_total = counts["trials_per_dyad"][smallest_dyad]
    smallest_dyad_minority = min(
        counts["condition_balance_per_dyad"][smallest_dyad]["lie"],
        counts["condition_balance_per_dyad"][smallest_dyad]["truth"],
    )

    smallest_late_dyad = min(late_by_dyad, key=lambda k: late_by_dyad[k]["n_trials"])
    smallest_late = late_by_dyad[smallest_late_dyad]

    folds = {
        "exp1": {
            "design": "Pooled baseline / sanity check, not gated",
            "smallest_fold_total": counts["pooled_trial_count"],
            "smallest_fold_minority": counts["pooled_condition_balance"]["n_lie"] if counts["pooled_condition_balance"]["n_lie"] < counts["pooled_condition_balance"]["n_truth"] else counts["pooled_condition_balance"]["n_truth"],
            "dyads_considered": late_by_dyad,
            "assumption": "Pooled across all trials; not subject to the go/no-go gate (§7 explicitly scopes the gate to Experiments 2-8). Reported for completeness only.",
        },
        "exp2": {
            "design": "Leave-One-Dyad-Out CV. Test fold = one full held-out dyad.",
            "smallest_fold_total": smallest_dyad_total,
            "smallest_fold_minority": smallest_dyad_minority,
            "smallest_fold_dyad": smallest_dyad,
            "per_dyad_minority": {
                pid: min(v["lie"], v["truth"]) for pid, v in counts["condition_balance_per_dyad"].items()
            },
            "assumption": "No assumption needed; fold structure is fully specified by the LODO design in §12.",
        },
        "exp3": {
            "design": "Personalized, within-participant chronological split. Test fold = later-rounds held-out block per participant.",
            "smallest_fold_total": smallest_late["n_trials"],
            "smallest_fold_minority": smallest_late["minority"],
            "smallest_fold_dyad": smallest_late_dyad,
            "per_dyad_minority": {r["pair_id"]: r["minority"] for r in late_rows},
            "assumption": (
                "The spec's own example ('train rounds 1-30, test rounds 31-40') does not "
                "map onto this dataset's 'round' field, which only spans 1-11 per session "
                "(44 trials/round) -- not the 30+ rounds the example implies. Standing in "
                "for the held-out block: the late tercile of the dyad's own dyad_trial_seq "
                "(2e), i.e. train = early+middle, test = late. This is stated as an "
                "assumption, not a fact pinned down by the spec."
            ),
        },
        "exp4": {
            "design": "Dyad-specific, three model levels on the same held-out trials within each dyad. Test fold = within-dyad held-out block.",
            "smallest_fold_total": smallest_late["n_trials"],
            "smallest_fold_minority": smallest_late["minority"],
            "smallest_fold_dyad": smallest_late_dyad,
            "per_dyad_minority": {r["pair_id"]: r["minority"] for r in late_rows},
            "assumption": "Same late-tercile stand-in as Exp3, at dyad grain rather than participant grain.",
        },
        "exp5": {
            "design": "Increasing-history learning curve. Smallest test fold = the block remaining after the largest training prefix.",
            "smallest_fold_total": smallest_late["n_trials"],
            "smallest_fold_minority": smallest_late["minority"],
            "smallest_fold_dyad": smallest_late_dyad,
            "per_dyad_minority": {r["pair_id"]: r["minority"] for r in late_rows},
            "assumption": (
                "The spec's example prefix ladder (rounds 1-5/1-10/1-20/1-30) assumes a "
                "'round' granularity this dataset does not have (see Exp3's assumption). "
                "Translated onto dyad_trial_seq: the ladder is collapsed to the tercile "
                "split already defined in 2e -- train on the early tercile only, then "
                "early+middle, always testing on the late tercile. This yields a 2-point "
                "learning curve per dyad (early-only vs early+middle training) rather than "
                "the spec's literal 4-point ladder, and its smallest test fold is the same "
                "late tercile used by Exp3/Exp4."
            ),
        },
        "exp6": {
            "design": "Observer-only, early/middle/late per dyad. Test fold = one tercile of one dyad.",
            "smallest_fold_total": smallest_late["n_trials"],
            "smallest_fold_minority": smallest_late["minority"],
            "smallest_fold_dyad": smallest_late_dyad,
            "per_dyad_minority": {r["pair_id"]: r["minority"] for r in late_rows},
            "assumption": "Uses the same tercile split as 2e; smallest tercile is the binding constraint (late tercile is smallest or tied-smallest for every dyad here).",
        },
        "exp7": {
            "design": "Five input sets on identical folds (§17). Inherits whichever experiment's fold it is layered on.",
            "smallest_fold_total": min(smallest_late["n_trials"], smallest_dyad_total),
            "smallest_fold_minority": min(smallest_late["minority"], smallest_dyad_minority),
            "smallest_fold_dyad": smallest_late_dyad,
            "per_dyad_minority": {r["pair_id"]: r["minority"] for r in late_rows},
            "assumption": (
                "Exp7 does not define its own fold structure; it evaluates 5 input sets "
                "(deceiver/observer/both/inter-brain/EEG+behavioral) on identical folds "
                "inherited from Exp2 (LODO) or Exp4 (within-dyad), depending on which "
                "comparison is being made. Both inherited fold sizes are reported; the "
                "smaller of the two is the binding one."
            ),
        },
        "exp8": {
            "design": "Sliding windows x role (§18). Fold size does not shrink with window count, but tests multiply (6 windows x 2 roles = 12 per claim family).",
            "smallest_fold_total": smallest_dyad_total,
            "smallest_fold_minority": smallest_dyad_minority,
            "smallest_fold_dyad": smallest_dyad,
            "n_tests_per_claim_family": 12,
            "per_dyad_minority": {
                pid: min(v["lie"], v["truth"]) for pid, v in counts["condition_balance_per_dyad"].items()
            },
            "assumption": (
                "Assumes an LODO-style fold (same as Exp2) per window per role, since §18 "
                "does not itself specify a within-dyad chronological split -- window "
                "position, not train/test boundary, is what varies. Fold size is therefore "
                "the same as Exp2's; the risk in Exp8 is the multiple-comparison burden "
                "(12 tests per claim family), not fold size, and is called out separately "
                "in the gate table -- a correction (Bonferroni or Benjamini-Hochberg) is "
                "required regardless of the Clause A/B verdict."
            ),
        },
    }
    return folds


# ------------------------------------------------------------------------------
# Step 4 — apply the gate
# ------------------------------------------------------------------------------
def apply_gate(folds: dict, included_dyads_by_exp: dict) -> dict:
    verdicts = {}
    for exp_id, fold in folds.items():
        if exp_id == "exp1":
            verdicts[exp_id] = {
                "smallest_fold_total": fold["smallest_fold_total"],
                "smallest_fold_minority": fold["smallest_fold_minority"],
                "clause_a": None,
                "clause_b": None,
                "n_dyads_passing_clause_a": None,
                "n_dyads_considered": None,
                "verdict": "N/A (not gated)",
                "reason": "Pooled baseline / sanity check, explicitly excluded from the §7 gate's scope (Experiments 2-8 only).",
            }
            continue

        per_dyad_minority = fold.get("per_dyad_minority", {})
        included = included_dyads_by_exp.get(exp_id, list(per_dyad_minority.keys()))
        considered = {k: v for k, v in per_dyad_minority.items() if k in included}
        n_considered = len(considered)
        n_passing = sum(1 for v in considered.values() if v >= THRESHOLD_M)

        clause_a = fold["smallest_fold_minority"] >= THRESHOLD_M
        # Clause B: proportional 10/12 standard applied to the included-dyad count.
        min_needed = math.ceil(CLAUSE_B_MIN_FRACTION * n_considered) if n_considered else 0
        clause_b = n_passing >= min_needed

        verdict = "CONFIRMATORY" if (clause_a and clause_b) else "EXPLORATORY"
        if clause_a and clause_b:
            reason = f"Smallest fold minority ({fold['smallest_fold_minority']}) clears threshold ({THRESHOLD_M}); {n_passing}/{n_considered} included dyads individually clear it (needed >= {min_needed})."
        elif not clause_a:
            reason = f"Smallest fold minority ({fold['smallest_fold_minority']}) is below threshold ({THRESHOLD_M})."
        else:
            reason = f"Smallest fold clears threshold on average, but only {n_passing}/{n_considered} included dyads individually clear it (needed >= {min_needed})."

        verdicts[exp_id] = {
            "smallest_fold_total": fold["smallest_fold_total"],
            "smallest_fold_minority": fold["smallest_fold_minority"],
            "clause_a": clause_a,
            "clause_b": clause_b,
            "n_dyads_passing_clause_a": n_passing,
            "n_dyads_considered": n_considered,
            "verdict": verdict,
            "reason": reason,
        }
    return verdicts


# ------------------------------------------------------------------------------
# Step 5 — sub01_sub02 decision
# ------------------------------------------------------------------------------
SUB0102 = "sub01_sub02"

# Experiments that are role-agnostic / population-level and do not require
# role-symmetric (bidirectional) interaction history for a dyad:
EXPS_KEEPING_SUB0102 = {"exp1", "exp2"}
# All within-dyad, personalization, or role-comparison designs require both
# participants to have occupied both roles across the dyad's sessions, which
# sub01_sub02 structurally does not have (sub02 has zero deceiver trials --
# the reverse-role session is missing from the archive):
EXPS_EXCLUDING_SUB0102 = {"exp3", "exp4", "exp5", "exp6", "exp7", "exp8"}


def sub01_sub02_decision(counts: dict) -> dict:
    n12 = 12
    n11 = 11

    def min_two_sided_sign_test_p(n):
        # All n differences the same sign: two-sided p = 2 * 0.5^n, capped at 1.
        return min(1.0, 2 * (0.5**n))

    p_at_12 = min_two_sided_sign_test_p(n12)
    p_at_11 = min_two_sided_sign_test_p(n11)

    exp2_total = counts["trials_per_dyad"][SUB0102]
    exp2_minority = min(
        counts["condition_balance_per_dyad"][SUB0102]["lie"],
        counts["condition_balance_per_dyad"][SUB0102]["truth"],
    )
    exp2_clears_alone = exp2_minority >= THRESHOLD_M

    return {
        "options_considered": [
            {
                "option": "a",
                "label": "Keep in all experiments",
                "problem": (
                    "Within-dyad experiments get one dyad at half the fold size, and sub02 "
                    "has zero deceiver trials, so any deceiver-role or role-comparison "
                    "analysis is undefined (not merely small) for that participant."
                ),
            },
            {
                "option": "b",
                "label": "Keep for population-level (Exp2, LODO); exclude from experiments requiring both roles per dyad",
                "problem": None,
                "chosen": True,
            },
            {
                "option": "c",
                "label": "Exclude entirely",
                "problem": "Cleanest n, but costs a twelfth of the population-level sample for no power gain, since Exp2's fold for this dyad clears the threshold on its own.",
            },
        ],
        "chosen_option": "b",
        "reasoning": (
            "sub01_sub02's 484-trial LODO fold clears the go/no-go threshold on its own "
            f"(minority class {exp2_minority} >= {THRESHOLD_M}), so dropping it from Exp2 "
            "would cost population-level sample size for no power benefit. But Experiments "
            "3-8 either require deceiver-role history for sub02 (which does not exist -- "
            "sub02 was never recorded as deceiver, because the reverse-role session is "
            "missing from the archive, not because of anything about the participant) or "
            "compare the two roles against each other within the same dyad (Exp6, Exp7, "
            "Exp8's T_deceiver vs T_observer), which is undefined when one role has zero "
            "trials. Even where a one-directional computation is technically possible "
            "(e.g. Exp4's Person-Specific level for sub01-as-deceiver), including it would "
            "make this dyad's paired difference structurally asymmetric relative to the "
            "other 11 dyads, which all have bidirectional (role-swapped) history -- "
            "muddying the interpretation of a 'per-dyad' comparison that assumes each dyad "
            "contributed a comparable measurement. Excluding it from Exp3-8 keeps those 11 "
            "dyads homogeneous."
        ),
        "experiments_at_n12": sorted(EXPS_KEEPING_SUB0102),
        "experiments_at_n11": sorted(EXPS_EXCLUDING_SUB0102),
        "min_sign_test_p_at_n12": p_at_12,
        "min_sign_test_p_at_n11": p_at_11,
        "exp2_demoted_by_inclusion": not exp2_clears_alone,
        "exp2_fold_total": exp2_total,
        "exp2_fold_minority": exp2_minority,
        "limitations_note": (
            "sub02 is missing a deceiver session because of an archive gap (the reverse-"
            "role session for this dyad was not present in the downloaded behavioral log "
            "archive), not because of anything about the participant. This is a data-"
            "availability exclusion and belongs in the limitations (§42), not the results "
            "narrative."
        ),
    }


# ------------------------------------------------------------------------------
# Markdown / JSON writers
# ------------------------------------------------------------------------------
def fmt_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def write_gate_md_pass1(counts: dict, threshold_info: dict, generated_at: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    threshold_table = fmt_table(
        ["assumed AUC", "m", "SE", "95% CI half-width"],
        [
            (r["assumed_auc"], r["m"], r["se"], r["half_width"])
            for r in threshold_solve_table()
            if r["m"] in (25, 30, 50, 55, 58, 60, 65, 70)
        ],
    )

    trials_per_dyad_table = fmt_table(
        ["pair_id", "n_trials", "n_sessions", "min_seq", "max_seq"],
        [
            (
                pid,
                counts["trials_per_dyad"][pid],
                counts["sessions_per_dyad"][pid],
                counts["seq_range_per_dyad"][pid]["min"],
                counts["seq_range_per_dyad"][pid]["max"],
            )
            for pid in sorted(counts["trials_per_dyad"])
        ],
    )

    trials_per_role_table = fmt_table(
        ["participant_id", "deceiver_trials", "observer_trials"],
        [
            (pid, v.get("deceiver", 0), v.get("observer", 0))
            for pid, v in sorted(counts["trials_per_role_per_participant"].items())
        ],
    )

    condition_balance_table = fmt_table(
        ["pair_id", "n_lie", "n_truth", "n_total", "minority_class_fraction"],
        [
            (
                pid,
                v["lie"],
                v["truth"],
                v["n_total"],
                round(v["minority_class_fraction"], 4),
            )
            for pid, v in sorted(counts["condition_balance_per_dyad"].items())
        ],
    )

    tercile_table = fmt_table(
        ["pair_id", "bin", "n_trials", "n_lie", "n_truth"],
        [
            (r["pair_id"], r["bin"], r["n_trials"], r["n_lie"], r["n_truth"])
            for r in sorted(counts["tercile_rows"], key=lambda r: (r["pair_id"], ["early", "middle", "late"].index(r["bin"])))
        ],
    )

    verdict_table_placeholder = fmt_table(
        ["Exp", "Smallest fold total", "Smallest fold minority", "Dyads passing Clause A", "Clause A", "Clause B", "Verdict", "Reason"],
        [
            (exp, "NOT YET COMPUTED", "NOT YET COMPUTED", "NOT YET COMPUTED", "NOT YET COMPUTED", "NOT YET COMPUTED", "NOT YET COMPUTED", "NOT YET COMPUTED")
            for exp in ["1", "2", "3", "4", "5", "6", "7", "8"]
        ],
    )

    content = f"""# Trial-Count Gate (§7) — Go/No-Go

Generated at: {generated_at}
Source: `{TRIAL_TABLE_PATH.relative_to(REPO_ROOT)}` ({len(pd.read_csv(TRIAL_TABLE_PATH))} rows)
Spec sections: §7 (gate design), §12-§18 (experiment designs), §20 (paired test), §22 (CIs), §42 (limitations)

**This file is written in two passes.** This is pass 1: the threshold section
below is final and was written before any per-experiment fold size was compared
against it. The verdict table at the bottom is a placeholder (`NOT YET COMPUTED`)
until pass 2.

---

## Threshold and rationale

Anchored in the smallest test fold's ability to support an informative AUROC
confidence interval (§22 makes the CI the headline for demoted experiments).

- Formula: {threshold_info['formula']}
- Assumed effect size: AUC = {threshold_info['assumed_auc']} (conservative end of the
  0.65-0.70 range this literature lives in; the conservative anchor produces the
  more defensible, harder-to-clear threshold)
- Assumed class balance: {threshold_info['assumed_class_balance']}
- Target: 95% CI half-width <= {threshold_info['target_ci_half_width']}
- Solved: at m = {threshold_info['threshold_m']}, SE = {threshold_info['solved_se']}, half-width = {threshold_info['solved_half_width']}

Selected values from the solve table (m = minority-class trial count, both classes
assumed equal in size):

{threshold_table}

**THRESHOLD_M = {threshold_info['threshold_m']}** minority-class trials in the smallest
test fold.

Why not a lower threshold: {threshold_info['rejected_lower_threshold_reason']}

### Clause A (fold size)

{threshold_info['clause_a']}.

### Clause B (paired-test coverage)

{threshold_info['clause_b']}.

Because the primary metric (§20) is a paired test over dyads, an experiment where
Clause A holds only on average but fails for several dyads cannot produce enough
usable paired differences — the sign test's minimum attainable p-value degrades
sharply as n drops (see the sub01_sub02 section below for the actual numbers at
n=12 vs n=11). {CLAUSE_B_MIN_DYADS_OF_12} of 12 (a "no more than 2 dyads may fail
Clause A" standard) is a judgment call, not a derived number, and is stated as one.

---

## Count tables

### 2a. Trials per participant

Degenerate with 2b: both participants take part in every trial of their dyad, so
this equals their dyad's trial count (see 2b below; not duplicated here as a
separate table).

### 2b. Trials per dyad

{trials_per_dyad_table}

### 2c. Trials per role per participant

sub02's deceiver_trials = 0 is the numerical signature of the data-quality issue
documented in PROGRESS.md and revisited in the sub01_sub02 decision below.

{trials_per_role_table}

### 2d. Trials per condition per dyad, with class balance

Pooled across all 12 dyads: n_lie = {counts['pooled_condition_balance']['n_lie']},
n_truth = {counts['pooled_condition_balance']['n_truth']}, minority_class_fraction =
{round(counts['pooled_condition_balance']['minority_class_fraction'], 4)}.

{condition_balance_table}

### 2e. Early / middle / late chronological splits

**Definition:** Each dyad is split into terciles of its own `dyad_trial_seq` range,
using the dyad's own trial ordering rather than a global round number. Early =
first third, middle = second third, late = final third, with the remainder from a
non-divisible count assigned to the earliest bins so the late bin is never
inflated.

**Justification:**
(i) `round` resets per session, so a global round split would misalign dyads with
two sessions against the one dyad (sub01_sub02) with a single session;
(ii) per-dyad terciles keep the split proportional so a short dyad still
contributes all three bins rather than falling out of the late bin entirely;
(iii) equal-count terciles rather than equal-time terciles, because the gate is
about trial counts.

{tercile_table}

### 2f. Smallest test fold implied by each experiment

See the per-experiment assumptions log at the bottom of this file (written in
pass 2, alongside the verdicts) — each fold-boundary assumption that is not pinned
down by the spec is written out there, next to the number it produced.

---

## Per-experiment gate table (PLACEHOLDER — pass 2 fills this in)

{verdict_table_placeholder}

---

## sub01_sub02 decision

**NOT YET WRITTEN — pass 2.**

---

## Assumptions log

**NOT YET WRITTEN — pass 2.**
"""
    GATE_MD_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {GATE_MD_PATH} (pass 1: threshold fixed, verdicts NOT YET COMPUTED)")


def write_gate_md_pass2(counts, threshold_info, folds, verdicts, sub_decision, generated_at) -> None:
    existing = GATE_MD_PATH.read_text(encoding="utf-8")
    threshold_and_counts_section = existing.split("## Per-experiment gate table")[0]

    exp_names = {
        "exp1": "Exp 1 — Pooled baseline",
        "exp2": "Exp 2 — Universal (LODO)",
        "exp3": "Exp 3 — Personalized",
        "exp4": "Exp 4 — Dyad-specific",
        "exp5": "Exp 5 — Interaction history",
        "exp6": "Exp 6 — Observer-only",
        "exp7": "Exp 7 — One brain vs two",
        "exp8": "Exp 8 — Information onset",
    }

    verdict_rows = []
    for exp_id in ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8"]:
        v = verdicts[exp_id]
        verdict_rows.append(
            (
                exp_names[exp_id],
                v["smallest_fold_total"],
                v["smallest_fold_minority"],
                f"{v['n_dyads_passing_clause_a']}/{v['n_dyads_considered']}" if v["n_dyads_considered"] is not None else "N/A",
                v["clause_a"] if v["clause_a"] is not None else "N/A",
                v["clause_b"] if v["clause_b"] is not None else "N/A",
                v["verdict"],
                v["reason"],
            )
        )
    verdict_table = fmt_table(
        ["Experiment", "Smallest fold total", "Smallest fold minority", "Dyads passing Clause A", "Clause A", "Clause B", "Verdict", "Reason"],
        verdict_rows,
    )

    assumptions_rows = []
    for exp_id in ["exp3", "exp4", "exp5", "exp6", "exp7", "exp8"]:
        assumptions_rows.append((exp_names[exp_id], folds[exp_id]["assumption"]))
    assumptions_table = fmt_table(["Experiment", "Assumption"], assumptions_rows)

    sub_options_table = fmt_table(
        ["Option", "Label", "Note"],
        [
            (o["option"], o["label"], ("CHOSEN — " if o.get("chosen") else "") + (o["problem"] or ""))
            for o in sub_decision["options_considered"]
        ],
    )

    sub_section = f"""## sub01_sub02 decision

Dyad `sub01_sub02` has only 484 trials (one session, sub01 always deceiver)
against 968 for the other eleven dyads. The reverse-role session is missing from
the archive, so sub02 never appears as deceiver anywhere in this dataset.

### Options considered

{sub_options_table}

### Chosen: option {sub_decision['chosen_option']}

{sub_decision['reasoning']}

### Which experiments run at which n

- n = 12 (sub01_sub02 kept): {', '.join(exp_names[e] for e in sorted(sub_decision['experiments_at_n12']))}
- n = 11 (sub01_sub02 excluded): {', '.join(exp_names[e] for e in sorted(sub_decision['experiments_at_n11']))}

### Sign-test minimum attainable p-value

- At n = 12 (all differences the same sign, two-sided): p = {sub_decision['min_sign_test_p_at_n12']:.6f}
- At n = 11 (all differences the same sign, two-sided): p = {sub_decision['min_sign_test_p_at_n11']:.6f}

### Does sub01_sub02's inclusion demote Exp2?

No — verified, not assumed. Its LODO fold has {sub_decision['exp2_fold_total']} total
trials and {sub_decision['exp2_fold_minority']} in the minority class, which clears
THRESHOLD_M = {THRESHOLD_M} on its own ({sub_decision['exp2_fold_minority']} >= {THRESHOLD_M}).

### Limitations note

{sub_decision['limitations_note']}

---
"""

    content = (
        threshold_and_counts_section
        + "## Per-experiment gate table\n\n"
        + verdict_table
        + "\n\n---\n\n"
        + sub_section
        + "\n## Assumptions log\n\n"
        + assumptions_table
        + f"\n\n---\n\n_Pass 2 written at: {generated_at}_\n"
    )
    GATE_MD_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {GATE_MD_PATH} (pass 2: verdicts computed)")


def write_gate_json(counts, threshold_info, folds, verdicts, sub_decision, frozen_subtree, generated_at) -> None:
    gate_subtree = {
        "threshold": {
            "value": THRESHOLD_M,
            "formula": threshold_info["formula"],
            "assumed_auc": threshold_info["assumed_auc"],
            "assumed_class_balance": threshold_info["assumed_class_balance"],
            "target_ci_half_width": threshold_info["target_ci_half_width"],
            "solved_se": threshold_info["solved_se"],
            "solved_half_width": threshold_info["solved_half_width"],
            "clause_a": threshold_info["clause_a"],
            "clause_b": threshold_info["clause_b"],
            "rejected_lower_threshold_reason": threshold_info["rejected_lower_threshold_reason"],
        },
        "counts": {
            "trials_per_participant": counts["trials_per_participant"],
            "trials_per_dyad": counts["trials_per_dyad"],
            "sessions_per_dyad": counts["sessions_per_dyad"],
            "seq_range_per_dyad": counts["seq_range_per_dyad"],
            "trials_per_role_per_participant": {
                k: dict(v) for k, v in counts["trials_per_role_per_participant"].items()
            },
            "condition_balance_per_dyad": counts["condition_balance_per_dyad"],
            "pooled_condition_balance": counts["pooled_condition_balance"],
            "tercile_rows": counts["tercile_rows"],
        },
        "split_definition": {
            "method": "per-dyad tercile of dyad_trial_seq",
            "remainder_assignment": "earliest bins",
            "justification": [
                "round resets per session; a global round split would misalign dyads with two sessions against sub01_sub02's single session",
                "per-dyad terciles keep the split proportional so a short dyad still contributes all three bins",
                "equal-count terciles rather than equal-time terciles, because the gate is about trial counts",
            ],
        },
        "experiments": {
            exp_id: {**verdicts[exp_id], "smallest_fold_dyad": folds[exp_id].get("smallest_fold_dyad"), "assumption": folds[exp_id].get("assumption")}
            for exp_id in verdicts
        },
        "dyad_exclusions": {
            "sub01_sub02": {
                "chosen_option": sub_decision["chosen_option"],
                "reasoning": sub_decision["reasoning"],
                "experiments_at_n12": sub_decision["experiments_at_n12"],
                "experiments_at_n11": sub_decision["experiments_at_n11"],
                "min_sign_test_p_at_n12": sub_decision["min_sign_test_p_at_n12"],
                "min_sign_test_p_at_n11": sub_decision["min_sign_test_p_at_n11"],
                "exp2_demoted_by_inclusion": sub_decision["exp2_demoted_by_inclusion"],
                "limitations_note": sub_decision["limitations_note"],
            }
        },
        "assumptions": {exp_id: folds[exp_id]["assumption"] for exp_id in folds if exp_id != "exp1" and exp_id != "exp2"},
    }

    output = {
        "meta": {
            "generated_at": generated_at,
            "source_table": str(TRIAL_TABLE_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_row_count": int(len(pd.read_csv(TRIAL_TABLE_PATH))),
        },
        "gate": gate_subtree,
        "frozen": frozen_subtree,
    }
    GATE_JSON_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {GATE_JSON_PATH}")


def build_frozen_subtree(threshold_info, verdicts, sub_decision, generated_at):
    return {
        "hypotheses": {
            "H1": "Universal Deception Signature. Neural patterns associated with deception generalize across participants. If true, a model trained on most participants should predict deception in completely unseen participants.",
            "H2": "Person-Specific Deception Signature. Different people have different deception-related neural patterns. If true, models trained on prior data from the same person should outperform universal population models.",
            "H3": "Relationship-Specific Deception Signature (the project's main hypothesis). Repeated interaction causes neural responses to become specific to a particular pair. If true, models using prior interactions from the same dyad should outperform models trained only on population-level data.",
        },
        "primary_metric": {
            "unit_of_analysis": "dyad",
            "n": {"default": 12, "reduced": 11, "reduced_applies_to": sorted(EXPS_EXCLUDING_SUB0102)},
            "procedure": [
                "Compute the metric under both conditions for dyad d, on the same held-out trials.",
                "Take the difference Delta_d.",
                "Repeat for all included dyads (12, or 11 where sub01_sub02 is excluded).",
                "Test the resulting differences with a sign test and a paired permutation test (sign-flipping).",
                "Report the median Delta, the full distribution, the count of positive Delta, and the p-value.",
            ],
            "pooled_aggregate_note": "Reported as a descriptive summary only; it is not the test.",
            "nfi_definition": "NFI_d = Dyad-Specific_d - Population_d, reported as a distribution across dyads (median, spread, count above zero, paired-test p-value).",
        },
        "threshold": {
            "value": THRESHOLD_M,
            "formula": threshold_info["formula"],
            "assumed_auc": threshold_info["assumed_auc"],
        },
        "classification": {exp_id: verdicts[exp_id]["verdict"] for exp_id in verdicts},
        "null_result_reporting_plan": {
            "if_supported": (
                "Deception-related neural patterns generalized poorly across strangers but "
                "became more predictive when models incorporated information specific to "
                "the individual and dyad. The improvement was consistent across dyads, with "
                "[N] of 12 pairs showing positive dyad-specific gain. Neural responses in "
                "observers also became more informative across repeated interaction, "
                "suggesting that deception-related neural dynamics may partly depend on "
                "relationship history rather than representing a fully universal neural "
                "signature."
            ),
            "if_not_supported": (
                "Despite repeated interaction, dyad-specific history did not significantly "
                "improve prediction beyond participant-specific or population-level models. "
                "Across 12 dyads the paired differences were centered near zero, and the "
                "effect was not consistent in direction. This suggests that deception-"
                "related EEG patterns in this dataset are more strongly driven by "
                "individual or general neural characteristics than by relationship-specific "
                "adaptation."
            ),
            "reporting_rules": [
                "Whichever conclusion obtains is reported as written above.",
                "Exploratory experiments are reported with their confidence intervals and explicitly labeled underpowered (§7, §22).",
                "Anything decided after seeing results is labeled post-hoc (§8).",
                "Non-claims list carried by reference from §42.",
            ],
        },
        "frozen_at": generated_at,
        "amendments": [],
    }


def write_frozen_hypotheses(frozen_subtree, generated_at) -> None:
    h = frozen_subtree["hypotheses"]
    pm = frozen_subtree["primary_metric"]

    classification_table = fmt_table(
        ["Experiment", "Classification"],
        [(exp_id, verdict) for exp_id, verdict in frozen_subtree["classification"].items()],
    )

    content = f"""# Frozen Hypotheses and Primary Metric (§8)

Frozen at: {generated_at}. Written once, after the §7 gate (see
`results/trial_count_gate.md` and `results/gate.json`), before any model is fit.

---

## H1 — Universal Deception Signature

{h['H1']}

## H2 — Person-Specific Deception Signature

{h['H2']}

## H3 — Relationship-Specific Deception Signature (the project's main hypothesis)

{h['H3']}

---

## Primary metric

The §20 paired per-dyad test.

- Unit of analysis: dyad.
- n = {pm['n']['default']} for: {', '.join(sorted(EXPS_KEEPING_SUB0102))} (see the gate's sub01_sub02
  decision for why).
- n = {pm['n']['reduced']} for: {', '.join(pm['n']['reduced_applies_to'])} — sub01_sub02 excluded
  from these because it cannot supply role-symmetric (bidirectional) interaction
  history.
- Procedure:
{chr(10).join('  ' + str(i + 1) + '. ' + step for i, step in enumerate(pm['procedure']))}
- {pm['pooled_aggregate_note']}
- **Neural Familiarity Index (§24):** `{pm['nfi_definition']}`

---

## The go/no-go threshold and its rationale

Value: THRESHOLD_M = {frozen_subtree['threshold']['value']} minority-class trials in the
smallest test fold. Formula: {frozen_subtree['threshold']['formula']}. Assumed
effect size: AUC = {frozen_subtree['threshold']['assumed_auc']}. Full derivation and
rejected-alternatives reasoning: `results/trial_count_gate.md`, "Threshold and
rationale" section (copied there verbatim, not reproduced twice here).

---

## Confirmatory vs exploratory classification

{classification_table}

---

## Null-result reporting plan

**If relationship-specific learning is supported (§47):**

> {frozen_subtree['null_result_reporting_plan']['if_supported']}

**If relationship-specific learning is not supported (§48):**

> {frozen_subtree['null_result_reporting_plan']['if_not_supported']}

Both conclusions are written before results are seen. Whichever obtains is
reported as written above.

Reporting rules:
{chr(10).join('- ' + rule for rule in frozen_subtree['null_result_reporting_plan']['reporting_rules'])}

The full non-claims list is carried by reference from §42 of
`can_your_brain_learn_a_liar_workflow.md` (what this project does not claim: no
thought-reading, not a real-world lie detector, does not determine whether
arbitrary people are lying, no claims beyond the controlled experimental setting,
inter-brain synchrony is not proof of direct communication, and the 12-dyad
sample requires cautious interpretation).

---

## Amendment policy

> This file is frozen as of {generated_at}. It is not silently edited. Any change is
> appended below as a dated amendment with an explicit reason, leaving the original text
> in place. An amendment made after results were seen is labeled post-hoc.

## Amendments

(none)
"""
    FROZEN_MD_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {FROZEN_MD_PATH}")


# ------------------------------------------------------------------------------
# Main — phased so the threshold is written to disk before any fold size is
# compared against it (see plan step 3's enforced sequencing).
# ------------------------------------------------------------------------------
def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "full"

    df = load_and_validate()
    counts = compute_counts(df)
    threshold_info = get_threshold_info()

    if phase == "threshold":
        import datetime

        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        write_gate_md_pass1(counts, threshold_info, generated_at)
        print(f"\nTHRESHOLD WRITTEN: THRESHOLD_M = {THRESHOLD_M} (AUC={ASSUMED_AUC}, half-width={threshold_info['solved_half_width']})")
        print("Stop here. Append the PROGRESS.md mid-entry with a real timestamp before running `finish`.")
        return

    if phase not in ("finish", "full"):
        print(f"Unknown phase: {phase}. Use 'threshold', 'finish', or 'full'.", file=sys.stderr)
        raise SystemExit(1)

    if phase == "finish" and not GATE_MD_PATH.exists():
        print("results/trial_count_gate.md does not exist yet — run `threshold` phase first.", file=sys.stderr)
        raise SystemExit(1)

    import datetime

    if phase == "full":
        generated_at_pass1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        write_gate_md_pass1(counts, threshold_info, generated_at_pass1)

    folds = fold_2f(counts)
    verdicts = apply_gate(folds, {e: (list(counts["trials_per_dyad"].keys()) if e in EXPS_KEEPING_SUB0102 else [d for d in counts["trials_per_dyad"] if d != SUB0102]) for e in folds})
    sub_decision = sub01_sub02_decision(counts)

    generated_at_pass2 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    write_gate_md_pass2(counts, threshold_info, folds, verdicts, sub_decision, generated_at_pass2)

    frozen_subtree = build_frozen_subtree(threshold_info, verdicts, sub_decision, generated_at_pass2)
    write_gate_json(counts, threshold_info, folds, verdicts, sub_decision, frozen_subtree, generated_at_pass2)
    write_frozen_hypotheses(frozen_subtree, generated_at_pass2)

    print("\nDone.")
    for exp_id in ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8"]:
        v = verdicts[exp_id]
        print(f"  {exp_id}: {v['verdict']} (fold={v['smallest_fold_total']}, minority={v['smallest_fold_minority']})")


if __name__ == "__main__":
    main()
