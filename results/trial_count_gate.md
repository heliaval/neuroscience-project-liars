# Trial-Count Gate (§7) — Go/No-Go

Generated at: 2026-08-20 09:07
Source: `data\processed\trial_table.csv` (22264 rows)
Spec sections: §7 (gate design), §12-§18 (experiment designs), §20 (paired test), §22 (CIs), §42 (limitations)

**This file is written in two passes.** This is pass 1: the threshold section
below is final and was written before any per-experiment fold size was compared
against it. The verdict table at the bottom is a placeholder (`NOT YET COMPUTED`)
until pass 2.

---

## Threshold and rationale

Anchored in the smallest test fold's ability to support an informative AUROC
confidence interval (§22 makes the CI the headline for demoted experiments).

- Formula: Hanley-McNeil (1982) SE(AUC) = sqrt[AUC(1-AUC) + (n1-1)(Q1-AUC^2) + (n2-1)(Q2-AUC^2)] / sqrt(n1*n2); Q1=AUC/(2-AUC), Q2=2*AUC^2/(1+AUC)
- Assumed effect size: AUC = 0.65 (conservative end of the
  0.65-0.70 range this literature lives in; the conservative anchor produces the
  more defensible, harder-to-clear threshold)
- Assumed class balance: balanced (n1=n2=m), consistent with the pooled minority fraction ~0.48 observed in this dataset (2d)
- Target: 95% CI half-width <= 0.1
- Solved: at m = 60, SE = 0.05, half-width = 0.098

Selected values from the solve table (m = minority-class trial count, both classes
assumed equal in size):

| assumed AUC | m | SE | 95% CI half-width |
|---|---|---|---|
| 0.65 | 25 | 0.0779 | 0.1527 |
| 0.65 | 30 | 0.071 | 0.1392 |
| 0.65 | 50 | 0.0548 | 0.1074 |
| 0.65 | 55 | 0.0522 | 0.1024 |
| 0.65 | 58 | 0.0508 | 0.0997 |
| 0.65 | 60 | 0.05 | 0.098 |
| 0.65 | 65 | 0.048 | 0.0941 |
| 0.65 | 70 | 0.0462 | 0.0906 |
| 0.7 | 25 | 0.0743 | 0.1456 |
| 0.7 | 30 | 0.0677 | 0.1327 |
| 0.7 | 50 | 0.0522 | 0.1024 |
| 0.7 | 55 | 0.0498 | 0.0976 |
| 0.7 | 58 | 0.0485 | 0.095 |
| 0.7 | 60 | 0.0476 | 0.0934 |
| 0.7 | 65 | 0.0458 | 0.0897 |
| 0.7 | 70 | 0.0441 | 0.0864 |

**THRESHOLD_M = 60** minority-class trials in the smallest
test fold.

Why not a lower threshold: The plan's a-priori expected landing zone (m~25-30) yields a half-width of ~0.14-0.15 at AUC=0.65-0.70 -- more than 40% wider than the +/-0.10 target, wide enough that a demoted 'exploratory' experiment's CI would look no worse than a genuinely underpowered one. m=60 was used instead because it is what the Hanley-McNeil arithmetic actually solves to at the stated assumptions; rounding down to the expected figure would have fitted the threshold to a prior guess rather than to the target CI precision.

### Clause A (fold size)

smallest test fold has >= 60 trials in its minority class.

### Clause B (paired-test coverage)

at least 10 of 12 dyads (or the equivalent ~83% fraction at reduced n) individually satisfy Clause A.

Because the primary metric (§20) is a paired test over dyads, an experiment where
Clause A holds only on average but fails for several dyads cannot produce enough
usable paired differences — the sign test's minimum attainable p-value degrades
sharply as n drops (see the sub01_sub02 section below for the actual numbers at
n=12 vs n=11). 10 of 12 (a "no more than 2 dyads may fail
Clause A" standard) is a judgment call, not a derived number, and is stated as one.

---

## Count tables

### 2a. Trials per participant

Degenerate with 2b: both participants take part in every trial of their dyad, so
this equals their dyad's trial count (see 2b below; not duplicated here as a
separate table).

### 2b. Trials per dyad

| pair_id | n_trials | n_sessions | min_seq | max_seq |
|---|---|---|---|---|
| sub01_sub02 | 484 | 1 | 1 | 484 |
| sub03_sub06 | 968 | 2 | 1 | 968 |
| sub04_sub05 | 968 | 2 | 1 | 968 |
| sub07_sub08 | 968 | 2 | 1 | 968 |
| sub09_sub10 | 968 | 2 | 1 | 968 |
| sub11_sub12 | 968 | 2 | 1 | 968 |
| sub13_sub14 | 968 | 2 | 1 | 968 |
| sub15_sub16 | 968 | 2 | 1 | 968 |
| sub17_sub18 | 968 | 2 | 1 | 968 |
| sub19_sub22 | 968 | 2 | 1 | 968 |
| sub20_sub21 | 968 | 2 | 1 | 968 |
| sub23_sub24 | 968 | 2 | 1 | 968 |

### 2c. Trials per role per participant

sub02's deceiver_trials = 0 is the numerical signature of the data-quality issue
documented in PROGRESS.md and revisited in the sub01_sub02 decision below.

| participant_id | deceiver_trials | observer_trials |
|---|---|---|
| sub01 | 484 | 0 |
| sub02 | 0 | 484 |
| sub03 | 484 | 484 |
| sub04 | 484 | 484 |
| sub05 | 484 | 484 |
| sub06 | 484 | 484 |
| sub07 | 484 | 484 |
| sub08 | 484 | 484 |
| sub09 | 484 | 484 |
| sub10 | 484 | 484 |
| sub11 | 484 | 484 |
| sub12 | 484 | 484 |
| sub13 | 484 | 484 |
| sub14 | 484 | 484 |
| sub15 | 484 | 484 |
| sub16 | 484 | 484 |
| sub17 | 484 | 484 |
| sub18 | 484 | 484 |
| sub19 | 484 | 484 |
| sub20 | 484 | 484 |
| sub21 | 484 | 484 |
| sub22 | 484 | 484 |
| sub23 | 484 | 484 |
| sub24 | 484 | 484 |

### 2d. Trials per condition per dyad, with class balance

Pooled across all 12 dyads: n_lie = 5338,
n_truth = 5794, minority_class_fraction =
0.4795.

| pair_id | n_lie | n_truth | n_total | minority_class_fraction |
|---|---|---|---|---|
| sub01_sub02 | 264 | 220 | 484 | 0.4545 |
| sub03_sub06 | 460 | 508 | 968 | 0.4752 |
| sub04_sub05 | 442 | 526 | 968 | 0.4566 |
| sub07_sub08 | 447 | 521 | 968 | 0.4618 |
| sub09_sub10 | 471 | 497 | 968 | 0.4866 |
| sub11_sub12 | 424 | 544 | 968 | 0.438 |
| sub13_sub14 | 459 | 509 | 968 | 0.4742 |
| sub15_sub16 | 457 | 511 | 968 | 0.4721 |
| sub17_sub18 | 399 | 569 | 968 | 0.4122 |
| sub19_sub22 | 526 | 442 | 968 | 0.4566 |
| sub20_sub21 | 489 | 479 | 968 | 0.4948 |
| sub23_sub24 | 500 | 468 | 968 | 0.4835 |

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

| pair_id | bin | n_trials | n_lie | n_truth |
|---|---|---|---|---|
| sub01_sub02 | early | 162 | 79 | 83 |
| sub01_sub02 | middle | 161 | 94 | 67 |
| sub01_sub02 | late | 161 | 91 | 70 |
| sub03_sub06 | early | 323 | 148 | 175 |
| sub03_sub06 | middle | 323 | 148 | 175 |
| sub03_sub06 | late | 322 | 164 | 158 |
| sub04_sub05 | early | 323 | 156 | 167 |
| sub04_sub05 | middle | 323 | 145 | 178 |
| sub04_sub05 | late | 322 | 141 | 181 |
| sub07_sub08 | early | 323 | 146 | 177 |
| sub07_sub08 | middle | 323 | 139 | 184 |
| sub07_sub08 | late | 322 | 162 | 160 |
| sub09_sub10 | early | 323 | 162 | 161 |
| sub09_sub10 | middle | 323 | 173 | 150 |
| sub09_sub10 | late | 322 | 136 | 186 |
| sub11_sub12 | early | 323 | 147 | 176 |
| sub11_sub12 | middle | 323 | 136 | 187 |
| sub11_sub12 | late | 322 | 141 | 181 |
| sub13_sub14 | early | 323 | 154 | 169 |
| sub13_sub14 | middle | 323 | 155 | 168 |
| sub13_sub14 | late | 322 | 150 | 172 |
| sub15_sub16 | early | 323 | 140 | 183 |
| sub15_sub16 | middle | 323 | 147 | 176 |
| sub15_sub16 | late | 322 | 170 | 152 |
| sub17_sub18 | early | 323 | 130 | 193 |
| sub17_sub18 | middle | 323 | 135 | 188 |
| sub17_sub18 | late | 322 | 134 | 188 |
| sub19_sub22 | early | 323 | 167 | 156 |
| sub19_sub22 | middle | 323 | 190 | 133 |
| sub19_sub22 | late | 322 | 169 | 153 |
| sub20_sub21 | early | 323 | 161 | 162 |
| sub20_sub21 | middle | 323 | 178 | 145 |
| sub20_sub21 | late | 322 | 150 | 172 |
| sub23_sub24 | early | 323 | 150 | 173 |
| sub23_sub24 | middle | 323 | 181 | 142 |
| sub23_sub24 | late | 322 | 169 | 153 |

### 2f. Smallest test fold implied by each experiment

See the per-experiment assumptions log at the bottom of this file (written in
pass 2, alongside the verdicts) — each fold-boundary assumption that is not pinned
down by the spec is written out there, next to the number it produced.

---

## Per-experiment gate table

| Experiment | Smallest fold total | Smallest fold minority | Dyads passing Clause A | Clause A | Clause B | Verdict | Reason |
|---|---|---|---|---|---|---|---|
| Exp 1 — Pooled baseline | 11132 | 5338 | N/A | N/A | N/A | N/A (not gated) | Pooled baseline / sanity check, explicitly excluded from the §7 gate's scope (Experiments 2-8 only). |
| Exp 2 — Universal (LODO) | 484 | 220 | 12/12 | True | True | CONFIRMATORY | Smallest fold minority (220) clears threshold (60); 12/12 included dyads individually clear it (needed >= 10). |
| Exp 3 — Personalized | 161 | 70 | 11/11 | True | True | CONFIRMATORY | Smallest fold minority (70) clears threshold (60); 11/11 included dyads individually clear it (needed >= 10). |
| Exp 4 — Dyad-specific | 161 | 70 | 11/11 | True | True | CONFIRMATORY | Smallest fold minority (70) clears threshold (60); 11/11 included dyads individually clear it (needed >= 10). |
| Exp 5 — Interaction history | 161 | 70 | 11/11 | True | True | CONFIRMATORY | Smallest fold minority (70) clears threshold (60); 11/11 included dyads individually clear it (needed >= 10). |
| Exp 6 — Observer-only | 161 | 70 | 11/11 | True | True | CONFIRMATORY | Smallest fold minority (70) clears threshold (60); 11/11 included dyads individually clear it (needed >= 10). |
| Exp 7 — One brain vs two | 161 | 70 | 11/11 | True | True | CONFIRMATORY | Smallest fold minority (70) clears threshold (60); 11/11 included dyads individually clear it (needed >= 10). |
| Exp 8 — Information onset | 484 | 220 | 11/11 | True | True | CONFIRMATORY | Smallest fold minority (220) clears threshold (60); 11/11 included dyads individually clear it (needed >= 10). |

---

## sub01_sub02 decision

Dyad `sub01_sub02` has only 484 trials (one session, sub01 always deceiver)
against 968 for the other eleven dyads. The reverse-role session is missing from
the archive, so sub02 never appears as deceiver anywhere in this dataset.

### Options considered

| Option | Label | Note |
|---|---|---|
| a | Keep in all experiments | Within-dyad experiments get one dyad at half the fold size, and sub02 has zero deceiver trials, so any deceiver-role or role-comparison analysis is undefined (not merely small) for that participant. |
| b | Keep for population-level (Exp2, LODO); exclude from experiments requiring both roles per dyad | CHOSEN —  |
| c | Exclude entirely | Cleanest n, but costs a twelfth of the population-level sample for no power gain, since Exp2's fold for this dyad clears the threshold on its own. |

### Chosen: option b

sub01_sub02's 484-trial LODO fold clears the go/no-go threshold on its own (minority class 220 >= 60), so dropping it from Exp2 would cost population-level sample size for no power benefit. But Experiments 3-8 either require deceiver-role history for sub02 (which does not exist -- sub02 was never recorded as deceiver, because the reverse-role session is missing from the archive, not because of anything about the participant) or compare the two roles against each other within the same dyad (Exp6, Exp7, Exp8's T_deceiver vs T_observer), which is undefined when one role has zero trials. Even where a one-directional computation is technically possible (e.g. Exp4's Person-Specific level for sub01-as-deceiver), including it would make this dyad's paired difference structurally asymmetric relative to the other 11 dyads, which all have bidirectional (role-swapped) history -- muddying the interpretation of a 'per-dyad' comparison that assumes each dyad contributed a comparable measurement. Excluding it from Exp3-8 keeps those 11 dyads homogeneous.

### Which experiments run at which n

- n = 12 (sub01_sub02 kept): Exp 1 — Pooled baseline, Exp 2 — Universal (LODO)
- n = 11 (sub01_sub02 excluded): Exp 3 — Personalized, Exp 4 — Dyad-specific, Exp 5 — Interaction history, Exp 6 — Observer-only, Exp 7 — One brain vs two, Exp 8 — Information onset

### Sign-test minimum attainable p-value

- At n = 12 (all differences the same sign, two-sided): p = 0.000488
- At n = 11 (all differences the same sign, two-sided): p = 0.000977

### Does sub01_sub02's inclusion demote Exp2?

No — verified, not assumed. Its LODO fold has 484 total
trials and 220 in the minority class, which clears
THRESHOLD_M = 60 on its own (220 >= 60).

### Limitations note

sub02 is missing a deceiver session because of an archive gap (the reverse-role session for this dyad was not present in the downloaded behavioral log archive), not because of anything about the participant. This is a data-availability exclusion and belongs in the limitations (§42), not the results narrative.

---

## Assumptions log

| Experiment | Assumption |
|---|---|
| Exp 3 — Personalized | The spec's own example ('train rounds 1-30, test rounds 31-40') does not map onto this dataset's 'round' field, which only spans 1-11 per session (44 trials/round) -- not the 30+ rounds the example implies. Standing in for the held-out block: the late tercile of the dyad's own dyad_trial_seq (2e), i.e. train = early+middle, test = late. This is stated as an assumption, not a fact pinned down by the spec. |
| Exp 4 — Dyad-specific | Same late-tercile stand-in as Exp3, at dyad grain rather than participant grain. |
| Exp 5 — Interaction history | The spec's example prefix ladder (rounds 1-5/1-10/1-20/1-30) assumes a 'round' granularity this dataset does not have (see Exp3's assumption). Translated onto dyad_trial_seq: the ladder is collapsed to the tercile split already defined in 2e -- train on the early tercile only, then early+middle, always testing on the late tercile. This yields a 2-point learning curve per dyad (early-only vs early+middle training) rather than the spec's literal 4-point ladder, and its smallest test fold is the same late tercile used by Exp3/Exp4. |
| Exp 6 — Observer-only | Uses the same tercile split as 2e; smallest tercile is the binding constraint (late tercile is smallest or tied-smallest for every dyad here). |
| Exp 7 — One brain vs two | Exp7 does not define its own fold structure; it evaluates 5 input sets (deceiver/observer/both/inter-brain/EEG+behavioral) on identical folds inherited from Exp2 (LODO) or Exp4 (within-dyad), depending on which comparison is being made. Both inherited fold sizes are reported; the smaller of the two is the binding one. |
| Exp 8 — Information onset | Assumes an LODO-style fold (same as Exp2) per window per role, since §18 does not itself specify a within-dyad chronological split -- window position, not train/test boundary, is what varies. Fold size is therefore the same as Exp2's; the risk in Exp8 is the multiple-comparison burden (12 tests per claim family), not fold size, and is called out separately in the gate table -- a correction (Bonferroni or Benjamini-Hochberg) is required regardless of the Clause A/B verdict. |

---

_Pass 2 written at: 2026-08-20 09:08_
