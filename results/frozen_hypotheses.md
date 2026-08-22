# Frozen Hypotheses and Primary Metric (§8)

Frozen at: 2026-08-20 09:08. Written once, after the §7 gate (see
`results/trial_count_gate.md` and `results/gate.json`), before any model is fit.

---

## H1 — Universal Deception Signature

Universal Deception Signature. Neural patterns associated with deception generalize across participants. If true, a model trained on most participants should predict deception in completely unseen participants.

## H2 — Person-Specific Deception Signature

Person-Specific Deception Signature. Different people have different deception-related neural patterns. If true, models trained on prior data from the same person should outperform universal population models.

## H3 — Relationship-Specific Deception Signature (the project's main hypothesis)

Relationship-Specific Deception Signature (the project's main hypothesis). Repeated interaction causes neural responses to become specific to a particular pair. If true, models using prior interactions from the same dyad should outperform models trained only on population-level data.

---

## Primary metric

The §20 paired per-dyad test.

- Unit of analysis: dyad.
- n = 12 for: exp1, exp2 (see the gate's sub01_sub02
  decision for why).
- n = 11 for: exp3, exp4, exp5, exp6, exp7, exp8 — sub01_sub02 excluded
  from these because it cannot supply role-symmetric (bidirectional) interaction
  history.
- Procedure:
  1. Compute the metric under both conditions for dyad d, on the same held-out trials.
  2. Take the difference Delta_d.
  3. Repeat for all included dyads (12, or 11 where sub01_sub02 is excluded).
  4. Test the resulting differences with a sign test and a paired permutation test (sign-flipping).
  5. Report the median Delta, the full distribution, the count of positive Delta, and the p-value.
- Reported as a descriptive summary only; it is not the test.
- **Neural Familiarity Index (§24):** `NFI_d = Dyad-Specific_d - Population_d, reported as a distribution across dyads (median, spread, count above zero, paired-test p-value).`

---

## The go/no-go threshold and its rationale

Value: THRESHOLD_M = 60 minority-class trials in the
smallest test fold. Formula: Hanley-McNeil (1982) SE(AUC) = sqrt[AUC(1-AUC) + (n1-1)(Q1-AUC^2) + (n2-1)(Q2-AUC^2)] / sqrt(n1*n2); Q1=AUC/(2-AUC), Q2=2*AUC^2/(1+AUC). Assumed
effect size: AUC = 0.65. Full derivation and
rejected-alternatives reasoning: `results/trial_count_gate.md`, "Threshold and
rationale" section (copied there verbatim, not reproduced twice here).

---

## Confirmatory vs exploratory classification

| Experiment | Classification |
|---|---|
| exp1 | N/A (not gated) |
| exp2 | CONFIRMATORY |
| exp3 | CONFIRMATORY |
| exp4 | CONFIRMATORY |
| exp5 | CONFIRMATORY |
| exp6 | CONFIRMATORY |
| exp7 | CONFIRMATORY |
| exp8 | CONFIRMATORY |

---

## Null-result reporting plan

**If relationship-specific learning is supported (§47):**

> Deception-related neural patterns generalized poorly across strangers but became more predictive when models incorporated information specific to the individual and dyad. The improvement was consistent across dyads, with [N] of 12 pairs showing positive dyad-specific gain. Neural responses in observers also became more informative across repeated interaction, suggesting that deception-related neural dynamics may partly depend on relationship history rather than representing a fully universal neural signature.

**If relationship-specific learning is not supported (§48):**

> Despite repeated interaction, dyad-specific history did not significantly improve prediction beyond participant-specific or population-level models. Across 12 dyads the paired differences were centered near zero, and the effect was not consistent in direction. This suggests that deception-related EEG patterns in this dataset are more strongly driven by individual or general neural characteristics than by relationship-specific adaptation.

Both conclusions are written before results are seen. Whichever obtains is
reported as written above.

Reporting rules:
- Whichever conclusion obtains is reported as written above.
- Exploratory experiments are reported with their confidence intervals and explicitly labeled underpowered (§7, §22).
- Anything decided after seeing results is labeled post-hoc (§8).
- Non-claims list carried by reference from §42.

The full non-claims list is carried by reference from §42 of
`can_your_brain_learn_a_liar_workflow.md` (what this project does not claim: no
thought-reading, not a real-world lie detector, does not determine whether
arbitrary people are lying, no claims beyond the controlled experimental setting,
inter-brain synchrony is not proof of direct communication, and the 12-dyad
sample requires cautious interpretation).

---

## Amendment policy

> This file is frozen as of 2026-08-20 09:08. It is not silently edited. Any change is
> appended below as a dated amendment with an explicit reason, leaving the original text
> in place. An amendment made after results were seen is labeled post-hoc.

## Amendments

**Amendment 1 — 2026-08-22, pre-results, not post-hoc.** Experiment 4's unit-of-analysis
count is reduced from n = 11 to **n = 10**. `sub19_sub22` is excluded from exp4 as a
*unit of analysis* because it is structurally untestable at dyad grain: its only
participant with deceiver-role data (`sub19`; `sub22` was lost in §9 preprocessing) is
the session-1 deceiver, so the dyad's late tercile of `dyad_trial_seq` — exp4's held-out
block per `results/gate.json`'s frozen exp4 assumption — contains no deceiver trials at
all. This is a data-availability exclusion of the same kind as sub01_sub02's, determined
from data structure before any exp4 model was fit; no exp4 result had been computed when
this amendment was written. `sub19_sub22`'s deceiver rows remain in exp4's Universal
training pool. This amendment applies to exp4 only; exp5–exp8 are untouched and remain
at n = 11 pending their own analyses.

**Amendment 2 — 2026-08-22, pre-results, not post-hoc.**

1. Experiment 5's unit-of-analysis count drops from **n = 11 to n = 10**, excluding
   `sub19_sub22` for the same structural reason Amendment 1 gave for exp4: its only
   deceiver-role participant, `sub19`, is the session-1 deceiver, so the dyad's late
   tercile of `dyad_trial_seq` (exp5's held-out block, seq [647,968]) contains zero
   deceiver trials. `sub19_sub22`'s rows remain in exp5's other-dyad/Universal training
   pool. `sub01_sub02` remains excluded from exp3–exp8 as already stated above.

2. **Exp5-specific gate re-check**, computed at the finer grain §15's design actually
   requires (the frozen `gate.json` exp5 row was computed at exp3's participant-grain
   `sub01_sub02` numbers, which do not cover this design). `THRESHOLD_M = 60` and the
   Clause A ("smallest test-fold minority >= THRESHOLD_M") / Clause B ("at least 10 of
   12 dyads, or the equivalent ~83% fraction at reduced n — at n=10 that is >= 9")
   wording are taken **unchanged** from the original gate and are **not re-derived**,
   only re-applied at exp5's finer grain, against the ten dyads exp4's amendment already
   established as tested:

   | analysis | test-fold grain | Clause A | Clause B | designation |
   |---|---|---|---|---|
   | 1A — positional early/middle/late (dyad-grain terciles) | 322/324/322, min minority 130 | PASS | 10/10 PASS | confirmatory on power, interpretation confounded with participant identity, reported with that confound stated in the headline |
   | 1B — within-participant early/middle/late | 162/161/161, min minority 45 | **FAIL** | **8/10 FAIL** | **EXPLORATORY — underpowered**, reported with CIs and explicitly labelled, per §7 and §15 |
   | 2 — increasing-history learning curve | late tercile, 322, min minority 134 | PASS | 10/10 PASS | CONFIRMATORY at the two frozen rungs (k=322, k=646); the three additional rungs are pre-declared exploratory |
   | 3 — same-dyad vs other-dyad volume control | late tercile, 322, min minority 134 | PASS | 10/10 PASS | CONFIRMATORY at the k=646 headline rung; other rungs exploratory alongside Analysis 2's |

3. **Analysis 1B is demoted to exploratory / underpowered** by that check. The failing
   bins are `sub24` (45 minority trials, its middle within-participant bin — the same
   participant exp3 already flagged as its single sub-threshold case) and `sub17` (59
   minority trials, its middle within-participant bin). It is reported anyway, with
   confidence intervals and an explicit underpowered label, per §7's rule that a
   gate-failing analysis is demoted and reported, never silently dropped.

4. The frozen exp5 assumption's **2-point curve (k=322, k=646) remains the confirmatory
   design**. Three further ladder rungs (k=81, 162, 484) and Analysis 1 (both grains) are
   added as **pre-declared exploratory** extensions. No confirmatory claim depends on
   them.

5. Full claim hierarchy for exp5, declared before any number exists:

   | tier | claim | correction |
   |---|---|---|
   | PRIMARY | `history_gain` (k=646 vs k=322, same-dyad) | none (single primary) |
   | SECONDARY | `same_vs_other_dyad_at_646`; `positional_late_minus_early` | none — both pre-registered |
   | EXPLORATORY | all other rung comparisons, curve slopes, difference-in-differences, `within_person_late_minus_early` | Benjamini–Hochberg, α = 0.05, within the exploratory family |

   `within_person_late_minus_early` is exploratory on two independent grounds — the gate
   failure above and the fact that it is not in the frozen design — and is labelled
   underpowered wherever it appears.

6. **This amendment applies to exp5 only.** exp6–exp8 stay at n = 11 pending their own
   analyses.

**Note on scope relative to Amendment 1:** this amendment is larger than Amendment 1 — it
changes exp5's n, adds two analyses (the positional early/middle/late comparison and the
same-dyad-vs-other-dyad volume control) beyond the originally frozen 2-point curve, and
demotes one of them (Analysis 1B) to exploratory. Flagged here and in `PROGRESS.md`
prominently, not buried alongside routine amendments.

**Amendment 3 — 2026-08-22, pre-results, not post-hoc.**

1. Experiment 6's unit-of-analysis count drops from **n = 11 to n = 10**, excluding
   `sub19_sub22` for the same structural reason Amendment 1 gave for exp4 and Amendment 2
   gave for exp5, on the observer side: its observer rows span `dyad_trial_seq` [1,484]
   only, so the dyad's late tercile [647,968] — exp6's held-out block per `results/gate.json`'s
   frozen assumption — contains zero observer rows for this dyad. `sub19_sub22`'s 484
   observer rows remain in exp6's other-dyad/Universal training pool. `sub01_sub02` remains
   excluded from exp3–exp8 as already stated above.

2. **Exp6-specific gate re-check**, computed at the grain §16's design actually requires
   (the frozen `gate.json` exp6 row records `smallest_fold_total: 161,
   smallest_fold_minority: 70` — exp3's participant-grain numbers for the now-excluded
   `sub01_sub02`, and does not cover exp6's actual grains). `THRESHOLD_M = 60` and the
   Clause A ("smallest test-fold minority >= THRESHOLD_M") / Clause B ("at least 10 of 12
   dyads, or the equivalent ~83% fraction at reduced n — at n=10 that is >= 9") wording are
   taken **unchanged** from the original gate and are **not re-derived**, only re-applied
   at exp6's real grains, against the same ten dyads exp4's and exp5's amendments already
   established as tested:

   | analysis | test-fold grain | Clause A | Clause B | designation |
   |---|---|---|---|---|
   | 1A — positional early/middle/late (dyad-grain terciles) | 320–324, min minority 130 | PASS | 10/10 PASS | **CONFIRMATORY on power**; interpretation confounded with participant identity (the observer of a dyad's early block and the observer of its late block are different people — see the experiment module's Fact 2), stated in the headline, not a footnote |
   | 1B — within-observer early/middle/late (participant-grain terciles) | 162/161/161, min minority 45 | **FAIL** | **8/10 FAIL** | **EXPLORATORY — underpowered**, reported with CIs and explicitly labelled, per §7 |
   | 2 — §21 absolute-decodability permutation null | same bins as 1A | PASS | 10/10 PASS | **CONFIRMATORY** (a permutation test, not a §20 paired test) |

3. **Analysis 1B is demoted to exploratory / underpowered** by that check. The failing
   bins are `sub23` (45 minority trials, its middle within-observer bin) and `sub18` (59
   minority trials, its middle within-observer bin) — the same seq blocks as exp5's
   failing bins (`sub24`=45, `sub17`=59), with the identities mirrored because the
   observer of a given seq block is the partner of exp5's deceiver on that block. It is
   reported anyway, with confidence intervals and an explicit underpowered label, per §7's
   rule that a gate-failing analysis is demoted and reported, never silently dropped.

4. Exp6's confirmatory design is the **dyad-grain positional early/middle/late comparison,
   tested per §20** (sign test plus sign-flip permutation test on the median paired
   difference, n = 10). The **§21 label-shuffling absolute-decodability null is a
   pre-registered secondary claim** — §21 names Experiments 1, 2, 6, and 8 as the
   absolute-performance claims requiring this null; it is not satisfied by the §20 test,
   which answers a different question. The within-observer grain (Analysis 1B) and the
   middle-bin contrasts (`positional_middle_minus_early`, `positional_late_minus_middle`)
   are pre-declared exploratory.

5. Full claim hierarchy for exp6, declared before any number exists:

   | tier | claim | correction |
   |---|---|---|
   | PRIMARY | `observer_positional_late_minus_early` (§20 paired, n = 10) | none (single primary) |
   | SECONDARY | `observer_decodability_pooled` (§21 permutation null on the pooled observer bins) | none — pre-registered |
   | EXPLORATORY | `observer_within_person_late_minus_early`; `observer_positional_middle_minus_early`; `observer_positional_late_minus_middle`; the 30 per-bin §21 nulls | Benjamini–Hochberg, α = 0.05, within the exploratory family |

   `observer_within_person_late_minus_early` is exploratory on two independent grounds —
   the gate failure above and the fact that it is not in the frozen design — and is
   labelled underpowered wherever it appears.

6. **§16 reporting constraint.** No claim of subconscious or unconscious lie detection may
   be made from exp6 alone. A stronger claim than "statistical decodability of a signal"
   would require, at minimum, all four of: (a) behavioural corroboration — a link between
   the model's trial-level confidence and the observer's own `observer_guess`, over and
   above ground truth (not tested here — a different target, a different experiment); (b)
   effect size unambiguously above chance, not a couple of AUROC points; (c) consistency
   across a clear majority of the ten dyads, with a §20 p-value surviving the exploratory
   family correction; (d) robustness to mundane alternatives — at minimum surviving a
   control for behavioural columns (`outcome`, `points`, response timing), which exp6 does
   not build. Absent these, exp6's output may state decodability of a signal, never
   detection or knowledge by the observer.

7. **This amendment applies to exp6 only.** exp7 and exp8 stay at n = 11 pending their own
   analyses.

**Note on scope relative to Amendments 1 and 2:** exp6's n-reduction and gate re-check
follow the identical structural pattern as exp4's Amendment 1 and exp5's Amendment 2 — the
same excluded dyad (`sub19_sub22`), the same reason (zero rows in the late tercile), and
(per exp6's Fact 4) numerically identical gate arithmetic to exp5's, because the six
NaN-dropped rows are the same six trials in both the observer and deceiver roles. Flagged
here and in `PROGRESS.md` for visibility, consistent with how Amendments 1 and 2 were
flagged.


**Amendment 4 — 2026-08-22, pre-results, not post-hoc.**

1. **exp7's n is UNCHANGED at 11.** Amendment 3 stated that "exp7 and exp8 stay at
   n = 11 pending their own analyses"; exp7's analysis is now done and confirms it.
   Unlike exp4/exp5/exp6, exp7 does not use a within-dyad chronological block, so
   `sub19_sub22` — whose session-2 archive is unrecoverable and whose late tercile
   is therefore empty — is a perfectly valid held-out fold here, with 484 intact trials.
   `sub01_sub02` remains excluded from exp3–exp8 as already frozen. Tested dyads:

       sub03_sub06  sub04_sub05  sub07_sub08  sub09_sub10  sub11_sub12  sub13_sub14
       sub15_sub16  sub17_sub18  sub19_sub22  sub20_sub21  sub23_sub24

2. **exp7's fold structure resolves to leave-one-dyad-out at TRIAL grain.**
   `results/gate.json`'s `assumptions.exp7` left this open ("inherited from Exp2 (LODO)
   or Exp4 (within-dyad), depending on which comparison is being made"). It is resolved
   here, before any fit, to LODO: §17's question is about where information lives in
   brains generally, which §19 assigns to LODO; exp4 already owns the within-dyad
   question; and LODO is the only reading under which the pre-registered n = 11 is
   attainable. Rows are TRIALS, not participant-trials — exp7 must score both
   participants' data as one prediction per trial, which exp1–exp6 never needed.

3. **Gate re-check at exp7's own grain.** `gate.json`'s exp7 row records
   `smallest_fold_total = 161`, `smallest_fold_minority = 70`,
   `smallest_fold_dyad = "sub01_sub02"` — exp3's participant-grain chronological numbers
   for a dyad exp7 does not test. `THRESHOLD_M = 60` and the Clause A
   ("smallest test-fold minority >= THRESHOLD_M") / Clause B ("at least 10 of 12 included
   dyads individually clear it") wording are re-applied UNCHANGED at exp7's real grain:

   | analysis | test-fold grain | smallest fold | smallest minority | Clause A | Clause B | designation |
   |----------|-----------------|---------------|-------------------|----------|----------|-------------|
   | exp7 LODO | one dyad's trials | 484 (sub19_sub22) | 236 (sub19_sub22) | PASS | 11/11 PASS | CONFIRMATORY |

   exp7's CONFIRMATORY designation therefore stands on its own measured numbers, not on
   the stale row. `gate.json` is frozen and is NOT edited.

4. **Claim hierarchy, fixed before any number exists.**
   - PRIMARY (confirmatory, uncorrected): `both_brains_vs_deceiver_eeg`.
   - SECONDARY (confirmatory family, pre-registered, uncorrected):
     `observer_eeg_vs_deceiver_eeg`, `interbrain_vs_deceiver_eeg`,
     `eeg_plus_behavioral_vs_deceiver_eeg`.
   - EXPLORATORY (Benjamini-Hochberg, alpha = 0.05, within family): the remaining six
     pairwise comparisons among the five input sets.
   - Per-input-set absolute AUROCs are DESCRIPTIVE, never a test. §21's permutation-null
     requirement names exp1/exp2/exp6/exp8, not exp7; every exp7 claim is a comparison,
     for which §20's sign-flip test is the correct null.
   - `supported` for the primary is set by the pre-fixed rule
     `median_delta > 0 AND sign_test_p < 0.05 AND permutation_p < 0.05`; exploratory
     entries report BH-adjusted p-values and leave `supported` null.

5. **Two behavioural columns are excluded from the `eeg_plus_behavioral` input set**, for
   reasons measured before any fit: `pinfo_bart_score` is a per-participant constant
   (1-2 distinct values per dyad) and is therefore a participant-identity proxy of the
   kind §19 exists to exclude; `trials_so_far` correlates 1.000 with `dyad_trial_seq` and
   is the chronological position index under another name, which would let the
   behavioural set win on a positional signal that exp5 already owns. `round` is
   likewise not added. `outcome` remains excluded from every model as already frozen.

6. **This amendment applies to exp7 only.** exp8 stays at n = 11 pending its own analysis.
